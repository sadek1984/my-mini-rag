from fastapi import FastAPI, APIRouter, status, Request, Depends, Query
from fastapi.responses import JSONResponse
from routes.schemes.nlp import PushRequest, SearchRequest
from models.ProjectModel import ProjectModel
from models.ChunkModel import ChunkModel
from controllers import NLPController, EnhancedNLPController
from models import ResponseSignal
from tqdm.auto import tqdm
from controllers.ProjectControllers import ProjectControllers
from controllers.ProcessControllers import DirectPesticideController
from pydantic import BaseModel
from typing import Optional, List
import os
from fastapi.responses import StreamingResponse # <-- Add this import
from routes.schemes.nlp import PostLocalQuery, PostLocalIndex, PostDocQuery, PostDocIndex
from models.risk_assessment_models import LocalAnswerRequest
from pydantic import Field
from models.risk_assessment_models import RiskAssessmentSummary, PopulationType
from datetime import datetime



import logging

logger = logging.getLogger('uvicorn.error')

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1", "nlp"],
)

class LocalIndexRequest(BaseModel):
    data_folder_path: str
    months: Optional[List[str]] = None
    do_reset: Optional[bool] = False
    year: str


class LocalAnswerRequest(BaseModel):
    query: str
    n_results: Optional[int] = 100
    llm_provider: str = Field(default="openai")
    prompt_template: Optional[str] = Field(default="en")
    
    # Add the missing field and population type for the assessment
    include_risk_assessment: Optional[bool] = Field(default=False, description="Flag to run health risk assessment on the results")
    population_type: Optional[PopulationType] = Field(default=PopulationType.ADULT_AVERAGE, description="Target population for risk assessment")



@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: int, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    chunk_model = await ChunkModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value
            }
        )
    
    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0

     # create collection if not exists
    collection_name = nlp_controller.create_collection_name(project_id=project.project_id)

    _ = await request.app.vectordb_client.create_collection(
        collection_name=collection_name,
        embedding_size=request.app.embedding_client.embedding_size,
        do_reset=push_request.do_reset,
    )

    # setup batching
    total_chunks_count = await chunk_model.get_total_chunks_count(project_id=project.project_id)
    pbar = tqdm(total=total_chunks_count, desc="Vector Indexing", position=0)
    # Add this debugging line right after
    print(f"DEBUG: Found {(total_chunks_count)} chunks in the database for project {project_id}.")


    while has_records:
        page_chunks = await chunk_model.get_poject_chunks(project_id=project.project_id, page_no=page_no)
        if len(page_chunks):
            page_no += 1
        
        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break 

        chunks_ids = [c.chunk_id for c in page_chunks]
        idx += len(page_chunks)
        
        is_inserted = await nlp_controller.index_into_vector_db(
            project=project,
            chunks=page_chunks,
             chunks_ids=chunks_ids
        )

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value
                }
            )

        pbar.update(len(page_chunks))
        inserted_items_count += len(page_chunks)
        
    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )
@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: str):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser, 
    )

    collection_info = await nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )
    

@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: int, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )
    

    results = await nlp_controller.search_vector_db_collection(
        project=project, text=search_request.text, limit=search_request.limit
    )

    if not results:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value
                }
            )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": [ result.dict()  for result in results ]
        }
    )

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: int, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
    )

    answer, full_prompt, chat_history = await nlp_controller.answer_rag_question(
        project=project,
        query=search_request.text,
        limit=search_request.limit,
    )

    if not answer:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.RAG_ANSWER_ERROR.value
                }
        )
    
    return JSONResponse(
        content={
            "signal": ResponseSignal.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history
        }
    )

    

# 4. Add this simple endpoint
@nlp_router.post("/simple/answer/{project_id}")
async def simple_answer(request: Request, project_id: int, search_request: SearchRequest):
    """
    Enhanced simple endpoint with better formatting
    """
    try:
        project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
        project = await project_model.get_project_or_create_one(project_id=project_id)
        project_path = ProjectControllers().get_project_path(project_id=str(project_id))
        
        # Use enhanced direct approach
        direct_controller = DirectPesticideController(project_path)
        answer = direct_controller.answer_query(
            query=search_request.text,
            generation_client=request.app.generation_client
        )
        
        return JSONResponse(content={
            "signal": "success",
            "answer": answer,
            "approach": "enhanced_direct_processing",
            "query": search_request.text
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"signal": "error", "message": str(e)}
        )

# 4. Quick Data Validation Helper
def validate_and_clean_pesticide_data(raw_data: str) -> str:
    """
    Quick validation and cleaning of pesticide data output
    """
    lines = raw_data.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and separators
        if not line or line.startswith('---') or line.startswith('==='):
            continue
        
        # Clean up table formatting
        if '|' in line:
            # Ensure proper table formatting
            parts = [part.strip() for part in line.split('|')]
            if len(parts) > 2:  # Valid table row
                cleaned_line = '| ' + ' | '.join(parts[1:-1]) + ' |'
                cleaned_lines.append(cleaned_line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

@nlp_router.post("/local/index/{project_id}")
async def index_local_data(
    request: Request, 
    project_id: int, 
    index_request: LocalIndexRequest
):
    """Index local Excel files for a specific year"""
    try:
        # Validate year
        if index_request.year not in ["2022", "2023", "2024", "2025"]:
            return JSONResponse(
                status_code=400,
                content={
                    "signal": "INVALID_YEAR",
                    "message": f"Year must be one of: 2022, 2023, 2024, 2025. Got: {index_request.year}"
                }
            )
        
        # Check OpenAI API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return JSONResponse(
                status_code=400,
                content={
                    "signal": "OPENAI_API_KEY_MISSING",
                    "message": "Please set OPENAI_API_KEY environment variable"
                }
            )
        
        # Initialize controller
        enhanced_controller = EnhancedNLPController(
            vectordb_client=request.app.vectordb_client,
            generation_client=request.app.generation_client,
            embedding_client=request.app.embedding_client,
            template_parser=request.app.template_parser,
            gemini_api_key=None
        )
        
        # *** FIX: Construct the FULL path including the year ***
        year_folder_path = f"{index_request.data_folder_path}/{index_request.year}"
        
        print(f"=== INDEXING DEBUG ===")
        print(f"Year: {index_request.year}")
        print(f"Base path: {index_request.data_folder_path}")
        print(f"Year folder path: {year_folder_path}")
        print("=" * 30)
        
        # Process Excel files - pass the YEAR-SPECIFIC path
        documents, metadata = enhanced_controller.process_local_excel_files(
            data_folder_path=year_folder_path,  # <-- This should be /app/data/2022
            months=index_request.months
        )
        
        if not documents:
            return JSONResponse(
                status_code=400,
                content={
                    "signal": "NO_LOCAL_DATA_FOUND",
                    "message": f"No Excel files found in {year_folder_path}",
                    "year": index_request.year,
                    "searched_path": year_folder_path
                }
            )
        
        print(f"Found {len(documents)} documents to index")
        if documents:
            print(f"Sample: {documents[0][:200]}...")
        
        # Create collection
        collection = enhanced_controller.create_local_collection(
            project_id=project_id,
            documents=documents,
            metadata_list=metadata,
            do_reset=index_request.do_reset,
            year=index_request.year
        )
        
        return JSONResponse(
            content={
                "signal": "LOCAL_INDEX_SUCCESS",
                "indexed_documents": len(documents),
                "project_id": project_id,
                "year": index_request.year,
                "collection_name": f"local_project_{project_id}_year_{index_request.year}",
                "data_folder": year_folder_path,
                "indexed_at": datetime.now().isoformat()
            }
        )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "signal": "LOCAL_INDEX_ERROR",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
@nlp_router.post("/local/answer/{project_id}")
async def local_answer_endpoint(request: Request, project_id: int,
                               answer_request: LocalAnswerRequest,
                               year: Optional[str] = Query(default=None)):
    """Answer queries using locally indexed data with Gemini"""
    try:
        # Get year from query parameter
        target_year = year or str(datetime.now().year)
        
        # Get GEMINI API key (you're using Gemini, not OpenAI!)
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return JSONResponse(
                status_code=400,
                content={
                    "signal": "GEMINI_API_KEY_MISSING",
                    "message": "Please set GEMINI_API_KEY environment variable"
                }
            )

        # Initialize controller WITH Gemini key
        enhanced_controller = EnhancedNLPController(
            vectordb_client=request.app.vectordb_client,
            generation_client=request.app.generation_client,
            embedding_client=request.app.embedding_client,
            template_parser=request.app.template_parser,
            gemini_api_key=gemini_api_key  # Pass the actual key!
        )
        
        # Get answer from local data
        result = await enhanced_controller.answer_with_local_data(
            project_id=project_id,
            query=answer_request.query,
            n_results=answer_request.n_results,
            include_risk_assessment=answer_request.include_risk_assessment,
            population_type=answer_request.population_type,  # Remove .value
            year=target_year
        )

        if not result.get("success", False):
            return JSONResponse(
                status_code=400,
                content={
                    "signal": "LOCAL_ANSWER_ERROR",
                    "error": result.get("error", "Unknown error"),
                    "debug_info": result.get("debug_info", {})
                }
            )

        # Return response
        return JSONResponse(content={
            "signal": "LOCAL_ANSWER_SUCCESS",
            "answer": result["answer"],
            "data_frame": result.get("data_frame", []),
            "columns": result.get("columns", []),
            "retrieved_docs_count": result.get("retrieved_docs_count", 0),
            "full_prompt": result.get("full_prompt", ""),
            "collection_total": result.get("collection_total", 0),
            "risk_assessment": result.get("risk_assessment"),
            "project_id": project_id,
            "year": target_year
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "signal": "LOCAL_ANSWER_ERROR",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
    
# Also add a simple test endpoint to check raw ChromaDB data
@nlp_router.get("/local/raw-data/{project_id}")
async def get_raw_data_sample(request: Request, project_id: int):
    """Get raw sample data from ChromaDB to see actual content"""
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        enhanced_controller = EnhancedNLPController(gemini_api_key=gemini_api_key)
        collection_name = f"local_project_{project_id}"
        
        collection = enhanced_controller.chroma_client.get_collection(
            name=collection_name,
            embedding_function=enhanced_controller.embed_fn
        )
        
        # Get first few documents without querying
        raw_data = collection.get(limit=5)
        
        sample_data = {
            "collection_count": collection.count(),
            "sample_documents": []
        }
        
        for i, (doc, metadata) in enumerate(zip(
            raw_data.get("documents", []),
            raw_data.get("metadatas", [])
        )):
            sample_data["sample_documents"].append({
                "index": i,
                "metadata": metadata,
                "content_length": len(doc),
                "content_preview": doc[:1000],  # First 1000 chars
                "contains_key_terms": {
                    "pepper": "pepper" in doc.lower() or "فلفل" in doc,
                    "pesticide": "pesticide" in doc.lower() or "مبيد" in doc,
                    "result": "result" in doc.lower() or "نتيجة" in doc
                }
            })
        
        return JSONResponse(content=sample_data)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    

@nlp_router.post(
    "/local/query/export/{project_id}",
    summary="Query local data and get the result as an Excel file",
    tags=["Local Data Processing"],
    response_description="An Excel file containing the query results.",
)
async def query_local_data_export(project_id: int, data: PostLocalQuery):
    """
    Sends a query to the local data collection and returns a structured Excel file.
    """
    try:
        # --- THIS IS THE FIX ---
        # 1. Get the Gemini API Key, which is required by the controller.
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return JSONResponse(
                status_code=500,
                content={"detail": "GEMINI_API_KEY is not configured on the server."}
            )
            
        # 2. Create an instance of the CORRECT controller for local data.
        enhanced_controller = EnhancedNLPController(gemini_api_key=gemini_api_key)
        
        # 3. Call the method on the created instance.
        excel_buffer = await enhanced_controller.answer_and_export_local_data(
            project_id=project_id,
            query=data.query,
            n_results=data.n_results
        )
        
        if excel_buffer is None:
            return JSONResponse(
                status_code=400, 
                content={"detail": "Could not generate Excel file. The model's response may not have been in a parsable table format."}
            )
        
        headers = {
            'Content-Disposition': 'attachment; filename="query_results.xlsx"'
        }
        
        return StreamingResponse(
            excel_buffer, 
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers
        )
    except Exception as e:
        logger.error(f"Error during Excel export: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": f"An internal error occurred: {str(e)}"})



@nlp_router.get("/local/debug-path/{year}")
async def debug_data_path(year: str):
    """Debug endpoint to check folder structure"""
    import os
    from pathlib import Path
    
    base_path = f"/app/data/{year}"
    
    debug_info = {
        "year": year,
        "base_path": base_path,
        "exists": os.path.exists(base_path),
        "is_dir": os.path.isdir(base_path) if os.path.exists(base_path) else False,
        "contents": []
    }
    
    if os.path.exists(base_path):
        try:
            # List all items in the directory
            for item in os.listdir(base_path):
                item_path = os.path.join(base_path, item)
                item_info = {
                    "name": item,
                    "is_dir": os.path.isdir(item_path),
                    "is_file": os.path.isfile(item_path),
                }
                
                if os.path.isdir(item_path):
                    # If it's a directory, list its contents
                    try:
                        subcontents = os.listdir(item_path)
                        item_info["files_count"] = len([f for f in subcontents if f.endswith('.xlsx')])
                        item_info["sample_files"] = [f for f in subcontents if f.endswith('.xlsx')][:3]
                    except Exception as e:
                        item_info["error"] = str(e)
                
                debug_info["contents"].append(item_info)
        except Exception as e:
            debug_info["error"] = str(e)
    
    return JSONResponse(content=debug_info)




# @nlp_router.post("/local/index/{project_id}")
# async def index_local_data(request: Request, project_id: int, index_request: LocalIndexRequest):
#     """Index local Excel files for a project"""
#     try:
#         # Check if GEMINI_API_KEY exists
#         gemini_api_key = os.getenv("GEMINI_API_KEY")
#         if not gemini_api_key:
#             return JSONResponse(
#                 status_code=400,
#                 content={
#                     "signal": "GEMINI_API_KEY_MISSING",
#                     "message": "Please set GEMINI_API_KEY environment variable"
#                 }
#             )
        
#         # Initialize enhanced controller
#         enhanced_controller = EnhancedNLPController(
#             vectordb_client=request.app.vectordb_client,
#             generation_client=request.app.generation_client,
#             embedding_client=request.app.embedding_client,
#             template_parser=request.app.template_parser,
#             gemini_api_key=gemini_api_key
#         )
        
#         # Process local Excel files
#         documents, metadata = enhanced_controller.process_local_excel_files(
#             data_folder_path=index_request.data_folder_path,
#             months=index_request.months
#         )
        
#         if not documents:
#             return JSONResponse(
#                 status_code=400,
#                 content={
#                     "signal": "NO_LOCAL_DATA_FOUND",
#                     "message": f"No Excel files found in {index_request.data_folder_path}",
#                     "searched_months": index_request.months or "all default months"
#                 }
#             )
#         # --- ADD YOUR DEBUG CODE HERE ---
#         # This is the perfect place to inspect the documents
#         print(f"--- DEBUG: Inspecting Documents ---")
#         print(f"Found {len(documents)} documents to index.")
#         if documents:
#             print("Sample of the first document:")
#             print(f"Content: {documents[0][:500]}...") # Print the first 500 characters
#             print(f"Metadata: {metadata[0]}")
#         print(f"--- END DEBUG ---")

#         # Create ChromaDB collection
#         collection = enhanced_controller.create_local_collection(
#             project_id=project_id,
#             documents=documents,
#             metadata_list=metadata,
#             do_reset=index_request.do_reset
#         )
        
#         return JSONResponse(
#             content={
#                 "signal": "LOCAL_INDEX_SUCCESS",
#                 "indexed_documents": len(documents),
#                 "project_id": project_id,
#                 "collection_name": f"local_project_{project_id}",
#                 "data_folder": index_request.data_folder_path
#             }
#         )
            
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "signal": "LOCAL_INDEX_ERROR",
#                 "error": str(e)
#             }
#         )


# @nlp_router.post("/local/answer/{project_id}")
# async def local_answer_endpoint(request: Request, project_id: int, answer_request: LocalAnswerRequest):
#     """Answer queries using locally indexed data - Enhanced with debugging"""
#     try:
#         gemini_api_key = os.getenv("GEMINI_API_KEY")
#         if not gemini_api_key:
#             return JSONResponse(
#                 status_code=400,
#                 content={
#                     "signal": "GEMINI_API_KEY_MISSING",
#                     "message": "Please set GEMINI_API_KEY environment variable"
#                 }
#             )
        
#         # Log the incoming request
#         print(f"=== LOCAL ANSWER REQUEST ===")
#         print(f"Project ID: {project_id}")
#         print(f"Query: {answer_request.query}")
#         print(f"N Results: {answer_request.n_results}")
#         print("=" * 30)
        
#         # Initialize enhanced controller
#         enhanced_controller = EnhancedNLPController(
#             vectordb_client=request.app.vectordb_client,
#             generation_client=request.app.generation_client,
#             embedding_client=request.app.embedding_client,
#             template_parser=request.app.template_parser,
#             gemini_api_key=gemini_api_key
#         )
        
#         # Get answer from local data
#         result = await enhanced_controller.answer_with_local_data(
#             project_id=project_id,
#             query=answer_request.query,
#             n_results=answer_request.n_results
#         )
        
#         # Log the result
#         print(f"=== LOCAL ANSWER RESULT ===")
#         print(f"Success: {result.get('success', False)}")
#         if result.get('success'):
#             print(f"Answer length: {len(result.get('answer', ''))}")
#             print(f"Retrieved docs: {result.get('retrieved_docs_count', 0)}")
#             print(f"Debug info: {result.get('debug_info', {})}")
#         else:
#             print(f"Error: {result.get('error', 'Unknown error')}")
#         print("=" * 30)
        
#         if not result.get("success", False):
#             return JSONResponse(
#                 status_code=400,
#                 content={
#                     "signal": "LOCAL_ANSWER_ERROR",
#                     "error": result.get("error", "Unknown error occurred"),
#                     "debug_info": result.get("debug_info", {})
#                 }
#             )
        
#         return JSONResponse(
#             content={
#                 "signal": "LOCAL_ANSWER_SUCCESS",
#                 "answer": result["answer"],
#                 "full_prompt": result.get("full_prompt", ""),
#                 "retrieved_docs_count": result.get("retrieved_docs_count", 0),
#                 "collection_total": result.get("collection_total", 0),
#                 "project_id": project_id,
#                 "debug_info": result.get("debug_info", {})
#             }
#         )
        
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return JSONResponse(
#             status_code=500,
#             content={
#                 "signal": "LOCAL_ANSWER_ERROR",
#                 "error": str(e),
#                 "traceback": traceback.format_exc()
#             }
#         )

