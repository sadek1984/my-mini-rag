from fastapi import FastAPI, APIRouter, status, Request, Depends
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

class LocalAnswerRequest(BaseModel):
    query: str
    n_results: Optional[int] = 100


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
async def index_local_data(request: Request, project_id: int, index_request: LocalIndexRequest):
    """Index local Excel files for a project"""
    try:
        # Check if GEMINI_API_KEY exists
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return JSONResponse(
                status_code=400,
                content={
                    "signal": "GEMINI_API_KEY_MISSING",
                    "message": "Please set GEMINI_API_KEY environment variable"
                }
            )
        
        # Initialize enhanced controller
        enhanced_controller = EnhancedNLPController(
            vectordb_client=request.app.vectordb_client,
            generation_client=request.app.generation_client,
            embedding_client=request.app.embedding_client,
            template_parser=request.app.template_parser,
            gemini_api_key=gemini_api_key
        )
        
        # Process local Excel files
        documents, metadata = enhanced_controller.process_local_excel_files(
            data_folder_path=index_request.data_folder_path,
            months=index_request.months
        )
        
        if not documents:
            return JSONResponse(
                status_code=400,
                content={
                    "signal": "NO_LOCAL_DATA_FOUND",
                    "message": f"No Excel files found in {index_request.data_folder_path}",
                    "searched_months": index_request.months or "all default months"
                }
            )
        # --- ADD YOUR DEBUG CODE HERE ---
        # This is the perfect place to inspect the documents
        print(f"--- DEBUG: Inspecting Documents ---")
        print(f"Found {len(documents)} documents to index.")
        if documents:
            print("Sample of the first document:")
            print(f"Content: {documents[0][:500]}...") # Print the first 500 characters
            print(f"Metadata: {metadata[0]}")
        print(f"--- END DEBUG ---")

        # Create ChromaDB collection
        collection = enhanced_controller.create_local_collection(
            project_id=project_id,
            documents=documents,
            metadata_list=metadata,
            do_reset=index_request.do_reset
        )
        
        return JSONResponse(
            content={
                "signal": "LOCAL_INDEX_SUCCESS",
                "indexed_documents": len(documents),
                "project_id": project_id,
                "collection_name": f"local_project_{project_id}",
                "data_folder": index_request.data_folder_path
            }
        )
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "signal": "LOCAL_INDEX_ERROR",
                "error": str(e)
            }
        )

@nlp_router.post("/local/answer/{project_id}")
async def local_answer_endpoint(request: Request, project_id: int, answer_request: LocalAnswerRequest):
    """Answer queries using locally indexed data"""
    try:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return JSONResponse(
                status_code=400,
                content={
                    "signal": "GEMINI_API_KEY_MISSING",
                    "message": "Please set GEMINI_API_KEY environment variable"
                }
            )
        
        # Initialize enhanced controller
        enhanced_controller = EnhancedNLPController(
            vectordb_client=request.app.vectordb_client,
            generation_client=request.app.generation_client,
            embedding_client=request.app.embedding_client,
            template_parser=request.app.template_parser,
            gemini_api_key=gemini_api_key
        )
        
        # Get answer from local data
        result = await enhanced_controller.answer_with_local_data(
            project_id=project_id,
            query=answer_request.query,
            n_results=answer_request.n_results
        )
        
        if not result.get("success", False):
            return JSONResponse(
                status_code=400,
                content={
                    "signal": "LOCAL_ANSWER_ERROR",
                    "error": result.get("error", "Unknown error occurred")
                }
            )
        
        return JSONResponse(
            content={
                "signal": "LOCAL_ANSWER_SUCCESS",
                "answer": result["answer"],
                "full_prompt": result.get("full_prompt", ""),
                "retrieved_docs_count": result.get("retrieved_docs_count", 0),
                "collection_total": result.get("collection_total", 0),
                "project_id": project_id
            }
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "signal": "LOCAL_ANSWER_ERROR",
                "error": str(e)
            }
        )
    


# Add this debug endpoint to check collection status
@nlp_router.get("/local/debug/{project_id}")
async def debug_collection(request: Request, project_id: int):
    enhanced_controller = EnhancedNLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        template_parser=request.app.template_parser,
        gemini_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    collection_name = f"local_project_{project_id}"
    try:
        collection = enhanced_controller.chroma_client.get_collection(collection_name)
        return {"collection_exists": True, "count": collection.count()}
    except:
        return {"collection_exists": False, "available": [c.name for c in enhanced_controller.chroma_client.list_collections()]}