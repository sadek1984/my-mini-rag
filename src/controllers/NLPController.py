import logging
import json
from typing import List, Dict, Tuple, Optional, Any
import os
import pandas as pd
import numpy as np
from pathlib import Path
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
from google.api_core import retry
import logging
from tqdm import tqdm
import io

from .BaseControllers import BaseControllers
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnum import DocumentTypeEnum

import re
from bidi.algorithm import get_display

# استيراد الـ Parser المتخصص الجديد لتقارير المبيدات
from stores.llm.templates.pesticide_templates import PesticideTemplateParser

from utils.data_cleaning import clean_pesticide_data, load_and_prepare_excel, format_rows_as_text

from io import BytesIO # <-- Add this import
from utils.response_parser import parse_llm_response_to_df
from services.risk_assessment_service import RiskAssessmentService
from models.risk_assessment_models import PopulationType, RiskAssessmentSummary





class NLPController(BaseControllers):
    """
    Enhanced NLP Controller that integrates pesticide-specific query analysis
    with improved document retrieval and filtering logic.
    """

    def __init__(self, vectordb_client, generation_client, 
                 embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        
        # الـ Parser العام للـ RAG القياسي
        self.template_parser = template_parser
        # الـ Parser المتخصص لتقارير المبيدات
        self.pesticide_template_parser = PesticideTemplateParser()
        
        self.logger = logging.getLogger(__name__)

    # --- الدوال المتوافقة مع المشروع الأصلي (بدون تغيير) ---

    def create_collection_name(self, project_id: str):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()
    
    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)
    
    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(collection_name=collection_name)
        return json.loads(json.dumps(collection_info, default=lambda x: x.__dict__))
    
    async def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        collection_name = self.create_collection_name(project_id=project.project_id)
        texts = [c.chunk_text for c in chunks]
        metadata = [c.chunk_metadata for c in chunks]
        vectors = self.embedding_client.embed_text(text=texts, 
                                                  document_type=DocumentTypeEnum.DOCUMENT.value)
        await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )
        await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )
        return True

    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 100):
        collection_name = self.create_collection_name(project_id=project.project_id)
        vectors = self.embedding_client.embed_text(text=text, 
                                                 document_type=DocumentTypeEnum.QUERY.value)
        if not vectors or not vectors[0]: 
            return False
        
        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vectors[0],
            limit=limit
        )
        return results if results else False
    
    # --- الدوال الجديدة والمنطق المدمج المحسن ---

    def analyze_query_intent(self, query: str) -> dict:
        """
        Enhanced query intent analysis with better keyword detection.
        """
        query_lower = query.lower()
        intent_info = {
            'type': 'general', 
            'filters': {},
            'keywords': [],
            'search_terms': []
        }

        # Intent detection with Arabic and English support
        if any(term in query_lower for term in ['غير مطابق', 'non-compliant', 'non compliant', 'failed', 'exceed']):
            intent_info['type'] = 'non_compliant'
            intent_info['filters']['result_contains'] = 'غير مطابق'
            intent_info['keywords'].extend(['غير مطابق', 'non-compliant'])
        elif any(term in query_lower for term in ['مطابق', 'compliant', 'passed', 'acceptable']) and 'غير مطابق' not in query_lower:
            intent_info['type'] = 'compliant'
            intent_info['filters']['result_contains'] = 'مطابق'
            intent_info['keywords'].extend(['مطابق', 'compliant'])
        elif any(term in query_lower for term in ['summary', 'ملخص', 'تقرير', 'report', 'overview']):
            intent_info['type'] = 'summary'

        # Enhanced sample type detection
        sample_mappings = {
            'pepper': ['pepper', 'فلفل', 'bell pepper', 'capsicum'],
            'tomato': ['tomato', 'طماطم', 'tomatoes'],
            'cucumber': ['cucumber', 'خيار', 'cucumbers'],
            'eggplant': ['eggplant', 'باذنجان', 'aubergine'],
            'lettuce': ['lettuce', 'خس', 'salad'],
            'carrot': ['carrot', 'جزر', 'carrots']
        }
        
        for sample_type, keywords in sample_mappings.items():
            if any(keyword in query_lower for keyword in keywords):
                intent_info['filters']['sample_name_contains'] = sample_type
                intent_info['keywords'].extend(keywords)
                break

        # Build search terms for better vector retrieval
        intent_info['search_terms'] = intent_info['keywords'].copy()
        if intent_info['type'] != 'general':
            intent_info['search_terms'].append(query)
        
        return intent_info

    def filter_documents_by_intent(self, documents: List, intent_info: dict) -> List:
        """
        Enhanced document filtering based on intent analysis.
        """
        if not documents or intent_info['type'] == 'general':
            return documents
        
        filtered_docs = []
        
        for doc in documents:
            metadata = getattr(doc, 'metadata', {})
            content = getattr(doc, 'text', getattr(doc, 'page_content', ''))
            
            # Skip if not pesticide data
            if metadata.get('record_type') != 'pesticide_reading':
                continue
            
            match = True
            
            # Filter by result compliance
            if 'result_contains' in intent_info['filters']:
                expected_result = intent_info['filters']['result_contains']
                actual_result = str(metadata.get('result', '')).lower()
                content_lower = content.lower()
                
                if expected_result.lower() not in actual_result and expected_result.lower() not in content_lower:
                    match = False
            
            # Filter by sample name
            if 'sample_name_contains' in intent_info['filters']:
                expected_sample = intent_info['filters']['sample_name_contains']
                actual_sample = str(metadata.get('sample_name', '')).lower()
                content_lower = content.lower()
                
                # Check both metadata and content
                sample_found = False
                for keyword in sample_mappings.get(expected_sample, [expected_sample]):
                    if keyword.lower() in actual_sample or keyword.lower() in content_lower:
                        sample_found = True
                        break
                
                if not sample_found:
                    match = False
            
            if match:
                filtered_docs.append(doc)
        
        return filtered_docs

    async def enhanced_document_search(self, project: Project, query: str, intent_info: dict, limit: int = 10) -> List:
        """
        Enhanced document search with multiple strategies.
        """
        all_documents = []
        
        # Strategy 1: Direct query search
        direct_results = await self.search_vector_db_collection(project, query, limit)
        if direct_results:
            all_documents.extend(direct_results)
        
        # Strategy 2: Search with intent-specific terms
        for search_term in intent_info.get('search_terms', []):
            if search_term != query:  # Avoid duplicate searches
                term_results = await self.search_vector_db_collection(project, search_term, limit//2)
                if term_results:
                    all_documents.extend(term_results)
        
        # Strategy 3: Broader search for specific intents
        if intent_info['type'] in ['non_compliant', 'compliant']:
            broader_terms = ['pesticide', 'مبيد', 'analysis', 'تحليل', 'result', 'نتيجة']
            for term in broader_terms:
                broad_results = await self.search_vector_db_collection(project, term, limit//4)
                if broad_results:
                    all_documents.extend(broad_results)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        for doc in all_documents:
            doc_id = getattr(doc, 'id', id(doc))
            if doc_id not in seen:
                seen.add(doc_id)
                unique_docs.append(doc)
        
        return unique_docs

    

    def extract_structured_data(self, documents: list) -> list:
        """
        Extract structured data from documents for better processing
        """
        structured_data = []
        
        for doc in documents:
            content = getattr(doc, 'text', getattr(doc, 'page_content', ''))
            
            # Split content into lines and process each line
            lines = content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or 'sample code' in line.lower():
                    continue
                
                # Try to extract structured data from line
                data_parts = [part.strip() for part in line.split('\t') if part.strip()]
                
                # If we have enough parts, consider it a data row
                if len(data_parts) >= 4:
                    reading = {
                        'sample_code': data_parts[0] if len(data_parts) > 0 else '',
                        'sample_name': data_parts[1] if len(data_parts) > 1 else '',
                        'pesticide_name': data_parts[2] if len(data_parts) > 2 else '',
                        'limits': data_parts[3] if len(data_parts) > 3 else '',
                        'device_reading': data_parts[4] if len(data_parts) > 4 else '',
                        'result': data_parts[5] if len(data_parts) > 5 else '',
                        'raw_line': line
                    }
                    
                    # Add classification flags
                    reading['is_non_compliant'] = 'غير مطابق' in reading['result']
                    reading['is_compliant'] = 'مطابق' in reading['result'] and 'غير' not in reading['result']
                    reading['is_pepper'] = any(pepper_term in reading['sample_name'].lower() 
                                            for pepper_term in ['فلفل', 'pepper', 'فلف'])
                    
                    structured_data.append(reading)
        
        return structured_data

    def format_table_output(self, data: list, query_type: str = 'general') -> str:
        """
        Format data into clean table output
        """
        if not data:
            return "No matching data found."
        
        # Filter data based on query type
        if 'غير مطابق' in query_type or 'non-compliant' in query_type.lower():
            filtered_data = [item for item in data if item.get('is_non_compliant', False)]
        elif 'pepper' in query_type.lower() or 'فلفل' in query_type:
            filtered_data = [item for item in data if item.get('is_pepper', False)]
        elif 'pesticide' in query_type.lower() or 'مبيد' in query_type:
            filtered_data = data  # Show all for pesticide queries
        else:
            filtered_data = data
        
        if not filtered_data:
            return f"No data found matching the criteria: {query_type}"
        
        # Create formatted table
        output = []
        output.append("**Pesticide Analysis Results:**\n")
        
        # Table header
        output.append("| كود العينة | اسم العينة | اسم المبيد | الحدود | قراءة الجهاز | النتيجة |")
        output.append("|----------|----------|---------|-------|----------|-------|")
        
        # Table rows
        for item in filtered_data[:20]:  # Limit to 20 rows for readability
            row = f"| {item.get('sample_code', 'N/A')[:15]} | {item.get('sample_name', 'N/A')[:15]} | {item.get('pesticide_name', 'N/A')[:20]} | {item.get('limits', 'N/A')[:10]} | {item.get('device_reading', 'N/A')[:15]} | {item.get('result', 'N/A')[:10]} |"
            output.append(row)
        
        # Add summary
        output.append(f"\n**Summary:**")
        output.append(f"- Total records found: {len(filtered_data)}")
        output.append(f"- Records shown: {min(20, len(filtered_data))}")
        
        if len(filtered_data) > 20:
            output.append("- *Showing first 20 records*")
        
        return "\n".join(output)


    async def answer_rag_question(self, project: Project, query: str, limit: int = 100) -> tuple[str, str, list]:
        """Answer query using RAG - COLAB APPROACH"""
        print(f"Processing query: {query}")
        
        # Query vector store
        retrieved_documents = await self.search_vector_db_collection(project, query, limit)
        
        # Correct way to check if a list is empty or invalid
        if not retrieved_documents:
            return "No relevant documents found.", "", []
        
        # Extract structured data
        structured_data = self.extract_structured_data(retrieved_documents)
        
        if not structured_data:
            # Fallback to original approach
            combined_content = []
            for doc in retrieved_documents:
                content = getattr(doc, 'text', getattr(doc, 'page_content', ''))
                combined_content.append(content)
            
            combined_passage = " ".join(combined_content)
            
            # Clean up for prompt
            passage_oneline = combined_passage.replace("\n", " ")
            query_oneline = query.replace("\n", " ")
            
            # Create prompt (EXACT COLAB STYLE)
            prompt = f"""You are a helpful assistant summarizing pesticide readings.
    List readings where النتيجة (result) is 'غير مطابق' for pepper in a structured format.
    QUESTION: {query_oneline}
    PASSAGE: {passage_oneline}
    ⚠️ Important:
    - Do NOT summarize, list every individual reading.
    - Separate values clearly structured.
    - If there are multiple values, provide all of them.
    Example output:
    - كود العينة sample code, اسم العينة Sample Name, اسم المبيد Pesticide Name, الحدود Limits, قراءة الجهاز Reading of device, النتيجة result
    """
            
            # Generate answer
            try:
                answer = self.generation_model.generate_content(prompt)
                return answer.text, prompt, []  # Return answer, prompt, empty chat_history
            except Exception as e:
                return f"Error generating response: {e}", prompt, []
        
        else:
            # Handle structured data case (you need to implement this logic)
            # For now, returning the structured data or processing it further
            try:
                # You might want to format structured_data here
                # This is a placeholder - implement based on your structured_data format
                prompt = f"Process this structured data: {structured_data}"
                answer = self.generation_model.generate_content(prompt)
                return answer.text, prompt, []  # Return answer, prompt, empty chat_history
            except Exception as e:
                return f"Error generating response: {e}", "", []
            
    # --- NEW METHOD FOR ISO 17025 ---
    async def answer_iso_question(
        self,
        project: dict,
        query: str,
        limit: int = 5,
    ) -> tuple[str, str, list]:
        """
        Answers a question using RAG with special metadata filtering for ISO documents.
        This method is the core of the intelligent search.
        """
        # 1. Create a metadata filter based on keywords in the user's question.
        metadata_filter: Dict[str, Any] = {}
        question_lower = query.lower()

        if "procedure" in question_lower or "sop" in question_lower:
            metadata_filter = {"document_type": "Procedure"}
        elif "quality manual" in question_lower or "policy" in question_lower:
            metadata_filter = {"document_type": "Quality Manual"}
        elif "work instruction" in question_lower or "how to" in question_lower:
            metadata_filter = {"document_type": "Work Instruction"}
        
        # 2. Perform similarity search using the vector DB
        # Note: We assume the search method is updated to accept a metadata_filter
        embedding = await self.embedding_client.embed_query(query)
        chunks = await self.vectordb_client.search(
            project_id=project.get("id"),
            embedding=embedding,
            limit=limit,
            metadata_filter=metadata_filter  # <-- The key improvement
        )

        if not chunks:
            return None, None, None

        # 3. Generate the prompt and get the answer from the LLM
        context = "\n---\n".join([chunk.get("content") for chunk in chunks])
        full_prompt = self.template_parser.get_prompt(
            context=context,
            query=query,
            template_name="rag_en" # Or your default template
        )
        answer = await self.generation_client.generate(full_prompt)
        
        # 4. Prepare sources from the retrieved chunks' metadata
        sources = list(set([
            chunk.get("metadata", {}).get('source_file', 'Unknown') 
            for chunk in chunks
        ]))
        
        return answer, full_prompt, sources

            


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function for ChromaDB using Gemini"""
    
    def __init__(self, api_key: str, model: str = "models/text-embedding-004"):
        genai.configure(api_key=api_key)
        self.model = model
        self.document_mode = True
    
    def __call__(self, input: Documents) -> Embeddings:
        embedding_task = "retrieval_document" if self.document_mode else "retrieval_query"
        retry_policy = {"retry": retry.Retry(predicate=retry.if_transient_error)}
        
        try:
            response = genai.embed_content(
                model=self.model,
                content=input,
                task_type=embedding_task,
                request_options=retry_policy,
            )
            return response["embedding"]
        except Exception as e:
            print(f"Embedding error: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * 768 for _ in input]


# ========== ENHANCED NLP CONTROLLER ==========
# class EnhancedNLPController:
#     """Enhanced NLP Controller with local data processing"""
    
#     def __init__(self, 
#                  vectordb_client=None,
#                  generation_client=None,
#                  embedding_client=None,
#                  template_parser=None,
#                  gemini_api_key: str = None):
        
#         # Keep existing clients for compatibility
#         self.vectordb_client = vectordb_client
#         self.generation_client = generation_client
#         self.embedding_client = embedding_client
#         self.template_parser = template_parser
        
#         # New ChromaDB setup for local data
#         self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
#         if self.gemini_api_key:
#             # Use PersistentClient to save data to disk
#             self.chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
#             self.embed_fn = GeminiEmbeddingFunction(
#                 api_key=self.gemini_api_key,
#                 model="models/text-embedding-004"
#             )
            
#             # Configure Gemini for generation
#             genai.configure(api_key=self.gemini_api_key)
#             self.generation_model = genai.GenerativeModel("gemini-1.5-flash")
#         else:
#             self.chroma_client = None
#             self.embed_fn = None
#             self.generation_model = None
        
#         self.logger = logging.getLogger(__name__)


import openai
from openai import OpenAI

class EnhancedNLPController:
    """Enhanced NLP Controller with OpenAI integration"""
    
    def __init__(self, 
                 vectordb_client=None,
                 generation_client=None,
                 embedding_client=None,
                 template_parser=None,
                 gemini_api_key: str = None):
        
        # Keep existing clients for compatibility
        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        
        # New ChromaDB setup for local data
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if self.gemini_api_key:
            # Use PersistentClient to save data to disk
            self.chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
            self.embed_fn = GeminiEmbeddingFunction(
                api_key=self.gemini_api_key,
                model="models/text-embedding-004"
            )

        # Use OpenAI instead of Gemini
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            # Initialize OpenAI client
            self.openai_client = OpenAI(api_key=openai_api_key)
            
            # Use ChromaDB with your existing embedding setup
            self.chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
            
            # Use your existing embedding function instead of Gemini
            # You'll need to implement an OpenAI embedding function or use your existing one
          
        else:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.logger = logging.getLogger(__name__)


    async def answer_and_export_local_data(self, project_id: int, query: str, n_results: int = 100) -> Optional[io.BytesIO]:
        """
        Answers a query using local ChromaDB data and returns the result as an Excel file in memory.
        (Simplified version)
        """
        self.logger.info(f"Received export request for project {project_id} (simplified logic)")
        # First, get the standard text-based answer from the existing method
        response_data = await self.answer_with_local_data(project_id, query, n_results)

        if not response_data.get("success"):
            self.logger.error("Failed to get a successful response from the LLM. Cannot generate Excel file.")
            return None

        answer_text = response_data.get("answer")
        if not answer_text:
            self.logger.error("LLM response was empty. Cannot generate Excel file.")
            return None
            
        # Use our robust utility to parse the markdown table into a DataFrame
        self.logger.info("Parsing LLM response into a DataFrame...")
        df = parse_llm_response_to_df(answer_text)

        if df is None or df.empty:
            self.logger.error("Failed to parse the LLM response into a structured table or the result was empty.")
            return None
        # --- NEW STEP: Apply calculations based on the query ---
        df = self._apply_query_calculations(df, query)
        if df.empty:
            self.logger.warning("DataFrame is empty after applying calculations. No data to export.")
            # Still return an empty file so the user knows the query yielded no results
            pass

        # Save the resulting DataFrame directly to an in-memory Excel file
        self.logger.info(f"Successfully created DataFrame with {len(df)} rows. Saving to Excel format.")
        output_buffer = io.BytesIO()
        df.to_excel(output_buffer, index=False, sheet_name="Query Results")
        output_buffer.seek(0)  # Rewind the buffer to the beginning

        return output_buffer






    ############### NEW METHODS FOR Arabic text direction ###############

    def fix_arabic_text_direction(self, text):
        """Simple Arabic text preservation without complex reshaping"""
        if not isinstance(text, str):
            return text
        
        # Just return the text as-is for now - no reshaping
        # The libraries are causing text to disappear
        return text

    def process_local_excel_files(self, data_folder_path: str, months: List[str] = None) -> tuple:
        """Process local Excel files from month folders"""
        if months is None:
            months = ["Jan","Feb", "March", "April", "May", "June",
                    "July", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        documents = []
        metadata_list = []
        data_path = Path(data_folder_path)
        
        if not data_path.exists():
            raise ValueError(f"Data folder not found: {data_path}")
        
        self.logger.info(f"Processing local Excel files from: {data_path}")
        
        for month_folder in months:
            month_path = data_path / month_folder
            
            if not month_path.exists():
                self.logger.warning(f"Month folder not found: {month_path}")
                continue
                
            self.logger.info(f"Processing month: {month_folder}")
            
            for file_path in month_path.glob("*.xlsx"):
            
                try:
                    # Go back to simple approach - just match Colab exactly
                    # df = pd.read_excel(file_path, skiprows=10)
                    
                    # 1. Use the robust function to load the Excel file into a DataFrame
                    df = load_and_prepare_excel(str(file_path))
                    
                    if df is None or df.empty:
                        self.logger.warning(f"File skipped, could not find valid headers or data: {file_path.name}")
                        continue
                    
                    # 2. Clean the DataFrame to remove rows with no pesticide data
                    cleaned_df = clean_pesticide_data(df)
                    
                    if cleaned_df.empty:
                        self.logger.warning(f"File skipped after cleaning (no valid pesticide rows): {file_path.name}")
                        continue

                    # 3. NEW: Format each valid row into a descriptive sentence
                    row_texts = format_rows_as_text(cleaned_df)
                    
                    # Add each sentence as a separate document to be indexed
                    for text in row_texts:
                        documents.append(text)
                        metadata_list.append({
                            "source_file": file_path.name,
                            "month": month_folder,
                            "file_path": str(file_path),
                            "data_type": "local_excel_row",
                        })
                    
                    self.logger.info(f"Processed: {file_path.name}, created {len(row_texts)} documents from rows.")

                except Exception as e:
                    self.logger.error(f"Critical error processing {file_path.name}: {e}")


        self.logger.info(f"Total local documents processed: {len(documents)}")
        return documents, metadata_list
        print(documents[0][:20])  # Print first 5 chars of first document for verification



    def create_local_collection(self, project_id: int, documents: List[str], 
                               metadata_list: List[Dict], do_reset: bool = False):
        """Create ChromaDB collection for local data"""
        if not self.chroma_client:
            raise ValueError("ChromaDB not initialized. Check GEMINI_API_KEY.")
        
        # Set to document mode for indexing
        self.embed_fn.document_mode = True
        
        collection_name = f"local_project_{project_id}"
        
        # Delete existing collection if reset requested
        if do_reset:
            try:
                self.chroma_client.delete_collection(name=collection_name)
                self.logger.info(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass
        
        # Create or get collection
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn
        )
        
        # Add documents in batches
        batch_size = 50
        for i in tqdm(range(0, len(documents), batch_size), desc="Indexing local data"):
            batch_docs = documents[i:i+batch_size]
            batch_metadata = metadata_list[i:i+batch_size]
            batch_ids = [f"doc_{j}" for j in range(i, i+len(batch_docs))]
            
            try:
                collection.add(
                    documents=batch_docs,
                    ids=batch_ids,
                    metadatas=batch_metadata
                )
            except Exception as e:
                self.logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
        
        final_count = collection.count()
        self.logger.info(f"Local collection created: {final_count} documents indexed")
        return collection


    ############### NEW METHODS FOR DataFrame calculations ###############
         ############### e.g. reading > (X * limit) ###############
    

    def find_column_robust(self, df: pd.DataFrame, keywords: List[str]) -> str | None:
        """
        Robustly finds a column in a DataFrame by searching for keywords.
        Handles variations like newlines, extra spaces, and mixed languages.
        """
        for col in df.columns:
            # Normalize the column name to make matching more reliable
            normalized_col = re.sub(r'\s+', ' ', str(col)).strip().lower()
            for keyword in keywords:
                if keyword.lower() in normalized_col:
                    return col
        return None

    def analyze_pesticide_contamination_levels(self, df: pd.DataFrame, multiplier_threshold: float = 1.0) -> pd.DataFrame:
        """
        Analyzes pesticide contamination levels by comparing reading values with limits.
        Filters for samples where reading > (multiplier_threshold * limit).
        """
        self.logger.error(f"DEBUG: Starting analyze_pesticide_contamination_levels with threshold {multiplier_threshold}")
        
        try:
            self.logger.info(f"Analyzing contamination levels with threshold: {multiplier_threshold}x the limit")
            self.logger.info(f"Input DataFrame shape: {df.shape}")
            
            # Find the required columns using robust search
            self.logger.error(f"DEBUG: About to search for columns")
            reading_col = self.find_column_robust(df, ['قراءة الجهاز', 'reading of device', 'reading value', 'قراءة'])
            limit_col = self.find_column_robust(df, ['الحدود', 'limits', 'limit', 'حدود'])
            pesticide_col = self.find_column_robust(df, ['اسم المبيد', 'pesticide name', 'pesticide'])
            sample_col = self.find_column_robust(df, ['كود العينة', 'sample code', 'code'])
            
            self.logger.error(f"DEBUG: Found columns - Reading: {reading_col}, Limit: {limit_col}")
            self.logger.error(f"DEBUG: Additional columns - Pesticide: {pesticide_col}, Sample: {sample_col}")
            
            # Check if essential columns were found
            if not reading_col or not limit_col:
                self.logger.error("Cannot analyze contamination - missing essential columns (reading or limit)")
                self.logger.error(f"Available columns: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Create a working copy
            self.logger.error(f"DEBUG: Creating working copy of DataFrame")
            df_work = df.copy()
            
            # Log sample values before processing
            self.logger.error(f"DEBUG: Logging sample data before processing")
            for i in range(min(3, len(df_work))):
                row = df_work.iloc[i]
                reading_val = row.get(reading_col, 'N/A')
                limit_val = row.get(limit_col, 'N/A')
                self.logger.error(f"  Row {i}: Reading={reading_val}, Limit={limit_val}")
            
            # Clean and convert data
            self.logger.error(f"DEBUG: Starting data cleaning")
            # Handle Arabic "لا يوجد" (no data) values
            df_work[limit_col] = df_work[limit_col].replace(['لا يوجد', 'N/A', ''], pd.NA)
            df_work[reading_col] = df_work[reading_col].replace(['لا يوجد', 'N/A', ''], pd.NA)
            
            # Convert to numeric, handling errors gracefully
            self.logger.error(f"DEBUG: Converting to numeric")
            df_work[reading_col] = pd.to_numeric(df_work[reading_col], errors='coerce')
            df_work[limit_col] = pd.to_numeric(df_work[limit_col], errors='coerce')
            
            # Log conversion results
            self.logger.error(f"DEBUG: Logging data after numeric conversion")
            for i in range(min(3, len(df_work))):
                row = df_work.iloc[i]
                reading_val = row.get(reading_col, 'N/A')
                limit_val = row.get(limit_col, 'N/A')
                self.logger.error(f"  Row {i} after conversion: Reading={reading_val}, Limit={limit_val}")
            
            # Remove rows with missing or invalid data
            self.logger.error(f"DEBUG: About to drop NaN values")
            initial_count = len(df_work)
            df_work = df_work.dropna(subset=[reading_col, limit_col])
            final_count = len(df_work)
            
            self.logger.error(f"DEBUG: Data cleaning: {initial_count} -> {final_count} rows")
            
            if df_work.empty:
                self.logger.error("DEBUG: No valid data remaining after cleaning")
                return pd.DataFrame()
            
            # Calculate the contamination ratio for each sample
            self.logger.error(f"DEBUG: Calculating contamination ratios")
            df_work['contamination_ratio'] = df_work[reading_col] / df_work[limit_col]
            
            # Apply the filter: reading > (multiplier_threshold * limit)
            self.logger.error(f"DEBUG: About to apply contamination filter")
            contaminated_samples = df_work[df_work[reading_col] > (multiplier_threshold * df_work[limit_col])]
            
            self.logger.error(f"DEBUG: Contamination analysis results:")
            self.logger.error(f"  Samples analyzed: {len(df_work)}")
            self.logger.error(f"  Samples exceeding {multiplier_threshold}x limit: {len(contaminated_samples)}")
            
            # Log some examples
            if not contaminated_samples.empty:
                self.logger.error(f"DEBUG: Examples of contaminated samples:")
                for i, (idx, row) in enumerate(contaminated_samples.head(3).iterrows()):
                    reading_val = row[reading_col]
                    limit_val = row[limit_col]
                    ratio = row['contamination_ratio']
                    sample_name = row.get(sample_col, f'Row {idx}') if sample_col else f'Row {idx}'
                    pesticide_name = row.get(pesticide_col, 'Unknown') if pesticide_col else 'Unknown'
                    
                    self.logger.error(f"  {sample_name}: {pesticide_name} - Reading: {reading_val}, Limit: {limit_val}, Ratio: {ratio:.1f}x")
            else:
                self.logger.error(f"DEBUG: No samples found exceeding the threshold")
            
            # Remove the temporary calculation column before returning
            self.logger.error(f"DEBUG: Removing temporary calculation column")
            contaminated_samples = contaminated_samples.drop('contamination_ratio', axis=1)
            
            self.logger.error(f"DEBUG: Returning filtered DataFrame with {len(contaminated_samples)} rows")
            return contaminated_samples
            
        except Exception as e:
            self.logger.error(f"DEBUG: Exception in analyze_pesticide_contamination_levels: {e}")
            import traceback
            self.logger.error(f"DEBUG: Full traceback: {traceback.format_exc()}")
            return df

    def get_contamination_summary(self, df: pd.DataFrame) -> Dict:
        """
        Provides a summary of contamination levels in the dataset.
        
        Args:
            df: DataFrame containing pesticide test results
        
        Returns:
            Dictionary with contamination statistics
        """
        # Find required columns
        reading_col = self.find_column_robust(df, ['قراءة الجهاز', 'reading of device', 'reading value'])
        limit_col = self.find_column_robust(df, ['الحدود', 'limits', 'limit'])
        
        if not reading_col or not limit_col:
            return {"error": "Required columns not found"}
        
        # Clean and convert data
        df_work = df.copy()
        df_work[reading_col] = pd.to_numeric(df_work[reading_col], errors='coerce')
        df_work[limit_col] = pd.to_numeric(df_work[limit_col], errors='coerce')
        df_work = df_work.dropna(subset=[reading_col, limit_col])
        
        if df_work.empty:
            return {"error": "No valid data for analysis"}
        
        # Calculate ratios
        ratios = df_work[reading_col] / df_work[limit_col]
        
        # Generate summary statistics
        summary = {
            "total_samples": len(df_work),
            "samples_above_limit": len(ratios[ratios > 1.0]),
            "samples_above_2x_limit": len(ratios[ratios > 2.0]),
            "samples_above_5x_limit": len(ratios[ratios > 5.0]),
            "samples_above_10x_limit": len(ratios[ratios > 10.0]),
            "max_contamination_ratio": float(ratios.max()),
            "mean_contamination_ratio": float(ratios.mean()),
            "median_contamination_ratio": float(ratios.median())
        }
        
        return summary


    # --- METHOD WITH CORRECTIONS APPLIED ---
    # async def answer_with_local_data(self,
    #                                  project_id: int,
    #                                  query: str,
    #                                  n_results: int = 100,
    #                                  include_risk_assessment: bool = False,
    #                                  population_type: PopulationType = PopulationType.ADULT_AVERAGE
    #                                  ) -> Dict:
    #     """Answer query using local ChromaDB data with optional risk assessment"""
    #     if not self.chroma_client or not self.embed_fn or not self.generation_model:
    #         return {"success": False, "error": "Gemini API key not configured"}

    #     collection_name = f"local_project_{project_id}"
    #     try:
    #         collection = self.chroma_client.get_collection(name=collection_name, embedding_function=self.embed_fn)
    #         count = collection.count()
    #         if count == 0:
    #             return {"success": False, "error": "No documents in collection"}
    #     except Exception:
    #         return {"success": False, "error": f"Local collection not found: {collection_name}"}

    #     n_results = min(n_results, count)
    #     self.embed_fn.document_mode = False

    #     try:
    #         result = collection.query(query_texts=[query], n_results=n_results)
    #         if not result["documents"] or not result["documents"][0]:
    #             return {"success": False, "error": "No relevant documents found"}

    #         context_docs = result['documents'][0]
    #         context = "\n\n".join([doc for doc in context_docs])

    #         prompt = f"""
    #         **TASK:** Analyze the provided data to answer the user's question. You MUST format your entire response as a single, clean markdown table.
    #         **RULES:**
    #         1.  The table MUST have these exact, single-line headers: `Sample Code`, `Sample Name`, `Pesticide Name`, `Reading Value`, `Limit`, `Result`.
    #         2.  Do NOT include any text, explanation, or summary before or after the table.
    #         ---
    #         **USER QUESTION:** {query}
    #         ---
    #         **DATA:**
    #         {context}
    #         """

    #         answer = self.generation_model.generate_content(prompt)
    #         final_answer = answer.text
    #         df_for_risk = pd.DataFrame()

    #         try:
    #             df = parse_llm_response_to_df(answer.text)
    #             if df is not None and not df.empty:
    #                 self.logger.info(f"Successfully parsed LLM response into DataFrame with {len(df)} rows")
                   
    #                 self.logger.error(f"DEBUG: Initial DataFrame shape: {df.shape}")
    #                 self.logger.error(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
    #                 self.logger.error(f"DEBUG: First row data types:\n{df.dtypes}")
                    
    #                 df_filtered = self._apply_query_calculations(df, query)
                    
    #                 self.logger.error(f"DEBUG: After filtering shape: {df_filtered.shape}")
    #                 self.logger.error(f"DEBUG: df_for_risk is empty: {df_filtered.empty}")
                    
    #                 # Add this critical debug before risk assessment
    #                 if include_risk_assessment and not df_filtered.empty:
    #                     self.logger.error(f"DEBUG: About to call risk assessment with {len(df_filtered)} rows")
    
    #                     # 🔧 FIX: Prepare data for risk assessment
    #                     df_for_risk_clean = df_filtered.copy()
                        
    #                     # Convert numeric columns from strings to numbers
    #                     df_for_risk_clean['Reading Value'] = pd.to_numeric(df_for_risk_clean['Reading Value'], errors='coerce')
    #                     df_for_risk_clean['Limit'] = pd.to_numeric(df_for_risk_clean['Limit'], errors='coerce')
                        
    #                     # Remove any rows where conversion failed
    #                     df_for_risk_clean = df_for_risk_clean.dropna(subset=['Reading Value', 'Limit'])
                        
    #                     self.logger.error(f"DEBUG: After numeric conversion - Shape: {df_for_risk_clean.shape}")
    #                     self.logger.error(f"DEBUG: Data types after conversion:\n{df_for_risk_clean.dtypes}")
                        
    #                     if not df_for_risk_clean.empty:
    #                         try:
    #                             # 🔧 FIX: Ensure population_type is proper enum
    #                             risk_assessment_results = await self._perform_risk_assessment(
    #                                 df_for_risk_clean,  # Use cleaned DataFrame
    #                                 population_type     # This should already be PopulationType enum
    #                             )
    #                         except Exception as e:
    #                             self.logger.error(f"Risk assessment failed: {e}")
    #                             import traceback
    #                             self.logger.error(f"Full traceback: {traceback.format_exc()}")
    #                 if not df_filtered.equals(df):
    #                     self.logger.info(f"Filtering applied: {len(df)} -> {len(df_filtered)} rows")
    #                     final_answer = df_filtered.to_markdown(index=False) if not df_filtered.empty else "No data matches the specified criteria."
    #                     df_for_risk = df_filtered
    #                 else:
    #                     df_for_risk = df
    #             else:
    #                 self.logger.warning("Could not parse LLM response into DataFrame")

    #         except Exception as e:
    #             self.logger.error(f"Error applying calculations: {e}")

    #         risk_assessment_results = None
    #         # FIX: Use the 'include_risk_assessment' parameter passed into the method
    #         if include_risk_assessment and not df_for_risk.empty:
    #             try:
    #                 # 🔧 FIX: Convert string to enum if needed
    #                 if isinstance(population_type, str):
    #                     population_type = PopulationType(population_type)
    #                     self.logger.error(f"DEBUG: Converted to enum: {type(population_type)}, value: {population_type}")
                    
    #                 risk_assessment_results = await self._perform_risk_assessment(
    #                     df_for_risk_clean,
    #                     population_type
    #                 )
    #             except Exception as e:
    #                 self.logger.error(f"Risk assessment failed: {e}")

    #         return {
    #             "success": True,
    #             "answer": final_answer,
    #             "risk_assessment": risk_assessment_results.dict() if risk_assessment_results else None,
    #         }

    #     except Exception as e:
    #         self.logger.error(f"Error processing local query: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return {"success": False, "error": f"Error generating response: {str(e)}"}


    async def answer_with_local_data(self,
                                        project_id: int,
                                        query: str,
                                        n_results: int = 100,
                                        include_risk_assessment: bool = False,
                                        population_type: PopulationType = PopulationType.ADULT_AVERAGE
                                        ) -> Dict:
            """Answer query using local ChromaDB data with OpenAI"""
            
            if not self.chroma_client or not self.embed_fn:
                return {"success": False, "error": "OpenAI API key not configured"}

            collection_name = f"local_project_{project_id}"
            try:
                collection = self.chroma_client.get_collection(name=collection_name, embedding_function=self.embed_fn)
                count = collection.count()
                if count == 0:
                    return {"success": False, "error": "No documents in collection"}
            except Exception:
                return {"success": False, "error": f"Local collection not found: {collection_name}"}

            n_results = min(n_results, count)
            
            try:
                result = collection.query(query_texts=[query], n_results=n_results)
                if not result["documents"] or not result["documents"][0]:
                    return {"success": False, "error": "No relevant documents found"}

                context_docs = result['documents'][0]
                context = "\n\n".join([doc for doc in context_docs])

                prompt = f"""
                **TASK:** Analyze the provided data to answer the user's question. You MUST format your entire response as a single, clean markdown table.
                **RULES:**
                1.  The table MUST have these exact, single-line headers: `Sample Code`, `Sample Name`, `Pesticide Name`, `Reading Value`, `Limit`, `Result`.
                2.  Do NOT include any text, explanation, or summary before or after the table.
                ---
                **USER QUESTION:** {query}
                ---
                **DATA:**
                {context}
                """

                # Use OpenAI instead of Gemini
                response = self.openai_client.chat.completions.create(
                    model=os.getenv("GENERATION_MODEL_ID", "gpt-4o-mini"),
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that analyzes pesticide test data."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=int(os.getenv("GENERATION_DAFAULT_MAX_TOKENS", 2000)),
                    temperature=float(os.getenv("GENERATION_DAFAULT_TEMPERATURE", 0.1))
                )
                
                final_answer = response.choices[0].message.content
                df_for_risk = pd.DataFrame()

                try:
                    df = parse_llm_response_to_df(final_answer)
                    if df is not None and not df.empty:
                        self.logger.info(f"Successfully parsed LLM response into DataFrame with {len(df)} rows")
                    
                        df_filtered = self._apply_query_calculations(df, query)
                        
                        if include_risk_assessment and not df_filtered.empty:
                            # Prepare data for risk assessment
                            df_for_risk_clean = df_filtered.copy()
                            df_for_risk_clean['Reading Value'] = pd.to_numeric(df_for_risk_clean['Reading Value'], errors='coerce')
                            df_for_risk_clean['Limit'] = pd.to_numeric(df_for_risk_clean['Limit'], errors='coerce')
                            df_for_risk_clean = df_for_risk_clean.dropna(subset=['Reading Value', 'Limit'])
                            
                            if not df_for_risk_clean.empty:
                                try:
                                    risk_assessment_results = await self._perform_risk_assessment(
                                        df_for_risk_clean,
                                        population_type
                                    )
                                except Exception as e:
                                    self.logger.error(f"Risk assessment failed: {e}")
                        
                        if not df_filtered.equals(df):
                            self.logger.info(f"Filtering applied: {len(df)} -> {len(df_filtered)} rows")
                            final_answer = df_filtered.to_markdown(index=False) if not df_filtered.empty else "No data matches the specified criteria."
                            df_for_risk = df_filtered
                        else:
                            df_for_risk = df
                    else:
                        self.logger.warning("Could not parse LLM response into DataFrame")

                except Exception as e:
                    self.logger.error(f"Error applying calculations: {e}")

                risk_assessment_results = None
                if include_risk_assessment and not df_for_risk.empty:
                    try:
                        if isinstance(population_type, str):
                            population_type = PopulationType(population_type)
                        
                        risk_assessment_results = await self._perform_risk_assessment(
                            df_for_risk_clean,
                            population_type
                        )
                    except Exception as e:
                        self.logger.error(f"Risk assessment failed: {e}")

                return {
                    "success": True,
                    "answer": final_answer,
                    "risk_assessment": risk_assessment_results.dict() if risk_assessment_results else None,
                }

            except Exception as e:
                self.logger.error(f"Error processing local query: {e}")
                import traceback
                traceback.print_exc()
                return {"success": False, "error": f"Error generating response: {str(e)}"}

        # Keep all your other methods unchanged...

    async def _perform_risk_assessment(self, df: pd.DataFrame,
                                population_type: PopulationType) -> RiskAssessmentSummary:
        """Perform health risk assessment on DataFrame"""
        self.logger.info(f"Performing risk assessment for {len(df)} samples")
        self.logger.error(f"DEBUG: Risk assessment input - Shape: {df.shape}")
        self.logger.error(f"DEBUG: population_type type: {type(population_type)}")
        self.logger.error(f"DEBUG: population_type value: {population_type}")
        self.logger.error(f"DEBUG: DataFrame dtypes:\n{df.dtypes}")
        
        # Check if Reading Value and Limit are actually numeric
        sample_reading = df['Reading Value'].iloc[0] if not df.empty else None
        sample_limit = df['Limit'].iloc[0] if not df.empty else None
        self.logger.error(f"DEBUG: Sample reading type: {type(sample_reading)}, value: {sample_reading}")
        self.logger.error(f"DEBUG: Sample limit type: {type(sample_limit)}, value: {sample_limit}")
        
        risk_service = RiskAssessmentService()
        assessment_summary = risk_service.assess_dataframe_risk(
            df=df,
            population_type=population_type
        )
        self.logger.info(f"Risk assessment completed for {assessment_summary.total_samples_assessed} samples")
        return assessment_summary

            
    def _apply_query_calculations(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """
        Simplified contamination filtering to avoid tabulate issues
        """
        more_than_match = re.search(r'more than (\d+\.?\d*)\s*times the limit', query, re.IGNORECASE)
        if not more_than_match:
            return df

        multiplier = float(more_than_match.group(1))
        try:
            df_work = df.copy()
            # Ensure columns exist before trying to convert
            if 'Reading Value' in df_work.columns and 'Limit' in df_work.columns:
                df_work['Reading Value'] = pd.to_numeric(df_work['Reading Value'], errors='coerce')
                df_work['Limit'] = pd.to_numeric(df_work['Limit'], errors='coerce')
                
                # Drop rows where conversion might have failed
                df_work.dropna(subset=['Reading Value', 'Limit'], inplace=True)

                threshold = multiplier * df_work['Limit']
                filtered_df = df_work[df_work['Reading Value'] > threshold]
                return filtered_df
            else:
                self.logger.warning("Could not find 'Reading Value' or 'Limit' columns for calculation.")
                return df

        except Exception as e:
            self.logger.error(f"DEBUG: Error in simplified filtering: {e}")
            return df