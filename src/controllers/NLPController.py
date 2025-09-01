import logging
import json
from typing import List, Dict, Tuple, Optional, Any
import os
import pandas as pd
from pathlib import Path
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
from google.api_core import retry
import logging
from tqdm import tqdm

from .BaseControllers import BaseControllers
from models.db_schemes import Project, DataChunk
from stores.llm.LLMEnum import DocumentTypeEnum

import re
from bidi.algorithm import get_display
import arabic_reshaper

# استيراد الـ Parser المتخصص الجديد لتقارير المبيدات
from stores.llm.templates.pesticide_templates import PesticideTemplateParser

from utils.data_cleaning import clean_pesticide_data, load_and_prepare_excel, format_rows_as_text


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
class EnhancedNLPController:
    """Enhanced NLP Controller with local data processing"""
    
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
            
            # Configure Gemini for generation
            genai.configure(api_key=self.gemini_api_key)
            self.generation_model = genai.GenerativeModel("gemini-1.5-flash-latest")
        else:
            self.chroma_client = None
            self.embed_fn = None
            self.generation_model = None
        
        self.logger = logging.getLogger(__name__)

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
            months = ["Feb", "March", "April", "May", "June",
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
    
    async def answer_with_local_data(self, project_id: int, query: str, n_results: int = 100) -> Dict:
        """Answer query using local ChromaDB data"""
        if not self.chroma_client or not self.embed_fn or not self.generation_model:
            return {
                "success": False,
                "error": "Gemini API key not configured for local data processing"
            }
        
        # Validate n_results parameter
        if n_results <= 0:
            n_results = 10  # Default fallback
            
        collection_name = f"local_project_{project_id}"
        
        try:
            collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embed_fn
            )
        except Exception as e:
            return {
                "success": False,
                "error": f"Local collection not found: {collection_name}. Please index local data first."
            }
        
        # Check if collection has any documents
        count = collection.count()
        if count == 0:
            return {
                "success": False,
                "error": "No documents found in collection. Please index data first."
            }
        
        # Ensure n_results doesn't exceed available documents
        n_results = min(n_results, count)
        
        # Set to query mode
        self.embed_fn.document_mode = False
        
        try:
            # FIX 1: Use the actual query parameter instead of hardcoded query
            self.logger.info(f"Querying with: {query}")
            
            result = collection.query(
                query_texts=[query],  # Use the actual query parameter
                n_results=n_results
            )
            
            if not result["documents"] or not result["documents"][0]:
                return {
                    "success": False,
                    "error": "No relevant documents found in local data."
                }
            
            # Prepare context from retrieved documents
            context_docs = result['documents'][0]
            context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
            
            # Log retrieved documents for debugging
            self.logger.info(f"Retrieved {len(context_docs)} documents")
            if context_docs:
                self.logger.info(f"First document preview: {context_docs[0][:200]}...")
            
            # Combine all retrieved documents (EXACT COLAB STYLE)
            passages = context_docs
            combined_passage = " ".join(passages)
            
            # Clean up for prompt
            passage_oneline = combined_passage.replace("\n", " ")
            query_oneline = query.replace("\n", " ")
            
            # Create debug info before generation
            debug_info = {
                "query_used": query,
                "docs_found": len(context_docs),
                "first_doc_preview": context_docs[0][:300] if context_docs else "No docs",
                "combined_passage_length": len(combined_passage),
                "contains_pepper_data": "pepper" in combined_passage.lower() or "فلفل" in combined_passage,
                "contains_pesticide_data": "pesticide" in combined_passage.lower() or "مبيد" in combined_passage,
                "contains_result_data": "result" in combined_passage.lower() or "نتيجة" in combined_passage,
                "contains_nan_values": "NaN" in combined_passage,
                "contains_sample_codes": "P45-" in combined_passage
            }
            
            # Clean up for prompt
            passage_oneline = combined_passage.replace("\n", " ")
            query_oneline = query.replace("\n", " ")
            
            # FIX 2: Create prompt that handles messy data better

            prompt = f"""You are analyzing pesticide test data from Excel files.

            TASK: Extract all pesticide readings and test results from the data.

            QUESTION: {query_oneline}
            DATA: {passage_oneline}

            INSTRUCTIONS:
            - List each reading with: Sample Code, Sample Name, Pesticide Name, Reading Value, Limit, Result
            - Focus on actual data, ignore NaN and "لا يوجد" entries
            - Be precise and complete"""

            # FIX 3: Add more logging before generation    
            # Log the prompt for debugging
            self.logger.info(f"Generated prompt length: {len(prompt)}")
            self.logger.info(f"Prompt preview: {prompt[:300]}...")
            
            # Generate answer
            answer = self.generation_model.generate_content(prompt)
            
            return {
                "success": True,
                "answer": answer.text,
                "full_prompt": prompt,
                "retrieved_docs_count": len(passages),
                "collection_total": collection.count(),
                "debug_info": {
                    **debug_info,
                    "answer_length": len(answer.text),
                    "prompt_length": len(prompt),
                    "full_answer": answer.text  # Include full answer for debugging
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing local query: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "error": f"Error generating response: {str(e)}"
            }