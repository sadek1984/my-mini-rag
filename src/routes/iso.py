import logging
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import re

# --- Project Imports ---
from dependencies import get_db
from models.ChunkModel import ChunkModel
from models.ProjectModel import ProjectModel
from models.db_schemes.minirag.schemes.asset import Asset as DBAsset
from models.enums.ResponseEnums import ResponseSignal
from processors.iso_document_processor import parse_document_metadata
from stores.VectorDB.VectorDBInterface import VectorDBInterface
from langchain.text_splitter import RecursiveCharacterTextSplitter
from routes.schemes.nlp import SearchRequest
from controllers.NLPController import NLPController
import os
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/iso",
    tags=["ISO 17025 Management"]
)

@router.post("/upload_document/{project_id}")
async def upload_iso_document(
    project_id: str,
    file: UploadFile = File(...),
    document_type: str = Form(default="Unknown")
):
    """Upload with fixed metadata handling"""
    file_path = None
    try:
        logger.info(f"Starting upload for project {project_id}, file: {file.filename}")
        
        # 1. حفظ ومعالجة الملف
        file_path = f"/tmp/{file.filename}"
        content = await file.read()
        with open(file_path, "wb") as buffer:
            buffer.write(content)
        
        from processors.iso_document_processor import parse_document_metadata
        parsed_data = parse_document_metadata(file_path, document_type)
        doc_text = parsed_data["text"]
        doc_metadata = parsed_data["metadata"]
        
        # 2. إصلاح metadata للتوافق مع ChromaDB
        def fix_metadata_for_chromadb(metadata_dict):
            """تحويل metadata ليتوافق مع ChromaDB"""
            fixed_metadata = {}
            for key, value in metadata_dict.items():
                if isinstance(value, list):
                    # تحويل القوائم إلى strings مفصولة بفواصل
                    fixed_metadata[key] = ", ".join(map(str, value))
                elif isinstance(value, (dict, tuple)):
                    # تحويل القواميس والـ tuples إلى strings
                    fixed_metadata[key] = str(value)
                elif value is None:
                    # تحويل None إلى string فارغ
                    fixed_metadata[key] = ""
                elif isinstance(value, (str, int, float, bool)):
                    # القيم المدعومة مباشرة
                    fixed_metadata[key] = value
                else:
                    # أي نوع آخر، حوله إلى string
                    fixed_metadata[key] = str(value)
            return fixed_metadata
        
        # 3. فهرسة مع metadata مُصحح
        try:
            from controllers.NLPController import EnhancedNLPController
            
            nlp_controller = EnhancedNLPController(
                gemini_api_key=os.getenv("GEMINI_API_KEY")
            )
            
            if nlp_controller.chroma_client:
                # تقسيم النص
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=150
                )
                chunks = text_splitter.split_text(doc_text)
                
                # إنشاء المجموعة
                collection_name = f"local_project_{project_id}"
                collection = nlp_controller.chroma_client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=nlp_controller.embed_fn
                )
                
                # إعداد metadata مُصحح
                base_metadata = fix_metadata_for_chromadb(doc_metadata)
                base_metadata.update({
                    "source_file": file.filename,
                    "document_type": document_type,
                    "project_id": str(project_id),
                    "upload_timestamp": datetime.now().isoformat()
                })
                
                # إنشاء chunks مع metadata
                chunk_ids = [f"iso_{file.filename}_{i}" for i in range(len(chunks))]
                chunk_metadata = []
                
                for i in range(len(chunks)):
                    chunk_meta = base_metadata.copy()
                    chunk_meta["chunk_index"] = i
                    chunk_metadata.append(chunk_meta)
                
                # فهرسة
                collection.add(
                    documents=chunks,
                    ids=chunk_ids,
                    metadatas=chunk_metadata
                )
                
                logger.info(f"Successfully indexed {len(chunks)} chunks")
                
                return {
                    "message": f"Document '{file.filename}' uploaded and indexed successfully",
                    "project_id": project_id,
                    "document_type": document_type,
                    "chunks_indexed": len(chunks),
                    "file_size": len(content),
                    "iso_clauses": doc_metadata.get("related_clauses", []),
                    "metadata_keys": list(base_metadata.keys()),
                    "status": "success"
                }
            else:
                return {
                    "message": f"Document uploaded (ChromaDB not available)",
                    "project_id": project_id,
                    "file_size": len(content)
                }
                
        except Exception as index_error:
            logger.error(f"Indexing failed: {index_error}")
            return {
                "message": f"Document uploaded (indexing failed)",
                "project_id": project_id,
                "file_size": len(content),
                "error": f"Indexing error: {str(index_error)}"
            }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)



@router.post("/answer/{project_id}")
async def answer_iso_question(
    project_id: str,
    query_data: dict
):
    """Query ISO documents for specific project"""
    try:
        query_text = query_data.get("text", "")
        logger.info(f"Query for project {project_id}: {query_text}")
        
        from controllers.NLPController import EnhancedNLPController
        
        nlp_controller = EnhancedNLPController(
            gemini_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        if not nlp_controller.chroma_client:
            return {"error": "ChromaDB not available"}
        
        # البحث في المجموعة الصحيحة
        collection_name = f"local_project_{project_id}"
        
        try:
            collection = nlp_controller.chroma_client.get_collection(
                name=collection_name,
                embedding_function=nlp_controller.embed_fn
            )
        except:
            return {
                "project_id": project_id,
                "query": query_text,
                "answer": f"لم يتم العثور على مستندات للمشروع {project_id}. يرجى رفع مستندات أولاً.",
                "status": "no_collection"
            }
        
        # البحث
        results = collection.query(
            query_texts=[query_text],
            n_results=5
        )
        
        if not results["documents"] or not results["documents"][0]:
            return {
                "project_id": project_id,
                "query": query_text,
                "answer": "لم يتم العثور على معلومات ذات صلة في المستندات المرفوعة",
                "status": "no_results"
            }
        
        # تكوين السياق والإجابة
        context = "\n---\n".join(results["documents"][0])
        
        prompt = f"""
        Based on the following ISO 17025 documents, answer the question directly and concisely.

        Context: {context}

        Question: {query_text}

        IMPORTANT INSTRUCTIONS:
        - Answer in the same language as the question (English question → English answer, Arabic question → Arabic answer)
        - Be concise and direct
        - Use clear formatting without asterisks or special characters
        - Focus only on what the question asks for

        Answer:
        """
        
        # تحسينات الإجابة
        language = detect_query_language(query_text)
        context = "\n---\n".join(results["documents"][0])
        prompt = create_language_specific_prompt(query_text, context, language)

        answer = nlp_controller.generation_model.generate_content(prompt)
        cleaned_answer = clean_answer_text(answer.text)

        
        return {
            "project_id": project_id,
            "query": query_text,
            "answer": cleaned_answer,
            "success": True,
            "sources": list(set([meta.get("source_file", "Unknown") for meta in results["metadatas"][0]])),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {"error": str(e)}
    
##### New methods for language selectivity ######

def detect_query_language(query: str) -> str:
    """كشف لغة الاستعلام"""
    # البحث عن أحرف عربية
    arabic_pattern = re.compile(r'[\u0600-\u06FF]')
    if arabic_pattern.search(query):
        return "arabic"
    else:
        return "english"

def create_language_specific_prompt(query: str, context: str, language: str) -> str:
    """إنشاء prompt حسب اللغة"""
    
    if language == "arabic":
        return f"""
        بناءً على مستندات ISO 17025 التالية، أجب على السؤال بوضوح وإيجاز.

        السياق: {context}
        السؤال: {query}

        تعليمات مهمة:
        - أجب بالعربية
        - كن مباشر ومختصر
        - ركز على ما يطلبه السؤال فقط
        - استخدم تنسيق واضح بدون رموز خاصة

        الجواب:
        """
    else:
        return f"""
        Based on the following ISO 17025 documents, answer the question clearly and concisely.

        Context: {context}
        Question: {query}

        Important instructions:
        - Answer in English
        - Be direct and concise  
        - Focus only on what the question asks for
        - Use clear formatting without special characters

        Answer:
        """

def clean_answer_text(text: str) -> str:
    """تنظيف النص من الرموز الغريبة"""
    # إزالة الـ markdown asterisks
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # **bold** → bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # *italic* → italic
    
    # إزالة \n الغريبة
    text = text.replace('\\n', '\n')
    
    # إزالة المسافات الزائدة
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


async def simple_text_search(request, project_id: str, query: str):
    """بحث نصي بسيط كبديل لـ vector search"""
    try:
        # استخدام العقدة المحلية ChromaDB
        from controllers.NLPController import EnhancedNLPController
        
        nlp_controller = EnhancedNLPController(
            gemini_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        # استخدام local data إذا كان متاح
        if nlp_controller.chroma_client:
            result = await nlp_controller.answer_with_local_data(
                project_id=int(project_id) if project_id.isdigit() else 4,
                query=query,
                n_results=5
            )
            
            if result.get("success"):
                return {
                    "project_id": project_id,
                    "query": query,
                    "answer": result["answer"],
                    "sources": ["Local ChromaDB"],
                    "status": "success_local"
                }
        
        # إجابة تجريبية إذا فشل كل شيء
        return {
            "project_id": project_id,
            "query": query,
            "answer": f"تم استلام الاستعلام: {query}. النظام قيد التطوير.",
            "sources": [],
            "status": "fallback"
        }
        
    except Exception as e:
        logger.error(f"Fallback search failed: {e}")
        return {
            "project_id": project_id,
            "query": query,
            "answer": "عذراً، حدث خطأ في معالجة الاستعلام",
            "sources": [],
            "status": "error"
        }

def detect_document_type_from_query(query: str) -> dict:
    """تحديد نوع المستند من الاستعلام"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["procedure", "sop", "إجراء"]):
        return {"document_type": "Procedure"}
    elif any(word in query_lower for word in ["policy", "manual", "سياسة", "دليل"]):
        return {"document_type": "Quality Manual"} 
    elif any(word in query_lower for word in ["instruction", "how to", "تعليمات", "كيف"]):
        return {"document_type": "Work Instruction"}
    
    return {}  # بحث في جميع الأنواع

def create_iso_hierarchical_prompt(query: str, context: str, chunks: list) -> str:
    """إنشاء prompt للإجابة الهرمية"""
    
    # تحديد نوع المستند الأساسي
    doc_types = [chunk.get("metadata", {}).get("document_type", "Unknown") for chunk in chunks]
    primary_doc_type = max(set(doc_types), key=doc_types.count) if doc_types else "Unknown"
    
    prompt = f"""
    أجب على السؤال بالتنسيق الهرمي التالي لنظام ISO 17025:

    **المستوى:** {primary_doc_type}
    **النص المسترجع:** [اقتبس المعلومة المطلوبة من النص]
    **العلاقة:** [اشرح كيف ترتبط بالمستوى الأعلى - مثل SOP مرتبط بدليل الجودة]
    **السبب/السياق:** [أهمية هذه المعلومة في إطار ISO 17025]

    السياق من المستندات:
    {context}

    السؤال: {query}
    
    أجب باللغة العربية مع الاحتفاظ بالمصطلحات التقنية الإنجليزية عند الضرورة.
    """
    
    return prompt

# routes/iso.py - إضافة endpoint للمعالجة الهرمية
@router.post("/upload_hierarchy/{project_id}")
async def upload_iso_hierarchy(
    project_id: str,
    folder_path: str = Form(...)
):
    """رفع وفهرسة كامل هيكل ISO 17025"""
    try:
        from processors.iso_hierarchical_processor import ISO17025HierarchicalProcessor
        
        processor = ISO17025HierarchicalProcessor()
        hierarchy_data = processor.process_iso_folder_structure(folder_path)
        
        # فهرسة هرمية في ChromaDB
        from controllers.NLPController import EnhancedNLPController
        nlp_controller = EnhancedNLPController(
            gemini_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        indexed_docs = await index_hierarchical_documents(
            nlp_controller, hierarchy_data, project_id
        )
        
        return {
            "message": "ISO 17025 hierarchy processed successfully",
            "project_id": project_id,
            "documents_processed": len(hierarchy_data["documents"]),
            "relationships_discovered": len(hierarchy_data["relationships"]),
            "hierarchy_levels": [1, 2, 3, 4],
            "status": "hierarchical_success"
        }
        
    except Exception as e:
        logger.error(f"Hierarchy processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query_hierarchical/{project_id}")
async def query_iso_hierarchical(
    project_id: str,
    query_data: dict
):
    """استعلام هرمي ذكي لـ ISO 17025"""
    try:
        from engines.iso_hierarchical_search import ISO17025HierarchicalSearch
        from controllers.NLPController import EnhancedNLPController
        
        nlp_controller = EnhancedNLPController(
            gemini_api_key=os.getenv("GEMINI_API_KEY")
        )
        
        search_engine = ISO17025HierarchicalSearch(nlp_controller)
        
        result = await search_engine.hierarchical_search(
            query=query_data.get("text", ""),
            project_id=project_id
        )
        
        return {
            "project_id": project_id,
            "query": query_data.get("text", ""),
            "hierarchical_answer": result,
            "status": "hierarchical_success"
        }
        
    except Exception as e:
        logger.error(f"Hierarchical query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
async def index_hierarchical_documents(nlp_controller, hierarchy_data, project_id):
    """فهرسة المستندات الهرمية"""
    try:
        collection_name = f"local_project_{project_id}"
        collection = nlp_controller.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=nlp_controller.embed_fn
        )
        
        indexed_count = 0
        
        for doc_data in hierarchy_data["documents"]:
            # تقسيم النص إلى chunks
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_text(doc_data["text"])
            
            # إعداد metadata للفهرسة
            base_metadata = {
                "source_file": doc_data["metadata"]["source_file"],
                "document_type": doc_data["metadata"]["document_type"],
                "hierarchy_level": doc_data["hierarchy_level"],
                "hierarchy_role": doc_data["hierarchy_role"],
                "project_id": str(project_id)
            }
            
            # فهرسة chunks
            chunk_ids = [f"hier_{doc_data['metadata']['source_file']}_{i}" for i in range(len(chunks))]
            chunk_metadata = [
                {**base_metadata, "chunk_index": i} 
                for i in range(len(chunks))
            ]
            
            collection.add(
                documents=chunks,
                ids=chunk_ids,
                metadatas=chunk_metadata
            )
            
            indexed_count += len(chunks)
        
        return indexed_count
        
    except Exception as e:
        logger.error(f"Hierarchical indexing failed: {e}")
        raise e
        