# from .BaseControllers import BaseControllers
# from ..models.db_schemes import Project, DataChunk
# from ..stores.llm.LLMEnum import DocumentTypeEnum
# from typing import List
# import json

# class NLPController(BaseControllers):

#     def __init__(self, vectordb_client, generation_client, 
#                  embedding_client, template_parser):
#         super().__init__()

#         self.vectordb_client = vectordb_client
#         self.generation_client = generation_client 
#         self.embedding_client = embedding_client
#         self.template_parser = template_parser

#     def create_collection_name(self, project_id: str):
#         return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()
    
#     async def reset_vector_db_collection(self, project: Project):
#         collection_name = self.create_collection_name(project_id=project.project_id)
#         return await self.vectordb_client.delete_collection(collection_name=collection_name)
    
#     async def get_vector_db_collection_info(self, project: Project):
#         collection_name = self.create_collection_name(project_id=project.project_id)

#         collection_exists = await self.vectordb_client.is_collection_existed(collection_name)
    
#         if not collection_exists:
#             return {
#                 "collection_exists": False,
#                 "collection_name": collection_name,
#                 "message": "Collection not found. Index the project first."
#             }

#         collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)
        
#         if not collection_info:
#             return{
#                 "collection_exists": False,
#                 "collection_name": collection_name,
#                 "message": "Failed to retrieve collection info"
#             }
#     # this line converts the collection info to a JSON serializable format
#         return json.loads(
#             json.dumps(collection_info, default=lambda x: x.__dict__)
#         )
    
#     async def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
#                                    chunks_ids: List[int], 
#                                    do_reset: bool = False):
        
#         # step1: get collection name
#         collection_name = self.create_collection_name(project_id=project.project_id)

#         # step2: manage items
#         texts = [ c.chunk_text for c in chunks ]
#         metadata = [ c.chunk_metadata for c in  chunks]

#          # FIXED: Pass individual text, not the entire texts list
#         raw_vectors = self.embedding_client.embed_text(text=texts, 
#                                                     document_type=DocumentTypeEnum.DOCUMENT.value)
                    
        
#         # Fix: Extract actual vectors from nested lists
#         vectors = []
#         for i, raw_vector in enumerate(raw_vectors):
#             if isinstance(raw_vector, list) and len(raw_vector) > 0 and isinstance(raw_vector[0], list):
#                 vectors.append(raw_vector[0])  # Extract the actual vector
#             else:
#                 vectors.append(raw_vector)
        
#         print(f"Successfully generated {len([v for v in vectors if v])} embeddings")
        
#         # Debug vector shapes
#         for i, v in enumerate(vectors[:3]):  # Check first 3 vectors
#             print(f"Vector {i} type: {type(v)}, length: {len(v) if hasattr(v, '__len__') else 'N/A'}")
#             if hasattr(v, '__len__') and len(v) > 0:
#                 print(f"First few values: {v[:5]}")
        
#         print("=== END EMBEDDING DEBUG ===")

#         # step3: create collection if not exists
#         await self.vectordb_client.create_collection(
#             collection_name=collection_name,
#             embedding_size=self.embedding_client.embedding_size,
#             do_reset=do_reset,
#         )

#         # step4: insert into vector db

#         await self.vectordb_client.insert_many(
#             collection_name=collection_name,
#             texts=texts,
#             metadata=metadata,
#             vectors=vectors,
#             record_ids=chunks_ids,
#         )

#         return True

#     async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

#         # step1: get collection name
#         # query_vector = None
#         collection_name = self.create_collection_name(project_id=project.project_id)

#         # step2: get text embedding vector
#         print(f"=== QUERY EMBEDDING DEBUG ===")
#         print(f"Query text: {text}")
#         print(f"Embedding client type: {type(self.embedding_client)}")
        
#         raw_embedding = self.embedding_client.embed_text(text=text, 
#                                                     document_type=DocumentTypeEnum.QUERY.value)
        
#         print(f"Raw embedding type: {type(raw_embedding)}")
#         print(f"Raw embedding length: {len(raw_embedding) if hasattr(raw_embedding, '__len__') else 'N/A'}")
        
#         # Fix: Extract the actual vector from the nested list
#         if isinstance(raw_embedding, list) and len(raw_embedding) > 0 and isinstance(raw_embedding[0], list):
#             query_vector = raw_embedding[0]  # Get the first (and only) vector
#             print(f"Extracted vector length: {len(query_vector)}")
#         else:
#             query_vector = raw_embedding
        
#         print(f"Final query vector type: {type(query_vector)}")
#         print(f"Final query vector length: {len(query_vector) if hasattr(query_vector, '__len__') else 'N/A'}")
#         print("=== END QUERY EMBEDDING DEBUG ===")

#         if not query_vector:
#             self.logger.error("Failed to generate embedding for query text")
#             return False
        
#         # Ensure query_vector is a list of numbers
#         if not isinstance(query_vector, list):
#             self.logger.error(f"Expected list for query_vector, got {type(query_vector)}")
#             return False
            
#         if len(query_vector) == 0:
#             self.logger.error("Query vector is empty")
#             return False

#         # step3: do semantic search
#         results = await self.vectordb_client.search_by_vector(
#             collection_name=collection_name,
#             vector=query_vector,
#             limit=limit
#         )

#         if not results:
#             return False

#         return results
#     # answer_rag_question method retrieves relevant documents from search_vector_db_collection
#     # method and generates an answer using the LLM.
#     async def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        
#          # Debug 1: Verify generation client config
#         print("\n=== MODEL DEBUGGING ===")
#         print(f"Generation client type: {type(self.generation_client)}")
#         print(f"Current model ID: {getattr(self.generation_client, 'model_id', 'UNKNOWN')}")
#         print(f"Available models: {getattr(self.generation_client, 'available_models', 'UNKNOWN')}")

#         answer, full_prompt, chat_history = None, None, None

#         # step1: retrieve related documents
#         retrieved_documents = await self.search_vector_db_collection(
#             project=project,
#             text=query,
#             limit=limit,
#         )

#         if not retrieved_documents or len(retrieved_documents) == 0:
#             return answer, full_prompt, chat_history
        
#         # step2: Construct LLM prompt
#         system_prompt = self.template_parser.get("rag", "system_prompt")

#         documents_prompts = "\n".join([
#             self.template_parser.get("rag", "document_prompt", {
#                     "doc_num": idx + 1,
#                     "chunk_text": self.generation_client.process_text(doc.text),
#             })
#             for idx, doc in enumerate(retrieved_documents[:3])  # Limit to top 3
#         ])

#         footer_prompt = self.template_parser.get("rag", "footer_prompt", {
#             "query": query
#         })

#         # step3: Construct Generation Client Prompts
#         chat_history = [
#             self.generation_client.construct_prompt(
#                 prompt=system_prompt,
#                 role=self.generation_client.enums.SYSTEM.value,
#             )
#         ]

#         full_prompt = "\n\n".join([ documents_prompts,  footer_prompt])

#         # step4: Retrieve the Answer
#         answer = self.generation_client.generate_text(
#             prompt=full_prompt,
#             chat_history=chat_history
#         )

#         return answer, full_prompt, chat_history



from .BaseControllers import BaseControllers
from ..models.db_schemes import Project, DataChunk
from ..stores.llm.LLMEnum import DocumentTypeEnum
from typing import List
import json

class NLPController(BaseControllers):

    def __init__(self, vectordb_client, generation_client, 
                 embedding_client, template_parser):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser

    def create_collection_name(self, project_id: str):
        return f"collection_{self.vectordb_client.default_vector_size}_{project_id}".strip()
    
    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)
    
    async def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    async def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        
        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: manage items
        texts = [ c.chunk_text for c in chunks ]
        metadata = [ c.chunk_metadata for c in  chunks]
        vectors = self.embedding_client.embed_text(text=texts, 
                                                  document_type=DocumentTypeEnum.DOCUMENT.value)

        # step3: create collection if not exists
        _ = await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # step4: insert into vector db
        _ = await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
        )

        return True

    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):

        # step1: get collection name
        query_vector = None
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: get text embedding vector
        vectors = self.embedding_client.embed_text(text=text, 
                                                 document_type=DocumentTypeEnum.QUERY.value)

        if not vectors or len(vectors) == 0:
            return False
        
        if isinstance(vectors, list) and len(vectors) > 0:
            query_vector = vectors[0]

        if not query_vector:
            return False    

        # step3: do semantic search
        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=query_vector,
            limit=limit
        )

        if not results:
            return False

        return results
    
    async def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        
        answer, full_prompt, chat_history = None, None, None

        # step1: retrieve related documents
        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history
        
        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": self.generation_client.process_text(doc.text),
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {
            "query": query
        })

        # step3: Construct Generation Client Prompts
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([ documents_prompts,  footer_prompt])

        # step4: Retrieve the Answer
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history
