
from fastapi import FastAPI
from routes.base import base_router  
from routes.data import data_router
from routes.data import debug_router
from routes import nlp
from helper.config import get_settings
from stores.llm.LLMProviderFactory import LLMProviderFactory
from stores.VectorDB.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from utils.metrics import setup_metrics
from controllers import NLPController
from routes.risk_assessment_routes import risk_router

import logging
from pathlib import Path

app = FastAPI()

# Setup Prometheus metrics
setup_metrics(app)

async def startup_span():
    settings = get_settings()
    postgres_conn = f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"

    app.db_engine = create_async_engine(postgres_conn)
    app.db_client = sessionmaker(
        app.db_engine, class_=AsyncSession, expire_on_commit=False
    )
    llm_provider_factory = LLMProviderFactory(settings)
    vectordb_provider_factory = VectorDBProviderFactory(config=settings, db_client=app.db_client)


    # generation client
    app.generation_client = llm_provider_factory.create(provider=settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id = settings.GENERATION_MODEL_ID)
    print(settings.EMBEDDING_BACKEND)

    # embedding client
    app.embedding_client = llm_provider_factory.create(provider=settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(model_id=settings.EMBEDDING_MODEL_ID,
                                             embedding_size=settings.EMBEDDING_MODEL_SIZE)
    # app.template_parser = NLPController.template_parser

    # vector db client
    app.vectordb_client = vectordb_provider_factory.create(
    provider=settings.VECTOR_DB_BACKEND
    )
    await app.vectordb_client.connect()

    app.template_parser = TemplateParser(
        language=settings.PRIMARY_LANG,
        default_language=settings.DEFAULT_LANG,
    )


async def shutdown_span():
    app.db_engine.dispose()
    await app.vectordb_client.disconnect()
    
# app.router.lifespan.on_startup.append(startup_span)
# app.router.lifespan.on_shutdown.append(shutdown_span)
 
app.on_event("startup")(startup_span)
app.on_event("shutdown")(shutdown_span)

app.include_router(base_router)
app.include_router(data_router)
app.include_router(nlp.nlp_router)
app.include_router(debug_router)
app.include_router(risk_router, prefix="/api/v1")


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function to run the RAG system"""
    setup_logging()
    
    # Initialize configuration
    config = get_settings()
    
    # Verify data folder exists
    if not config.MAIN_FOLDER_PATH.exists():
        print(f"Data folder not found: {config.MAIN_FOLDER_PATH}")
        print("Please update MAIN_FOLDER_PATH in config.py")
        return
    
    # Initialize RAG system
    rag_system = NLPController(config)
    
    try:
        # Process data and create vector store
        rag_system.initialize_system()
        
        # Example queries
        test_queries = [
            "What is the reading for All pesticides as 'اسم المبيد Pesticide Name' in all positive readings?",
            "List readings where النتيجة (result) is 'غير مطابق' for pepper",
            "Show all non-compliant pesticide readings"
        ]
        
        print("\n" + "="*50)
        print("TESTING QUERIES")
        print("="*50)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 50)
            answer = rag_system.answer_query(query)
            print(answer)
            print("-" * 50)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()