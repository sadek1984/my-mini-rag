from .providers import QdrantDBProvider
from .VectorDBEnum import VectorDBEnums
from ...controllers.BaseControllers import BaseControllers
from sqlalchemy.orm import sessionmaker

class VectorDBProviderFactory:
    def __init__(self, config, db_client: sessionmaker=None):
        self.config = config
        self.base_controller = BaseControllers()
        self.db_client = db_client

    def create(self, provider: str):
        if provider == VectorDBEnums.QDRANT.value:
            db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_PATH)

            return QdrantDBProvider(
                db_client=db_path,
                distance_method=self.config.VECTOR_DB_DISTANCE_METHOD,
            ) 
        
        return None