from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId

class RetrievedDocument(BaseModel):
        text: str
        score: float


class DataChunk(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    # _id is private so make it alias for easy access
    chunk_text: str = Field(..., min_length=1)
    chunk_metadata: dict
    chunk_order: int = Field(..., gt=0)
    chunk_project_id: ObjectId
    chunk_asset_id: ObjectId

    class Config:
        arbitrary_types_allowed = True
    

    @classmethod # decorator for make static method for easy calling
    def get_indexes(cls): # cls refer to class not object
        return[{"key":[("chunk_project_id", 1)], # 1 for ascending , -1 for decending
                "name": "chunk_project_id_index_1",
                "unique": False
                }]
    
    
    