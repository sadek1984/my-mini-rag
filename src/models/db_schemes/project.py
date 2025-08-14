from pydantic import BaseModel, Field, validator
from typing import Optional
from bson.objectid import ObjectId

class Project(BaseModel):
    id: Optional[ObjectId] = Field(None, alias="_id")
    project_id: str = Field(..., min_length=1)

    @validator('project_id')
    def validate_project_id(cls, value):
        if not value.isalnum():
            raise ValueError('project_id must be alphanumeric')
        
        return value

    class Config:
        arbitrary_types_allowed = True

    @classmethod # decorator for make static method for easy calling
    def get_indexes(cls): # cls refer to class not object
        return[{"key":[("project_id", 1)], # 1 for ascending , -1 for decending
                "name": "project_id_index_1", 
                "unique": True
               }]