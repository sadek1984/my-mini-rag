from .BaseControllers import BaseControllers 
from fastapi import UploadFile
from models import ResponseSignal

class ProjectControllers(BaseControllers):
    def _init_(self):
        super().__init__() 
    def get_project_path(self, project_id: str):
        project_dir = os.path.join(self.files_dir, project_id)
        if not os.path.exists(project_dir):
            os.makedirs(project_dir)
        return project_dir
        