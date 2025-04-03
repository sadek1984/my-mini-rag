from fastapi import FastAPI, APIRouter, Depends, UploadFile, File, status
from fastapi.responses import JSONResponse
import os
from helper.config import get_settings, Settings
from controllers import DataControllers, ProjectControllers

data_router = APIRouter(
    prefix='/api/v1/data',
    tags=['api_v1', 'data'],
)

@data_router.post('/upload/{project_id}')  # ✅ Fixed Path
async def upload_data(project_id: str, file: UploadFile = File(...),  
                      app_settings: Settings = Depends(get_settings)):
    # Check and validate the file properties in controllers
    is_valid, result_signal = DataControllers().validate_uploaded_file(file=file)
    project_dir_path = ProjectControllers().get_project_path(project_id=project_id)
    if not is_valid:  # ✅ Check error case first
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": result_signal
            }
        )
    
    return {"signal": result_signal, "valid": is_valid, "filename": file.filename, "project_id": project_id}

    #project_dir_path = ProjectControllers().get_project_path(project_id=project_id)
    