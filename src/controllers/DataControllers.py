from .BaseControllers import BaseControllers
from fastapi import UploadFile
from helper.config import get_settings, Settings
from models import ResponseSignal

class DataControllers(BaseControllers):
    def __init__(self):
        super().__init__()
        self.size_scale = 1048576 # convert MB to bytes

    def validate_uploaded_file(self, file: UploadFile):
        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value
        if file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED.value
        return True, ResponseSignal.FILE_VALIDATED_SUCCESS.value


