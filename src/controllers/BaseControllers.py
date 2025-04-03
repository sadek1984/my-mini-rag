from helper.config import get_settings, Settings
import os

class BaseControllers:
    def __init__(self):
        self.app_settings = get_settings()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.files_dir = os.path.join(self.base_dir, "assets/files")  # Correct variable name
        os.makedirs(self.files_dir, exist_ok=True)  # Ensure directory exists