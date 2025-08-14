# src/controllers/ProcessControllers.py

from .BaseControllers import BaseControllers
from .ProjectControllers import ProjectControllers
import os
import nltk
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredExcelLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def _ensure_nltk_data():
    """
    Checks if the required NLTK data packages are available. If not, it downloads them.
    This version is more robust and specifically includes 'punkt_tab' as required by the error log.
    """
    # A dictionary mapping the download name to the resource path for verification.
    required_resources = {
        'punkt': 'tokenizers/punkt',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
        'punkt_tab': 'tokenizers/punkt_tab'  # Explicitly added to fix the error.
    }
    for download_name, resource_path in required_resources.items():
        try:
            # Check if the resource is already available using its specific path.
            nltk.data.find(resource_path)
            print(f"NLTK resource '{resource_path}' already available.")
        except LookupError:
            # If not available, download it using its package name.
            print(f"NLTK resource '{resource_path}' not found. Downloading '{download_name}'...")
            nltk.download(download_name)
            print(f"NLTK package '{download_name}' downloaded successfully.")

class ProcessControllers(BaseControllers):
    def __init__(self, project_id: str):
        super().__init__()
        self.project_id = project_id
        self.project_path = ProjectControllers().get_project_path(project_id=project_id)
        # Ensure NLTK data is ready before any processing happens.
        _ensure_nltk_data()

    def get_file_loader(self, file_id: str):
        """
        Determines the correct LangChain document loader based on the file extension.
        This version is more robust and handles different Excel extensions and cases.
        """
        # Get the file extension and convert it to lowercase to ensure the check is case-insensitive.
        file_ext = os.path.splitext(file_id)[-1].lower()
        file_path = os.path.join(self.project_path, file_id)

        # Check if the file actually exists before attempting to load it.
        if not os.path.exists(file_path):
            print(f"Error: File not found at path: {file_path}")
            return None

        # A dictionary mapping extensions to loaders for cleaner logic.
        loader_map = {
            '.pdf': PyMuPDFLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader, # Also handle older .xls format
            '.txt': TextLoader,
        }

        # Get the loader class from the map.
        loader_class = loader_map.get(file_ext)

        if loader_class:
            # Note: UnstructuredExcelLoader does not take an 'encoding' argument.
            # It accepts **unstructured_kwargs. For better granularity in RAG systems,
            # using mode="elements" is often a good choice, as it treats each cell as a document.
            if loader_class == UnstructuredExcelLoader:
                return UnstructuredExcelLoader(file_path, mode="elements")
            else:
                return loader_class(file_path)
        else:
            # If the extension is not in our map, it's an unsupported file type.
            print(f"Warning: Unsupported file type '{file_ext}' for file: {file_id}")
            return None

    def get_file_content(self, file_id: str):
        """
        Loads the file content using the appropriate loader.
        This now includes a check to ensure the loader was successfully created.
        """
        loader = self.get_file_loader(file_id=file_id)
        
        # This check prevents the "AttributeError: 'NoneType' object has no attribute 'load'"
        if loader is None:
            print(f"Error: Could not create a loader for {file_id}. The file type might be unsupported or the file may not exist.")
            # Return an empty list to handle this case gracefully downstream.
            return []
        
        try:  
            return loader.load()
        except Exception as e:
            print(f"Error loading file {file_id} with {type(loader).__name__}: {e}")
            return []

    def process_file_content(self, file_content: list, file_id: str,
                             chunk_size: int = 1000, overlap_size: int = 200):
        """
        Splits the loaded document content into chunks for processing.
        I've increased the default chunk_size and overlap_size for more context.
        """
        if not file_content:
            print(f"Warning: No content found in file {file_id} to process.")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap_size, length_function=len
        )
        
        # This approach safely extracts texts and their corresponding metadata.
        texts_to_split = [doc.page_content for doc in file_content if hasattr(doc, 'page_content') and doc.page_content]
        metadatas_to_split = [doc.metadata for doc in file_content if hasattr(doc, 'page_content') and doc.page_content]

        if not texts_to_split:
            print(f"Warning: No text content could be extracted from {file_id}.")
            return []

        chunks = text_splitter.create_documents(
            texts_to_split, metadatas=metadatas_to_split
        )
        return chunks
