
from ..LLMInterface import LLMInterface
import openai
from ..LLMEnum import OpenAIEnums
import logging

class OpenAIProvider(LLMInterface):
    # only working during startup of application from main.py
    def __init__(self, api_key: str, api_url: str = None,
                 default_input_max_characters: int = 1000,  # this value to control cost 
                 default_generation_max_output_tokens: int = 1000,
                 default_generation_temperature: float = 0.1,
                 model_id: str = "gpt-3.5-turbo"
                 ):

        self.model_id = model_id  # Make sure this is set

        
         # Debug initialization
        print(f"\n🔧 OpenAIProvider Initialized 🔧")
        print(f"Configured Model: {self.model_id}")
        
        
        self.api_key = api_key
        self.api_url = api_url
        self.available_models = ["gpt-3.5-turbo", "gpt-4"]
        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        openai.api_key = self.api_key
        if self.api_url:
            openai.api_base = self.api_url

        self.enums = OpenAIEnums
 
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(self, prompt: str, chat_history: list = [],
                      max_output_tokens: int = None, temperature: float = None):
        if not self.generation_model_id:
            self.logger.error("Generation model for OpenAI was not set")
            return None

        max_output_tokens = max_output_tokens if max_output_tokens else self.default_generation_max_output_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append(
            self.construct_prompt(prompt=prompt, role=OpenAIEnums.USER.value)
        )

        try:
            response = openai.ChatCompletion.create(
                model=self.generation_model_id,
                messages=chat_history,
                max_tokens=max_output_tokens,
                temperature=temperature
            )
        except Exception as e:
            self.logger.error(f"Error while generating text with OpenAI: {e}")
            return None

        if not response or not response.choices or not response.choices[0].message:
            self.logger.error("Invalid response from OpenAI")
            return None

        return response.choices[0].message.content

    def embed_text(self, text: str, document_type: str = None):
        if not self.embedding_model_id:
            self.logger.error("Embedding model for OpenAI was not set")
            return None

        try:
            response = openai.Embedding.create(
                model=self.embedding_model_id,
                input=text,
            )
        except Exception as e:
            self.logger.error(f"Error while embedding text with OpenAI: {e}")
            return None

        if not response or not response['data'] or not response['data'][0]['embedding']:
            self.logger.error("Invalid embedding response from OpenAI")
            return None

        return response['data'][0]['embedding']

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
