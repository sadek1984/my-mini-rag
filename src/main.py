
from fastapi import FastAPI
from routes.base import base_router  # Fixed import
from routes.data import data_router  # Fixed import

app = FastAPI()
app.include_router(base_router)
app.include_router(data_router)