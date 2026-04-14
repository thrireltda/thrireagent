from fastapi import FastAPI
from core.routes.ask import router as ask_router

app = FastAPI()
app.include_router(ask_router)