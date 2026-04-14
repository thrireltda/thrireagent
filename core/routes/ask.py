import os
from fastapi import APIRouter, BackgroundTasks
from typing import Dict
from core.engine import Engine

router = APIRouter()
engine = Engine()
@router.post("/ask")
def _(query: Dict[str, str], background_tasks: BackgroundTasks):
    t = query.get("text")
    if not t: return {}
    profile_file_path = str(os.getenv("PROFILE_FILE_PATH"))
    model_name = str(os.getenv("MODEL_NAME"))
    cortex_endpoint = str(os.getenv("CORTEX_ENDPOINT"))
    return engine.process(profile_file_path, t, model_name, cortex_endpoint, background_tasks)