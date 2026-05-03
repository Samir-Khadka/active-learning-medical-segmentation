import os
import threading
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import datetime

app = FastAPI(title="Medical AL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global control state
stop_event = threading.Event()

# Enhanced global state for real-time tracking
experiment_status = {
    "status": "idle",
    "current_strategy": "N/A",
    "current_cycle": 0,
    "total_cycles": 4,
    "current_epoch": 0,
    "latest_dice": 0.0,
    "progress_percent": 0,
    "logs": [],
    "analysis_image_path": ""
}

class StatusResponse(BaseModel):
    status: str
    current_strategy: str
    current_cycle: int
    total_cycles: int
    current_epoch: int
    latest_dice: float
    progress_percent: int
    logs: List[str]
    analysis_image_path: Optional[str] = ""

@app.get("/status", response_model=StatusResponse)
async def get_status():
    return experiment_status

@app.post("/run")
async def start_run(background_tasks: BackgroundTasks):
    if experiment_status["status"] in ("running", "stopping"):
        return {"message": "Experiment already in progress or stopping — please wait"}
    
    stop_event.clear()
    # Reset status
    experiment_status.update({
        "status": "running",
        "current_strategy": "Initializing",
        "current_cycle": 0,
        "current_epoch": 0,
        "latest_dice": 0.0,
        "progress_percent": 0,
        "logs": ["Experiment started at " + str(datetime.datetime.now())],
        "analysis_image_path": ""
    })
    
    background_tasks.add_task(execute_experiment)
    return {"message": "Experiment started"}

@app.post("/stop")
async def stop_run():
    if experiment_status["status"] in ("running", "stopping"):
        stop_event.set()
        experiment_status["status"] = "stopping"
        if "Stop requested" not in " ".join(experiment_status["logs"]):
            experiment_status["logs"].append("Stop requested. Waiting for current cycle to finish...")
        return {"message": "Stop command sent"}
    return {"message": "No experiment running"}

def progress_callback(data: Dict[str, Any]):
    """
    Callback function that engine.py will call to update the API state.
    Carefully merges data to avoid overwriting the logs list accidentally.
    """
    # Extract and handle log separately before update to avoid key collision
    log_entry = data.pop("log", None)
    experiment_status.update(data)
    if log_entry:
        experiment_status["logs"].append(log_entry)
        if len(experiment_status["logs"]) > 100:
            experiment_status["logs"] = experiment_status["logs"][-100:]

def execute_experiment():
    from main import main_with_callback
    try:
        main_with_callback(progress_callback, stop_event)
        if stop_event.is_set():
            experiment_status["status"] = "stopped"
            experiment_status["logs"].append("Experiment stopped by user.")
        else:
            experiment_status["status"] = "completed"
            experiment_status["progress_percent"] = 100
            experiment_status["logs"].append("Experiment completed successfully!")
    except Exception as e:
        import traceback
        experiment_status["status"] = "error"
        experiment_status["logs"].append(f"CRITICAL ERROR: {str(e)}")
        experiment_status["logs"].append(traceback.format_exc())

app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
