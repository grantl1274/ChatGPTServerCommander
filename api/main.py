import os
import sys
import time
import uvicorn
import webbrowser
import threading
import asyncio
from pathlib import Path
from fastapi import FastAPI, APIRouter, UploadFile, File, BackgroundTasks, HTTPException, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from schemas.chat_models import ChatRequest, ChatResponse, Action, WebSocketMessage, WebSocketResponse
from api.websocket_handler import handle_websocket, manager

# Add the project root to Python path so modules can be imported correctly
project_root = Path(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, str(project_root))

# Set the current working directory to the project root
os.chdir(project_root)

# Import LocalTunnel function
from scripts.localtunnel import start_localtunnel

# Initialize FastAPI app
app = FastAPI(title="DataHub API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 7000       # Change to match web application port
URL = f"http://localhost:{PORT}"

# Add new router
router = APIRouter(prefix="/api")  # Add prefix to all routes

# Add status endpoint
@router.get("/status")
async def get_status():
    """
    Get API server status and health information
    """
    import psutil
    import platform
    from datetime import datetime

    # Get system info
    process = psutil.Process()
    system_info = {
        "status": "online",
        "uptime": time.time() - process.create_time(),
        "server_time": datetime.now().isoformat(),
        "api_version": "1.0",
        "platform": platform.system(),
        "python_version": sys.version,
        "memory_usage": {
            "percent": process.memory_percent(),
            "rss": process.memory_info().rss / (1024 * 1024),  # MB
        },
        "cpu_usage": process.cpu_percent(),
        "port": PORT,
        "host": HOST
    }
    
    return JSONResponse(system_info)

# Simulate file processing with progress tracking
async def process_file_with_progress(file: UploadFile, background_tasks: BackgroundTasks, client_id: str):
    total_size = len(file.file.read())
    file.file.seek(0)  # Reset file pointer to beginning

    # Fake processing logic (replace with real processing logic)
    processed = 0
    chunk_size = 1024 * 1024  # 1 MB chunks
    while processed < total_size:
        await asyncio.sleep(1)  # Simulate processing delay
        processed += chunk_size
        if processed > total_size:
            processed = total_size

        # Send progress update
        progress = (processed / total_size) * 100
        await manager.send_progress_update(client_id, progress / 100, "Processing file...")

    return processed, total_size

@router.post("/documents/upload-with-progress")
async def upload_document_with_progress(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = BackgroundTasks(),
    client_id: str = None
):
    try:
        if not client_id:
            raise HTTPException(status_code=400, detail="Client ID is required for progress tracking")
            
        # Start processing the file in the background
        background_tasks.add_task(process_file_with_progress, file, background_tasks, client_id)
        return JSONResponse(
            status_code=200, 
            content={
                "message": "File processing started",
                "client_id": client_id
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the file: {str(e)}")

# WebSocket endpoint
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await handle_websocket(websocket, client_id)

# Include the router in the main app
app.include_router(router)

def open_browser():
    """Open browser after a short delay to ensure server has started"""
    time.sleep(2)  # Wait 2 seconds for the server to start
    print(f"Opening browser to {URL}")
    webbrowser.open(URL)

if __name__ == "__main__":
    print("=" * 60)
    print(f"Starting API server from: {project_root}")
    print("Using real data (not fake/mock data)")
    print(f"API will be available at: {URL}")
    print("=" * 60)

    # Start browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Start localtunnel for public access
    start_localtunnel(port=PORT, subdomain="slimy-kids-travel")

    try:
        # Run the uvicorn server with the API app
        uvicorn.run(
            "api.main:app", 
            host=HOST, 
            port=PORT, 
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("Server stopped by user.")
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Falling back to simple server mode...")

        # Fallback to simple_server.py if API server fails
        try:
            import subprocess
            subprocess.Popen([sys.executable, "simple_server.py"])
            print("Simple server started as fallback.")
        except Exception as fallback_error:
            print(f"Failed to start fallback server: {fallback_error}")
            print("Please restart manually.")

        # Keep the console open on error so user can see the message
        input("Press Enter to exit...")
