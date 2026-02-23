from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import cv2
import base64
import asyncio
import json
import time
import threading
import os
from pathlib import Path
from typing import Set, List, Dict, Optional, Tuple, Any
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from tracker import ObjectTracker, CameraCapture
from database import DatabaseManager
from verification_manager import VerificationManager
from improved_video_recording import RecordingManager
from improved_tracking_manager import ImprovedTrackingManager
import ai_broker as ai_broker

# ---------------- FASTAPI SETUP ---------------- #

app = FastAPI(title="Object Tracking Stream", version="3.0.0")

# Global instances
tracker: ObjectTracker = None
camera: CameraCapture = None
verification_manager: VerificationManager = None
recording_manager: RecordingManager = None
db_manager: DatabaseManager = None
tracking_manager: ImprovedTrackingManager = None
active_connections: Set[WebSocket] = set()

frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

@app.on_event("startup")
async def startup_event():
    global tracker, camera, verification_manager, recording_manager, db_manager, tracking_manager
    print("\n" + "="*70)
    print("🚀 Starting Wildlife Tracking Server with Verification System")
    print("="*70)
    
    try:
        # 1. Initialize YOLO tracker
        print("\n[1/6] Loading YOLO model...")
        tracker = ObjectTracker()
        
        # 2. Initialize camera
        print("[2/6] Initializing camera...")
        camera = CameraCapture()
        
        # 3. Initialize database manager
        print("[3/6] Setting up database...")
        backend_dir = Path(__file__).parent
        db_path = str(backend_dir / "tracking_data.db")
        db_manager = DatabaseManager(db_path=db_path)
        
        # 4. Initialize VerificationManager (file organization)
        print("[4/6] Initializing verification system...")
        verification_manager = VerificationManager(
            base_dir="events",
            retention_days=7,        # Keep rejected files for 7 days
            cleanup_interval=3600    # Cleanup every hour
        )
        
        # 5. Initialize RecordingManager (with verification support)
        print("[5/6] Initializing recording manager...")
        recording_manager = RecordingManager(
            verification_manager=verification_manager,
            fps=Config.TARGET_FPS,
            buffer_seconds=5,
            timeout_seconds=5,
            enabled=True
        )
        
        # 6. Initialize ImprovedTrackingManager (AI coordination)
        print("[6/6] Initializing tracking manager...")
        tracking_manager = ImprovedTrackingManager(
            db_manager=db_manager,
            recording_manager=recording_manager,
            verification_manager=verification_manager,
            enable_ai=True,
            ai_timeout=30.0,
            max_retries=3
        )
        
        print("\n" + "="*70)
        print("✓ Server initialization complete")
        print("="*70)
        print(f"📂 Directory Structure:")
        print(f"   Pending:  events/pending/")
        print(f"   Verified: events/verified/")
        print(f"   Rejected: events/rejected/")
        print(f"\n💾 Database: {db_path}")
        print(f"🤖 VLM Mode: {Config.VLM_MODE.upper()}")
        print(f"🔄 AI Retries: 3 attempts per detection")
        print(f"\n📡 Server running at http://{Config.HOST}:{Config.PORT}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.on_event("shutdown")
async def shutdown_event():
    global camera, recording_manager, verification_manager, tracking_manager, active_connections
    
    print("\n🛑 Shutting down server...")
    
    # Close all websocket connections
    for connection in active_connections.copy():
        try: 
            await connection.close()
        except: 
            pass
    active_connections.clear()
    print("  ✓ Closed websocket connections")
    
    # Stop tracking manager (stops AI worker thread)
    if tracking_manager:
        tracking_manager.stop()
        print("  ✓ Stopped tracking manager")
    
    # Stop all active recordings
    if recording_manager:
        recording_manager.stop_all()
        print("  ✓ Stopped recording manager")
    
    # Stop verification manager (stops cleanup thread)
    if verification_manager:
        verification_manager.stop()
        print("  ✓ Stopped verification manager")
    
    # Release camera resources
    if camera:
        camera.release()
        print("  ✓ Released camera")
    
    print("✓ Cleanup complete\n")

@app.get("/")
async def read_root():
    html_file = frontend_path / "index.html"
    return FileResponse(html_file)

@app.get("/health")
async def health_check():
    """Health check endpoint with detailed system status."""
    return {
        "status": "healthy",
        "camera_opened": camera.is_opened() if camera else False,
        "active_connections": len(active_connections),
        "recording_stats": recording_manager.get_stats() if recording_manager else {},
        "verification_stats": verification_manager.get_stats() if verification_manager else {},
        "tracking_stats": tracking_manager.get_stats() if tracking_manager else {},
        "timestamp": datetime.now().isoformat()
    }

# ---------------- API ENDPOINTS ---------------- #

@app.get("/api/events")
async def list_events() -> List[Dict]:
    """
    List all recorded events from the verification system.
    Returns events from: pending/, verified/{species}/, and rejected/
    """
    if not verification_manager:
        raise HTTPException(status_code=503, detail="Verification manager not initialized")
    
    events = []
    
    # 1. Get verified tracks (organized by species)
    verified_tracks = verification_manager.get_verified_tracks()
    for track_data in verified_tracks:
        track_id = track_data.get("track_id")
        # Extract relative path from track_dir
        if "track_dir" in track_data:
            track_dir = Path(track_data["track_dir"])
            # The path is relative to "events" directory
            try:
                rel_path = track_dir.relative_to(verification_manager.base_dir)
                track_data["relative_path"] = str(rel_path).replace("\\", "/") # Ensure forward slashes for URL
            except ValueError:
                track_data["relative_path"] = str(track_dir)
        
        # Enrich with database info
        if db_manager and track_id:
            db_obj = db_manager.get_tracking_object(track_id)
            if db_obj:
                track_data["ai_info"] = db_obj.get("ai_info")
                track_data["frame_snapshot"] = db_obj.get("frame_snapshot")
                track_data["verification_status"] = "verified"
        
        events.append(track_data)
    
    # 2. Get pending tracks (awaiting AI verification)
    pending_tracks = verification_manager.get_pending_tracks()
    for track_data in pending_tracks:
        track_id = track_data.get("track_id")
        # Extract relative path
        if "track_dir" in track_data:
            track_dir = Path(track_data["track_dir"])
            try:
                rel_path = track_dir.relative_to(verification_manager.base_dir)
                track_data["relative_path"] = str(rel_path).replace("\\", "/")
            except ValueError:
                track_data["relative_path"] = str(track_dir)
        
        if db_manager and track_id:
            db_obj = db_manager.get_tracking_object(track_id)
            if db_obj:
                track_data["ai_info"] = db_obj.get("ai_info")
                track_data["frame_snapshot"] = db_obj.get("frame_snapshot")
                track_data["verification_status"] = db_obj.get("verification_status", "pending")
        
        events.append(track_data)
    
    # Sort by timestamp (most recent first)
    events.sort(key=lambda x: x.get("start_time", ""), reverse=True)
    return events

@app.get("/api/events/{event_path:path}/video")
async def stream_event_video(event_path: str):
    """
    Stream video from new directory structure.
    
    Examples:
      - verified/Red_Fox/2026-02-13_14-23-45_track_123
      - pending/track_123
      - rejected/track_456
    """
    base_dir = Config.EVENTS_DIR
    video_file = base_dir / event_path / "video.mp4"
    
    if not video_file.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(path=video_file, media_type="video/mp4")

@app.get("/api/events/{event_path:path}/metadata")
async def get_event_metadata(event_path: str):
    """
    Get metadata from event directory.
    
    Args:
        event_path: Path like "verified/Red_Fox/2026-02-13_14-23-45_track_123"
    """
    base_dir = Config.EVENTS_DIR
    metadata_file = base_dir / event_path / "metadata.json"


    if not metadata_file.exists():
        raise HTTPException(status_code=404, detail="Metadata not found")
        
    with open(metadata_file, 'r') as f:
        data = json.load(f)
        
        # Enrich with database AI info
        if db_manager and "track_id" in data:
            obj = db_manager.get_tracking_object(data["track_id"])
            if obj:
                if obj.get("ai_info"):
                    data["ai_info"] = obj["ai_info"]
                data["frame_snapshot"] = obj.get("frame_snapshot")
                data["verification_status"] = obj.get("verification_status")
                
        return data

@app.get("/api/history")
async def get_history() -> List[Dict]:
    """
    Get unique species history from the last 10 minutes.
    """
    if not db_manager:
        return []
    return db_manager.get_unique_species_history(minutes=10)

@app.get("/api/verification/stats")
async def get_verification_stats():
    """
    Get comprehensive statistics about the verification system.
    Includes: verification, tracking, database, and recording stats.
    """
    if not verification_manager:
        raise HTTPException(status_code=503, detail="Verification manager not initialized")
    
    vm_stats = verification_manager.get_stats()
    tm_stats = tracking_manager.get_stats() if tracking_manager else {}
    db_stats = db_manager.get_stats() if db_manager else {}
    rec_stats = recording_manager.get_stats() if recording_manager else {}
    
    return {
        "verification": vm_stats,
        "tracking": tm_stats,
        "database": db_stats,
        "recording": rec_stats,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/config/vlm_mode")
async def get_vlm_mode():
    """Get current VLM mode."""
    return {"mode": ai_broker.get_vlm_mode()}

@app.post("/api/config/vlm_mode")
async def set_vlm_mode(data: Dict[str, str]):
    """Change VLM mode (qween_cloud, etc.)."""
    mode = data.get("mode")
    if not mode:
        raise HTTPException(status_code=400, detail="Mode is required")
    
    if ai_broker.set_vlm_mode(mode):
        # Broadcast to all clients
        for websocket in active_connections:
            try:
                await websocket.send_json({
                    "type": "vlm_mode_updated",
                    "data": {"mode": mode}
                })
            except:
                pass
        return {"status": "success", "mode": mode}
    else:
        raise HTTPException(status_code=400, detail="Invalid mode")

# ---------------- API QUERY ENDPOINT ---------------- #

from pydantic import BaseModel
import agents

class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def chat_query(request: ChatRequest):
    """
    Process natural language query using agents.py orchestration.
    Converts natural language to SQL, executes against database, and returns results.
    """
    try:
        print(f"📝 Processing chat query: {request.query}")
        
        # Invoke the agent workflow
        initial_state = {
            "query": request.query,
            "messages": [],
            "attempts": 0,
            "sql_query": "",
            "data": [],
            "error": None
        }
        
        
        result = agents.app.invoke(initial_state)
        
        # Extract results
        sql_query = result.get("sql_query", "")
        data = result.get("data", [])
        error = result.get("error")
        messages = result.get("messages", [])
        content = result.get("content", "")
        
        # Convert NaN values to None for JSON compliance
        import math
        if data:
            for row in data:
                for key, value in row.items():
                    if isinstance(value, float) and math.isnan(value):
                        row[key] = None
        
        # Format response
        response = {
            "success": not error and len(data) > 0,
            "query": request.query,
            "sql_query": sql_query,
            "data": data,
            "error": error,
            "messages": messages,
            "content": content,
            "row_count": len(data) if data else 0
        }
        
        print(f"✓ Query processed: {len(data)} rows returned")
        return response
        
    except Exception as e:
        print(f"✗ Chat query failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            # "query": request.query,
            "sql_query": "",
            "data": [],
            "error": str(e),
            "messages": [f"Error processing query: {str(e)}"],
            "row_count": 0
        }

# ---------------- WEBSOCKET STREAM ---------------- #

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global active_connections
    await websocket.accept()
    active_connections.add(websocket)
    
    # Register with tracking manager for lifecycle events
    if tracking_manager:
        tracking_manager.register_websocket(websocket)
        
    print(f"✓ New WebSocket connection (Total: {len(active_connections)})")
    
    try:
        await websocket.send_json({"type": "config", "data": Config.get_info()})
        
        while True:
            if not camera or not camera.is_opened():
                await asyncio.sleep(1)
                continue
            
            success, frame = camera.read()
            if not success:
                await asyncio.sleep(0.1)
                continue
            
            # Track objects - Returns annotated frame (with boxes) and metadata
            annotated_frame, metadata = tracker.process_frame(frame, track=True)
            
            # Extract detections
            detections_list = []
            if 'detections' in metadata:
                detections_list = metadata['detections']
            
            # Process tracking with ORIGINAL frame (for clean AI snapshots)
            # MUST be done BEFORE recording to apply ID remapping (persistence)
            if tracking_manager:
                await tracking_manager.process_detections(frame, detections_list)

            # Record ANNOTATED frame (with bounding boxes for visualization)
            if recording_manager:
                recording_manager.update_tracks(annotated_frame, detections_list)
            
            # Encode annotated frame for live stream
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), Config.JPEG_QUALITY]
            success, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)
            
            if success:
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                # Add recording stats to metadata
                rec_stats = recording_manager.get_stats() if recording_manager else {}
                metadata['is_recording'] = rec_stats.get('active_recordings', 0) > 0
                metadata['recording_info'] = rec_stats
                
                # Add verification stats
                if verification_manager:
                    vm_stats = verification_manager.get_stats()
                    metadata['verification_info'] = {
                        'pending': vm_stats.get('pending', 0),
                        'verified': vm_stats.get('verified', 0),
                        'rejected': vm_stats.get('rejected', 0)
                    }
                
                await websocket.send_json({
                    "type": "frame",
                    "image": jpg_as_text,
                    "metadata": metadata
                })
            
            await asyncio.sleep(1 / Config.TARGET_FPS)
            
    except WebSocketDisconnect:
        print(f"✗ WebSocket disconnected")
    except Exception as e:
        print(f"✗ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        active_connections.discard(websocket)
        if tracking_manager:
            tracking_manager.unregister_websocket(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        ws_ping_interval=Config.WEBSOCKET_PING_INTERVAL,
        ws_ping_timeout=Config.WEBSOCKET_PING_TIMEOUT
    )
