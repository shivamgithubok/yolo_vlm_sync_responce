"""
Improved Tracking Manager with Verification Support

Key improvements:
- Integrates with VerificationManager for proper file organization
- Better error handling and retry logic
- Marks stability frames for better snapshot quality
- Atomic operations for verification/rejection
- Enhanced logging and monitoring
"""

import asyncio
import threading
import queue
import time
import cv2
import numpy as np
import base64
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import Config
from backend.database import DatabaseManager
import backend.ai_broker as ai_broker

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrackingManager")


class ImprovedTrackingManager:
    """
    Enhanced Tracking Manager with verification support.
    
    Implements:
    - Two-phase commit for recordings (pending → verified/rejected)
    - Better error handling with retry logic
    - Snapshot quality optimization
    - Atomic state transitions
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        recording_manager: Any = None,
        verification_manager: Any = None,
        enable_ai: bool = True,
        ai_timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Initialize improved tracking manager.
        
        Args:
            db_manager: Database manager instance
            recording_manager: Recording manager instance
            verification_manager: Verification manager instance
            enable_ai: Enable AI processing
            ai_timeout: Timeout for AI requests
            max_retries: Maximum AI retry attempts
        """
        self.db_manager = db_manager
        self.recording_manager = recording_manager
        self.verification_manager = verification_manager
        self.enable_ai = enable_ai
        self.ai_timeout = ai_timeout
        self.max_retries = max_retries
        
        # --- STATE MANAGEMENT ---
        self.active_track_ids: Set[int] = set()
        self.track_last_seen: Dict[int, datetime] = {}
        self.track_state: Dict[int, str] = {}  # 'new', 'stable', 'ai_queued', 'verified', 'rejected'
        self.ws_connections: Set[Any] = set()
        
        # --- TUNING PARAMS ---
        self.TRACK_PERSISTENCE_TIMEOUT = 5.0
        self.STABILITY_THRESHOLD = 1.0
        
        # --- AI QUEUE ---
        self.ai_queue = queue.PriorityQueue(maxsize=20)
        
        # Pending tracks waiting for stability
        # Format: {track_id: {'start_time': datetime, 'snapshot': b64, 'queued': bool, 'class_name': str, 'frame_count': int}}
        self.pending_tracks_stability = {} 
        
        # ID Remapping: {new_yolo_id: existing_db_id}
        self.id_remapping: Dict[int, int] = {}

        
        # Capture main event loop
        self.main_loop = asyncio.get_event_loop()
        
        self.running = True
        
        # Load data from DB
        self._load_active_tracks()
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
        self.worker_thread.start()
        
        logger.info("✓ ImprovedTrackingManager initialized")
        if self.verification_manager:
            logger.info("  ✓ Verification system: ENABLED")
        logger.info(f"  ✓ AI processing: {'ENABLED' if self.enable_ai else 'DISABLED'}")
        logger.info(f"  ✓ Max retries: {self.max_retries}")

    def _load_active_tracks(self):
        """Initialize with empty active tracks."""
        self.active_track_ids = set()
        logger.info("✓ Initialized with empty active tracks")

    # --- WEBSOCKET HANDLING ---
    def register_websocket(self, websocket):
        """Register a websocket connection for updates."""
        self.ws_connections.add(websocket)
    
    def unregister_websocket(self, websocket):
        """Unregister a websocket connection."""
        self.ws_connections.discard(websocket)
    
    async def broadcast_message(self, message: Dict):
        """Async broadcast to all clients."""
        disconnected = set()
        for ws in self.ws_connections:
            try:
                await ws.send_json(message)
            except Exception:
                disconnected.add(ws)
        for ws in disconnected:
            self.ws_connections.discard(ws)

    def _broadcast_threadsafe(self, message: Dict):
        """Helper to broadcast from sync worker thread."""
        if self.main_loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast_message(message), self.main_loop)

    # --- MAIN VIDEO LOOP (30 FPS) ---
    async def process_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]):
        """
        Process detections from the video loop.
        MUST BE FAST - only updates state, doesn't run AI.
        
        Args:
            frame: Current video frame
            detections: List of detection dictionaries
        """
        current_time = datetime.now()
        current_track_ids = set()
        
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id is None:
                continue
            
            # 1. Filter Humans
            class_name = detection.get('class_name', 'unknown')
            if self._is_human(class_name):
                continue
            
            # Apply ID Remapping if exists
            if track_id in self.id_remapping:
                original_id = track_id
                track_id = self.id_remapping[track_id]
                detection['track_id'] = track_id  # Update detection object for downstream consumers
                # logger.debug(f"Mapped {original_id} -> {track_id}")

            current_track_ids.add(track_id)
            self.track_last_seen[track_id] = current_time

            # 2. Handle New Tracks
            if track_id not in self.active_track_ids:
                await self._handle_new_track(track_id, class_name, frame, detection, current_time)
            else:
                # Update last seen
                self.db_manager.update_last_seen(track_id, current_time)
                
                # Update frame count for stability tracking
                if track_id in self.pending_tracks_stability:
                    self.pending_tracks_stability[track_id]['frame_count'] += 1
                    
                    # Mark as stability frame if we're in the stability window
                    if 'is_stable' not in detection:
                        detection['is_stable'] = True

            # 3. Check Stability for AI Queue
            if self.enable_ai and track_id in self.pending_tracks_stability:
                await self._check_stability_and_queue(track_id, frame, detection, current_time)

        # 4. Handle Disappeared Tracks
        disappeared = self.active_track_ids - current_track_ids
        for track_id in disappeared:
            last_seen = self.track_last_seen.get(track_id)
            if last_seen and (current_time - last_seen).total_seconds() > self.TRACK_PERSISTENCE_TIMEOUT:
                await self._handle_disappeared_track(track_id)
                
            # Clean up stale remappings
            stale_mappings = []
            for yolo_id, db_id in self.id_remapping.items():
                if db_id not in self.active_track_ids and (current_time - self.track_last_seen.get(db_id, datetime.min)).total_seconds() > self.TRACK_PERSISTENCE_TIMEOUT:
                     stale_mappings.append(yolo_id)
            
            for yolo_id in stale_mappings:
                self.id_remapping.pop(yolo_id, None)
    
    def _is_human(self, class_name: str) -> bool:
        """Check if class name represents a human."""
        human_keywords = ["person", "people", "human", "man", "woman", "child", "boy", "girl"]
        return class_name.lower() in human_keywords
    
    async def _check_stability_and_queue(
        self, 
        track_id: int, 
        frame: np.ndarray, 
        detection: Dict, 
        current_time: datetime
    ):
        """
        Check if track is stable enough for AI processing.
        
        Args:
            track_id: Track ID
            frame: Current frame
            detection: Detection dictionary
            current_time: Current timestamp
        """
        track_data = self.pending_tracks_stability[track_id]
        
        duration = (current_time - track_data['start_time']).total_seconds()
        frame_count = track_data['frame_count']
        
        # Check stability: must exist for STABILITY_THRESHOLD seconds AND have enough frames
        is_stable = (
            duration > self.STABILITY_THRESHOLD and 
            frame_count >= int(self.STABILITY_THRESHOLD * 30)  # Assuming 30 FPS
        )
        
        if is_stable and not track_data['queued']:
            # Update snapshot with a better one if possible
            # Take snapshot from current frame (which is in the stability window)
            new_snapshot = self._extract_frame_crop(frame, detection)
            if new_snapshot:
                track_data['snapshot'] = new_snapshot
            
            logger.info(f"🚀 Track {track_id} is stable ({duration:.1f}s, {frame_count} frames). Queuing for AI.")
            track_data['queued'] = True
            
            # Update state
            self.track_state[track_id] = 'ai_queued'
            
            try:
                # Priority 1 (can be adjusted based on class type)
                job = (1, track_id, track_data['class_name'], track_data['snapshot'])
                self.ai_queue.put_nowait(job)
            except queue.Full:
                logger.warning(f"⚠️ AI Queue Full! Dropping AI request for Track {track_id}")

    async def _handle_new_track(
        self, 
        track_id: int, 
        class_name: str, 
        frame: np.ndarray, 
        detection: Dict, 
        timestamp: datetime
    ):
        """
        Register new track in DB and prepare for AI verification.
        
        Args:
            track_id: Track ID
            class_name: YOLO-detected class name
            frame: Current frame
            detection: Detection dictionary
            timestamp: Detection timestamp
        """
        logger.info(f"🆕 New track: {track_id} ({class_name})")
        
        # FIX #1: Check if track already exists in database (race condition prevention)
        # FIX #1: Check if track already exists in database (race condition prevention)
        # OR Check if we should resurrect an old track (Persistence Logic)
        
        # 1a. Check for recent verified track of same class (Persistence)
        recent_track = self.db_manager.get_recent_verified_track(class_name, minutes=5)
        
        if recent_track:
            old_track_id = recent_track['track_id']
            ai_name = recent_track.get('ai_common_name') or recent_track.get('class_name')
            
            logger.info(f"🔄 Resurrecting persistent track {old_track_id} ({ai_name}) for new detection {track_id}")
            
            # Create mapping
            self.id_remapping[track_id] = old_track_id
            
            # Update detection object immediately for this frame
            detection['track_id'] = old_track_id
            
            # Add to active tracks
            self.active_track_ids.add(old_track_id)
            self.track_last_seen[old_track_id] = timestamp
            self.track_state[old_track_id] = 'verified' # It's already verified
            
            # Notify UI that this track is active again
            await self.broadcast_message({
                "type": "track_resumed",
                "data": {
                    "track_id": old_track_id,
                    "class_name": ai_name,
                    "frame_snapshot": recent_track.get('frame_snapshot')
                }
            })

            # FIX for Imposter Handling: Queue for Re-Verification
            # Even though we trust it, we verify it again to catch "Identity Theft" (e.g., Person walking by)
            if old_track_id not in self.pending_tracks_stability:
                self.pending_tracks_stability[old_track_id] = {
                    'start_time': timestamp,
                    'snapshot': recent_track.get('frame_snapshot', ''),
                    'class_name': class_name, # Use NEW YOLO class for verification
                    'queued': False,
                    'frame_count': 1,
                    'is_resurrected': True # Flag to handle imposters later
                }
            return

        existing = self.db_manager.get_tracking_object(track_id)
        if existing:
            logger.warning(f"⚠️ Track {track_id} already exists in DB (race condition), using existing entry")
            self.active_track_ids.add(track_id)
            
            # Still add to stability monitor if not already there
            if track_id not in self.pending_tracks_stability:
                self.pending_tracks_stability[track_id] = {
                    'start_time': timestamp,
                    'snapshot': existing.get('frame_snapshot', ''),
                    'class_name': class_name,
                    'queued': False,
                    'frame_count': 1
                }
            return
        
        # Extract initial snapshot
        snapshot = self._extract_frame_crop(frame, detection)
        
        if snapshot is None:
            logger.warning(f"⚠️ Failed to extract snapshot for track {track_id}. Skipping.")
            return
        
        # Get pending directory path if using verification
        video_path = None
        metadata_path = None
        if self.verification_manager:
            pending_dir = self.verification_manager.get_pending_dir(track_id)
            if pending_dir:
                video_path = str(pending_dir / "video.mp4")
                metadata_path = str(pending_dir / "metadata.json")
        
        # Create DB entry with 'pending' status
        row_id = self.db_manager.create_tracking_object(
            track_id=track_id,
            class_name=class_name,
            first_seen=timestamp,
            ai_info=None,
            frame_snapshot=snapshot,
            video_path=video_path,
            metadata_path=metadata_path
        )
        
        # FIX #1b: Check if creation actually succeeded
        if row_id is None:
            logger.error(f"❌ Failed to create DB entry for track {track_id}, aborting")
            return
        
        self.active_track_ids.add(track_id)
        self.track_state[track_id] = 'new'
        
        # Notify UI
        await self.broadcast_message({
            "type": "track_new", 
            "data": {
                "track_id": track_id, 
                "class_name": class_name, 
                "frame_snapshot": snapshot,
                "status": "pending"
            }
        })
        
        # Add to stability monitor
        self.pending_tracks_stability[track_id] = {
            'start_time': timestamp,
            'snapshot': snapshot,
            'class_name': class_name,
            'queued': False,
            'frame_count': 1
        }
        
        # Start recording via recording manager
        if self.recording_manager:
            # Recording manager will use verification manager to create pending directory
            pass  # Already handled by update_tracks in recording_manager

    async def _handle_disappeared_track(self, track_id: int):
        """
        Handle track that has disappeared from view.
        
        Args:
            track_id: Track ID that disappeared
        """
        logger.info(f"👋 Track disappeared: {track_id}")
        
        # Check if it was ever queued for AI
        track_data = self.pending_tracks_stability.get(track_id)
        
        if track_data and not track_data['queued']:
            # Never made it to AI - cancel everything
            logger.info(f"🗑️  Track {track_id} disappeared before AI processing - canceling")
            
            # Cancel recording
            if self.recording_manager:
                self.recording_manager.cancel_recording(track_id)
            
            # Delete from DB ONLY if it's a new track (not resurrected)
            if not track_data.get('is_resurrected'):
                self.db_manager.delete_track(track_id)
            else:
                logger.info(f"🛡️  Preserving persistent track {track_id} (resurrected) despite cancellation")
            
            # Cancel in verification manager
            if self.verification_manager:
                self.verification_manager.cancel_pending(track_id)
                
            # NOTIFY UI TO REMOVE CARD
            self._broadcast_threadsafe({
                "type": "track_removed",
                "data": {"track_id": track_id}
            })

        
        elif self.track_state.get(track_id) == 'verified':
            # It was a verified track (resurrected) that just finished.
            # Recording is in 'pending/', we need to move it to 'verified/'
            logger.info(f"💾 Persistent track {track_id} finished. Archiving new segment.")
            
            # 1. Stop recording ensuring metadata is written
            if self.recording_manager:
                self.recording_manager.stop_recording(track_id)
                time.sleep(1.0) # Wait for file release
            
            # 2. Get info from DB
            db_obj = self.db_manager.get_tracking_object(track_id)
            if db_obj and self.verification_manager:
                class_name = db_obj.get('ai_common_name') or db_obj.get('class_name')
                safe_name = "".join([c if c.isalnum() else "_" for c in class_name])
                
                # 3. Move file
                verified_path = self.verification_manager.verify_and_move(
                    track_id, safe_name, db_obj.get('ai_info')
                )
                
                if verified_path:
                    # 4. Update DB to point to NEWest clip
                    self.db_manager.update_paths(
                        track_id,
                        video_path=str(verified_path / "video.mp4"),
                        metadata_path=str(verified_path / "metadata.json")
                    )
                    
                    # 5. Update Last Seen one last time (already done in loop, but good measure)
                    self.db_manager.update_last_seen(track_id, datetime.now())

        # Cleanup state
        self.active_track_ids.discard(track_id)
        self.pending_tracks_stability.pop(track_id, None)
        self.track_state.pop(track_id, None)
        self.track_last_seen.pop(track_id, None)

    # --- AI WORKER THREAD ---
    def _ai_worker_loop(self):
        """
        Background worker thread for AI processing.
        Handles retries and verification/rejection logic.
        """
        logger.info("👷 AI Worker Thread Started")
        
        while self.running:
            try:
                # Get job (blocks for 1 sec)
                _, track_id, class_name, frame_snapshot = self.ai_queue.get(timeout=1.0)
                
                # Check validity
                if track_id not in self.active_track_ids:
                    logger.info(f"⏩ Skipping stale track {track_id}")
                    self.ai_queue.task_done()
                    continue

                # Execute AI analysis with retry logic
                self._execute_ai_analysis_with_retry(track_id, class_name, frame_snapshot)
                
                self.ai_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker Loop Error: {e}")

    def _execute_ai_analysis_with_retry(
        self, 
        track_id: int, 
        class_name: str, 
        frame_snapshot: str
    ):
        """
        Execute AI analysis with automatic retry on failure.
        
        Args:
            track_id: Track ID
            class_name: YOLO class name
            frame_snapshot: Base64 encoded snapshot
        """
        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                # Get context
                history_str = self._get_recent_history()
                
                logger.info(f"🤖 AI Processing track {track_id} (attempt {retry_count + 1}/{self.max_retries})")
                
                # Call AI broker
                wildlife_info = ai_broker.get_wildlife_info(
                    class_name, frame_snapshot, history_str
                )

                # Normalize response
                if hasattr(wildlife_info, "model_dump"):
                    ai_info_dict = wildlife_info.model_dump()
                else:
                    ai_info_dict = wildlife_info
                
                # Process result
                self._process_ai_result(track_id, class_name, ai_info_dict, frame_snapshot)
                return  # Success!
                
            except Exception as e:
                retry_count += 1
                last_error = str(e)
                logger.warning(f"⚠️ AI attempt {retry_count} failed for track {track_id}: {e}")
                
                if retry_count < self.max_retries:
                    time.sleep(1)  # Brief delay before retry
        
        # All retries failed
        logger.error(f"❌ AI processing failed after {self.max_retries} attempts for track {track_id}")
        self._handle_ai_failure(track_id, last_error)
    
    def _process_ai_result(
        self, 
        track_id: int, 
        original_class: str, 
        ai_info_dict: Dict, 
        frame_snapshot: str
    ):
        """
        Process successful AI result and update database/files.
        
        Args:
            track_id: Track ID
            original_class: Original YOLO class
            ai_info_dict: AI analysis result
            frame_snapshot: Base64 snapshot
        """
        # Check if it's actually an animal
        is_animal = ai_info_dict.get("is_animal", True)
        is_person = ai_info_dict.get("is_person", False)
        
        if not is_animal or is_person:
            reason = "identified as person" if is_person else "not an animal"
            logger.info(f"🚫 Track {track_id} rejected: {reason}")
            
            # Reject the track
            self._reject_track(track_id, reason)
            return
        
        # SUCCESS - It's a verified animal!
        common_name = ai_info_dict.get("commonName", original_class)
        scientific_name = ai_info_dict.get("scientificName", "")
        
        # Sanitize name for file system
        safe_name = "".join([c if c.isalnum() else "_" for c in common_name])
        
        logger.info(f"✅ Track {track_id} verified as: {common_name} ({scientific_name})")

        # --- IMPOSTER DETECTION logic for Resurrected Tracks ---
        # Check if this was a resurrected track that turned out to be something else
        # We need to check if the new 'common_name' matches the OLD 'common_name' in DB.
        
        # Get existing DB record to compare
        db_obj = self.db_manager.get_tracking_object(track_id)
        if db_obj and db_obj.get('verification_status') == 'verified':
            old_name = db_obj.get('ai_common_name')
            # Simple check: if names are significantly different
            if old_name and common_name.lower() != old_name.lower():
                logger.warning(f"🕵️ IMPOSTER DETECTED! Track {track_id} was {old_name}, but AI found {common_name}")
                self._handle_imposter(track_id, ai_info_dict, frame_snapshot, common_name, safe_name)
                return

        # Normal Flow continues...
        
        # FIX #1: Remove track from active list IMMEDIATELY to prevent new recordings
        logger.info(f"🔒 Removing track {track_id} from active list to prevent re-detection")
        self.active_track_ids.discard(track_id)
        self.pending_tracks_stability.pop(track_id, None)
        
        # 1. Update database FIRST
        self.db_manager.update_ai_info(track_id, ai_info_dict)
        self.db_manager.update_class_name(track_id, safe_name)
        
        # 2. Rename recording (updates in-memory class name)
        if self.recording_manager:
            self.recording_manager.rename_recording(track_id, safe_name)
        
        # FIX #2: STOP recording BEFORE accessing metadata file
        if self.recording_manager:
            logger.info(f"🛑 Stopping recording for track {track_id} (this creates metadata.json)")
            self.recording_manager.stop_recording(track_id)
            # Give Windows time to:
            # 1. FFmpeg to finish writing
            # 2. metadata.json to be created
            # 3. File locks to be released
            time.sleep(1.0)  # Increased from 0.5 to 1.0 seconds
        
        # FIX #3: NOW update metadata.json (it exists after stop())
        if self.recording_manager:
            self.recording_manager.update_metadata_on_disk(track_id, safe_name, ai_info_dict)
        
        # 3. NOW safe to move to verified directory (no file lock, no re-recording)
        if self.verification_manager:
            verified_path = self.verification_manager.verify_and_move(
                track_id, safe_name, ai_info_dict
            )
            
            if verified_path:
                # Update database with new path
                self.db_manager.update_paths(
                    track_id,
                    video_path=str(verified_path / "video.mp4"),
                    metadata_path=str(verified_path / "metadata.json")
                )
                logger.info(f"✅ Successfully moved track {track_id} to {verified_path}")
            else:
                logger.error(f"❌ Failed to move track {track_id} to verified directory")
        
        # 4. Update state
        self.track_state[track_id] = 'verified'
        
        # 5. Notify UI
        self._broadcast_threadsafe({
            "type": "track_verified",
            "data": {
                "track_id": track_id,
                "class_name": safe_name,
                "ai_info": ai_info_dict,
                "frame_snapshot": frame_snapshot
            }
        })
    
    def _reject_track(self, track_id: int, reason: str):
        """
        Reject a track (not an animal or person detected).
        
        Args:
            track_id: Track ID
            reason: Rejection reason
        """
        
        # Check for IMPOSTER during rejection (e.g. Person detected on a Dog track)
        # If we are rejecting a track that was supposedly "verified", it's an imposter situation.
        if self.track_state.get(track_id) == 'verified':
             logger.warning(f"🕵️ IMPOSTER DETECTED (During Rejection)! Track {track_id} rejected as {reason}")
             # We need to SPLIT this. The original valid animal track is safe (in verified). 
             # The CURRENT recording (in pending) belongs to the "Person/Imposter".
             
             # 1. Stop the 'pending' recording for the Old ID.
             if self.recording_manager:
                 self.recording_manager.stop_recording(track_id)
                 time.sleep(1.0) # Release lock
                 
             # 2. We don't want to corrupt the Old ID in DB.
             # So we will just discard this "pending" segment by NOT moving it to verified.
             # And we should notify UI that the "Resume" was cancelled/ended.
             
             logger.info(f"🛑 Cancelled RESUME for track {track_id} due to rejection. Original track preserved.")
             
             # Cleanup local state
             self.active_track_ids.discard(track_id)
             self.track_state.pop(track_id, None)
             self.pending_tracks_stability.pop(track_id, None)
             
             # We effectively 'delete' the imposter segment here by doing nothing with it.
             # (Or we could move it to rejected as a separate event, but that requires ID splitting logic similar to _handle_imposter)
             # For simplicity now: Just drop the imposter segment.
             
             # Cancel pending recording cleanups
             if self.verification_manager:
                 self.verification_manager.cancel_pending(track_id)
                 
             # NOTIFY UI: Remove the card (since we are dropping this resume attempt)
             self._broadcast_threadsafe({
                "type": "track_removed",
                "data": {"track_id": track_id}
             })
             return

        # Normal Rejection Logic
        logger.info(f"🚫 Rejecting track {track_id}: {reason}")
        self.active_track_ids.discard(track_id)
        self.track_state.pop(track_id, None)
        self.pending_tracks_stability.pop(track_id, None)
        
        # 1. Cancel/stop recording
        if self.recording_manager:
            self.recording_manager.cancel_recording(track_id)
            time.sleep(0.5)  # Give time for FFmpeg to cleanup
        
        # 2. Move to rejected directory
        if self.verification_manager:
            self.verification_manager.reject_and_move(track_id, reason)
        
        # 3. Notify UI -> Send 'track_removed' so frontend clears the card
        self._broadcast_threadsafe({
            "type": "track_removed",
            "data": {
                "track_id": track_id,
                "reason": reason
            }
        })

    def _handle_imposter(self, old_track_id: int, ai_info: Dict, snapshot: str, new_name: str, safe_name: str):
        """
        Handle case where a verified track was resumed but AI found a DIFFERENT animal/person.
        We must SPLIT the track: 
        1. Preserve Old ID (it was correct before).
        2. Create NEW ID for this current segment (the imposter).
        """
        # 1. Generate NEW ID
        new_track_id = int(time.time()) + 10000 # Pseudo-unique ID
        logger.info(f"✂️ Splitting Track! {old_track_id} -> New ID {new_track_id} ({new_name})")
        
        # 2. Stop Recording for Old ID (this finalizes video in 'pending/track_OLD')
        if self.recording_manager:
            self.recording_manager.stop_recording(old_track_id)
            time.sleep(1.0)
            
        # 3. RENAME 'pending/track_OLD' -> 'pending/track_NEW'
        if self.verification_manager:
            pending_old = self.verification_manager.get_pending_dir(old_track_id)
            pending_new = self.verification_manager.pending_dir / f"track_{new_track_id}"
            
            if pending_old and pending_old.exists():
                try:
                    # Rename directory
                    pending_old.rename(pending_new)
                    logger.info(f"📁 Renamed pending folder: {pending_old} -> {pending_new}")
                except Exception as e:
                    logger.error(f"❌ Failed to rename imposter folder: {e}")

        # 4. Create DB Entry for NEW ID takes the current Identity
        # (This is a verified animal, just a DIFFERENT one)
        row_id = self.db_manager.create_tracking_object(
            track_id=new_track_id,
            class_name=safe_name, # YOLO class might have been wrong, use AI name
            first_seen=datetime.now(),
            ai_info=ai_info,
            frame_snapshot=snapshot,
            video_path=None, # Will be set by verification move
            metadata_path=None
        )
        
        # 5. Move to Verified (using New ID)
        if self.verification_manager:
             # Need to manually trigger verify_and_move for the NEW ID
             verified_path = self.verification_manager.verify_and_move(
                new_track_id, safe_name, ai_info
             )
             if verified_path:
                 self.db_manager.update_paths(
                     new_track_id,
                     str(verified_path / "video.mp4"),
                     str(verified_path / "metadata.json")
                 )
                 # Update status
                 self.db_manager.update_verification_status(new_track_id, 'verified')
                 
        # 6. Cleanup State
        self.active_track_ids.discard(old_track_id)
        self.pending_tracks_stability.pop(old_track_id, None)
        
        # 7. Notify UI of Split
        # Frontend doesn't know 'track_split', so we send Remove Old + Add New
        self._broadcast_threadsafe({
            "type": "track_removed",
            "data": {"track_id": old_track_id}
        })
        
        self._broadcast_threadsafe({
            "type": "track_new",
            "data": {
                "track_id": new_track_id, 
                "class_name": safe_name, 
                "frame_snapshot": snapshot,
                "status": "verified", # It's verifiable immediately
                "ai_info": ai_info
            }
        })
    
    def _handle_ai_failure(self, track_id: int, error_message: str):
        """
        Handle AI processing failure after all retries.
        
        Args:
            track_id: Track ID
            error_message: Error description
        """
        # 1. Update database with error
        self.db_manager.increment_error_count(track_id, error_message)
        
        # 2. Mark in verification manager
        if self.verification_manager:
            self.verification_manager.mark_as_error(track_id, error_message)
        
        # 3. Update recording metadata
        if self.recording_manager:
            self.recording_manager.update_metadata_on_disk(
                track_id, 
                "ERROR", 
                {"error": error_message}
            )
        
        # 4. Update state
        self.track_state[track_id] = 'error'
        
        # 5. Notify UI
        self._broadcast_threadsafe({
            "type": "track_error",
            "data": {
                "track_id": track_id,
                "error": error_message
            }
        })
        
        logger.error(f"❌ Track {track_id} marked as ERROR after {self.max_retries} failed attempts")
    
    def _get_recent_history(self) -> Optional[str]:
        """Get recent animal detection history for context."""
        try:
            recent = self.db_manager.get_recent_animal_history(limit=2)
            if recent:
                return ", ".join([r['common_name'] for r in recent if r.get('common_name')])
        except Exception as e:
            logger.debug(f"Could not get history: {e}")
        return None
    
    # --- HELPER METHODS ---
    def _extract_frame_crop(self, frame: np.ndarray, detection: Dict) -> Optional[str]:
        """
        Extract and encode a cropped region from the frame.
        
        Args:
            frame: Full video frame
            detection: Detection dictionary with bbox
            
        Returns:
            Base64 encoded JPEG string or None
        """
        try:
            bbox = detection.get('bbox')
            if not bbox or len(bbox) != 4:
                return None
            
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Clamp to frame boundaries
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            w_obj = x2 - x1
            h_obj = y2 - y1
            
            # Validate size
            if w_obj < 20 or h_obj < 20:
                return None
            
            # Add padding (10%)
            px = int(w_obj * 0.1)
            py = int(h_obj * 0.1)
            
            x1_padded = max(0, x1 - px)
            y1_padded = max(0, y1 - py)
            x2_padded = min(w, x2 + px)
            y2_padded = min(h, y2 + py)
            
            # Extract crop
            crop = frame[y1_padded:y2_padded, x1_padded:x2_padded]
            
            if crop.size == 0:
                return None
            
            # Encode as JPEG
            success, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                return None
            
            return base64.b64encode(buffer).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Error extracting frame crop: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Get tracking manager statistics."""
        return {
            "active_tracks": len(self.active_track_ids),
            "pending_stability": len(self.pending_tracks_stability),
            "ai_queue_size": self.ai_queue.qsize(),
            "track_states": dict(
                (state, sum(1 for s in self.track_state.values() if s == state))
                for state in ['new', 'stable', 'ai_queued', 'verified', 'rejected', 'error']
            )
        }
    
    def stop(self):
        """Stop the tracking manager."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)
        logger.info("✓ ImprovedTrackingManager stopped")