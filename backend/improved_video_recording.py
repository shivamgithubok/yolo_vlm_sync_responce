"""
Improved Video Recording Module with Verification Support

Key improvements:
- Records to pending/ directory initially
- Better snapshot timing (captures best frame during stability window)
- Supports file reorganization after AI verification
- Thread-safe metadata updates
"""

import cv2
import os
import json
import time
import threading
import subprocess
import shutil
import numpy as np
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging
from config import Config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VideoRecording")


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available."""
    return shutil.which("ffmpeg") is not None


class VideoRecorder:
    """
    Manages individual video recording session for a tracked object.
    Uses FFmpeg via subprocess to encode video with verification support.
    """
    
    def __init__(
        self, 
        track_id: int, 
        class_name: str, 
        frame_shape: Tuple[int, int], 
        fps: int, 
        track_dir: Path,
        pre_buffer_frames: List[np.ndarray] = None,
        verification_mode: bool = True
    ):
        """
        Initialize video recorder.
        
        Args:
            track_id: Unique tracking ID
            class_name: Initial class name (from YOLO)
            frame_shape: (height, width) of frames
            fps: Frames per second
            track_dir: Directory to save files (provided by VerificationManager)
            pre_buffer_frames: Pre-event frames to include
            verification_mode: If True, saves to pending directory
        """
        if not _ffmpeg_available():
            raise RuntimeError("FFmpeg is not installed or not in PATH.")

        self.track_id = track_id
        self.class_name = class_name
        self.fps = fps
        self.height, self.width = frame_shape
        self.track_dir = Path(track_dir)
        self.verification_mode = verification_mode
        
        # Calculate start time (accounting for buffer)
        buffer_duration = len(pre_buffer_frames) / fps if pre_buffer_frames else 0
        self.start_time = datetime.now() - timedelta(seconds=buffer_duration)
        
        # Files
        self.video_path = self.track_dir / "video.mp4"
        self.metadata_path = self.track_dir / "metadata.json"
        self.snapshot_path = self.track_dir / "snapshot.jpg"
        
        self.frame_count = 0
        self.is_active = True
        self.end_time = None
        self.lock = threading.Lock()
        
        # Snapshot selection - store multiple candidates and pick best later
        self.snapshot_candidates: List[Tuple[np.ndarray, float]] = []  # (frame, quality_score)
        self.max_snapshot_candidates = 5
        
        # # FFmpeg Process Initialization
        self.ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-profile:v", "main",
            "-level", "4.0",
            "-preset", "ultrafast", 
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(self.video_path)
        ]
        try:
            self.process = subprocess.Popen(
                self.ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            logger.info(f"📹 Started recording: track_{track_id}")
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            raise
        
        # Flush pre-buffer frames immediately
        if pre_buffer_frames:
            # Collect snapshot candidates from buffer
            mid_idx = len(pre_buffer_frames) // 2
            for i, frame in enumerate(pre_buffer_frames):
                self.write_frame(frame)
                
                # Add middle frames as snapshot candidates
                if abs(i - mid_idx) <= 2:
                    quality = self._calculate_frame_quality(frame)
                    self._add_snapshot_candidate(frame, quality)
    
    def _calculate_frame_quality(self, frame: np.ndarray) -> float:
        """
        Calculate frame quality score (higher is better).
        Uses Laplacian variance to measure sharpness.
        
        Args:
            frame: BGR frame
            
        Returns:
            Quality score (higher = sharper)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            return float(variance)
        except Exception:
            return 0.0
    
    def _add_snapshot_candidate(self, frame: np.ndarray, quality: float):
        """Add a frame as a potential snapshot candidate."""
        with self.lock:
            self.snapshot_candidates.append((frame.copy(), quality))
            
            # Keep only top N candidates by quality
            if len(self.snapshot_candidates) > self.max_snapshot_candidates:
                self.snapshot_candidates.sort(key=lambda x: x[1], reverse=True)
                self.snapshot_candidates = self.snapshot_candidates[:self.max_snapshot_candidates]
    
    def write_frame(self, frame: np.ndarray, is_stability_frame: bool = False):
        """
        Write a BGR frame to the FFmpeg pipe.
        
        Args:
            frame: BGR frame to write
            is_stability_frame: If True, consider this frame for snapshot
        """
        if not self.is_active or not self.process:
            return

        try:
            # Ensure frame matches initialized dimensions
            if frame.shape[0] != self.height or frame.shape[1] != self.width:
                frame = cv2.resize(frame, (self.width, self.height))

            self.process.stdin.write(frame.tobytes())
            self.frame_count += 1
            
            # Consider frames 5-15 as snapshot candidates
            if 5 <= self.frame_count <= 15 or is_stability_frame:
                quality = self._calculate_frame_quality(frame)
                self._add_snapshot_candidate(frame, quality)

        except BrokenPipeError:
            logger.warning(f"⚠️ FFmpeg pipe broken for track {self.track_id}")
            self.stop()
        except Exception as e:
            logger.error(f"⚠️ Error writing frame for track {self.track_id}: {e}")
    
    def _save_best_snapshot(self):
        """Save the best quality frame as snapshot."""
        with self.lock:
            if not self.snapshot_candidates:
                logger.warning(f"No snapshot candidates for track {self.track_id}")
                return False
            
            # Sort by quality and take the best
            self.snapshot_candidates.sort(key=lambda x: x[1], reverse=True)
            best_frame, quality = self.snapshot_candidates[0]
            
            try:
                cv2.imwrite(str(self.snapshot_path), best_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                logger.debug(f"Saved snapshot for track {self.track_id} (quality: {quality:.1f})")
                return True
            except Exception as e:
                logger.error(f"Error saving snapshot: {e}")
                return False
    
    def update_class_name(self, new_name: str):
        """
        Update the class name metadata (e.g. after AI analysis).
        Thread-safe.
        """
        with self.lock:
            self.class_name = new_name

    def stop(self) -> Optional[Dict]:
        """
        Stop recording, close pipe, and save metadata.
        
        Returns:
            Metadata dictionary if successful, None otherwise
        """
        if not self.is_active:
            return None
        
        with self.lock:
            self.is_active = False
            self.end_time = datetime.now()
        
        # Save the best snapshot before closing
        self._save_best_snapshot()
        
        # Close FFmpeg Pipe
        if self.process:
            try:
                # 1. Close stdin to signal EOF to FFmpeg
                if self.process.stdin:
                    self.process.stdin.close()
                
                # 2. Wait for process to finish and capture output (avoids deadlock)
                # communicate() reads stdout/stderr until EOF
                try:
                    stdout, stderr = self.process.communicate(timeout=10)
                    if self.process.returncode != 0:
                        logger.error(f"⚠️ FFmpeg exited with error code {self.process.returncode}")
                        if stderr:
                            logger.error(f"FFmpeg stderr: {stderr.decode('utf-8', errors='ignore')}")
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️ FFmpeg timed out for track {self.track_id}, killing...")
                    self.process.kill()
                    stdout, stderr = self.process.communicate() # Clean up pipes
                    
            except Exception as e:
                logger.error(f"⚠️ Error closing FFmpeg: {e}")
            finally:
                self.process = None
        
        # FIX #3b: Give OS time to release file locks (especially Windows)
        import time as time_module
        time_module.sleep(0.5)
        
        # Calculate duration
        duration = (self.end_time - self.start_time).total_seconds()
        
        # Save metadata
        metadata = {
            "event_id": f"track_{self.track_id}",
            "track_id": self.track_id,
            "class_name": self.class_name,
            "verification_status": "pending" if self.verification_mode else "unknown",
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_seconds": round(duration, 2),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "video_path": str(self.video_path),
            "snapshot_path": str(self.snapshot_path) if self.snapshot_path.exists() else None,
            "detected_classes": [self.class_name],
            "encoder": "ffmpeg/libx264",
            "created_at": datetime.now().isoformat()
        }
        
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✅ Stopped recording: track_{self.track_id} ({duration:.1f}s, {self.frame_count} frames)")
            return metadata
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            return None

    def cancel(self):
        """
        Stop recording and delete files.
        Used when the recording needs to be cancelled (e.g., false detection).
        """
        if not self.is_active:
            return
        
        with self.lock:
            self.is_active = False
        
        if self.process:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                pass
            self.process = None
        
        try:
            # Note: We don't delete the directory here - let VerificationManager handle it
            logger.info(f"🗑️ Cancelled recording: track_{self.track_id}")
        except Exception as e:
            logger.error(f"✗ Error cancelling recording: {e}")


class RecordingManager:
    """
    Manages multiple concurrent FFmpeg video recordings with verification support.
    Integrates with VerificationManager for proper file organization.
    """
    
    def __init__(
        self, 
        verification_manager = None,
        fps: int = 20,
        buffer_seconds: int = 5,   
        timeout_seconds: int = 5, 
        enabled: bool = True
    ):
        """
        Initialize recording manager.
        
        Args:
            verification_manager: VerificationManager instance for file organization
            fps: Frames per second
            buffer_seconds: Pre-event buffer duration
            timeout_seconds: Time to wait before stopping recording
            enabled: Enable/disable recording
        """
        self.verification_manager = verification_manager
        self.fps = fps
        self.disappear_timeout = timeout_seconds
        self.enabled = enabled
        
        # Check for FFmpeg
        if not _ffmpeg_available():
            logger.error("❌ CRITICAL: FFmpeg not found! Recording will be disabled.")
            self.enabled = False

        # Active recordings: {track_id: VideoRecorder}
        self.active_recordings: Dict[int, VideoRecorder] = {}
        # Track last seen time: {track_id: timestamp}
        self.last_seen: Dict[int, float] = {}
        # Recent metadata paths for tracks that just finished: {track_id: path}
        self.recent_metadata_paths: Dict[int, str] = {}
        
        # Ring Buffer
        self.buffer_maxlen = int(fps * buffer_seconds)
        self.frame_buffer = deque(maxlen=self.buffer_maxlen)
        
        self.lock = threading.Lock()
        
        if self.enabled:
            logger.info(f"🎬 RecordingManager: {buffer_seconds}s pre-buffer, {timeout_seconds}s timeout")
            if self.verification_manager:
                logger.info("   ✓ Verification mode: ENABLED")
    
    def update_metadata_on_disk(
        self, 
        track_id: int, 
        new_class_name: str, 
        ai_info: Optional[Dict] = None
    ) -> bool:
        """
        Update the metadata.json file on disk regardless of active recording status.
        
        Args:
            track_id: Track ID
            new_class_name: AI-verified class name
            ai_info: Complete AI information
            
        Returns:
            True if successful
        """
        path = None
        with self.lock:
            if track_id in self.active_recordings:
                path = self.active_recordings[track_id].metadata_path
            elif track_id in self.recent_metadata_paths:
                path = self.recent_metadata_paths[track_id]
        
        if not path:
            # Try to find it in verification manager's pending directory
            if self.verification_manager:
                pending_dir = self.verification_manager.get_pending_dir(track_id)
                if pending_dir:
                    path = pending_dir / "metadata.json"
        
        if not path or not Path(path).exists():
            logger.warning(f"Metadata file not found for track {track_id}")
            return False

        try:
            with self.lock:
                with open(path, 'r') as f:
                    metadata = json.load(f)
                
                metadata["class_name"] = new_class_name
                metadata["detected_classes"] = list(set(metadata.get("detected_classes", []) + [new_class_name]))
                
                if ai_info:
                    metadata["ai_info"] = ai_info
                    if ai_info.get("scientificName"):
                        metadata["scientific_name"] = ai_info["scientificName"]
                    if ai_info.get("commonName"):
                        metadata["common_name"] = ai_info["commonName"]

                with open(path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"📝 Updated metadata for track {track_id}: {new_class_name}")
                return True
        except Exception as e:
            logger.error(f"❌ Error updating metadata for track {track_id}: {e}")
            return False
    
    def update_tracks(self, frame: np.ndarray, detections: List[Dict]):
        """
        Update recordings based on current frame detections.
        
        Args:
            frame: Current video frame
            detections: List of detection dictionaries
        """
        if not self.enabled:
            return
        
        current_time = time.time()
        current_track_ids = set()
        
        with self.lock:
            # Append COPY of frame to buffer (important for FFmpeg stability)
            self.frame_buffer.append(frame.copy())

            for det in detections:
                track_id = det.get('track_id')
                if track_id is None:
                    continue
                
                current_track_ids.add(track_id)
                class_name = det.get('class_name') or det.get('class') or 'unknown'
                
                # Filter out 'person' if needed
                if class_name.lower() == "person":
                    continue

                self.last_seen[track_id] = current_time
                
                if track_id not in self.active_recordings:
                    self.start_recording(track_id, class_name, frame.shape[:2])
                else:
                    # Check if this is a stability frame (for better snapshot)
                    is_stability_frame = det.get('is_stable', False)
                    self.active_recordings[track_id].write_frame(frame, is_stability_frame)
            
            self.cleanup_disappeared_tracks(current_track_ids, current_time)
    
    def start_recording(
        self, 
        track_id: int, 
        class_name: str, 
        frame_shape: Tuple[int, int]
    ):
        """
        Start a new recording.
        
        Args:
            track_id: Track ID
            class_name: YOLO-detected class name
            frame_shape: (height, width) of frames
        """
        try:
            # Create pending directory via VerificationManager
            if self.verification_manager:
                track_dir = self.verification_manager.create_pending_recording(
                    track_id, class_name
                )
            else:
                # Fallback: create in default location
                track_dir = Config.EVENTS_DIR / f"track_{track_id}"
                track_dir.mkdir(parents=True, exist_ok=True)
            
            # Capture snapshot of buffer history
            pre_event_frames = list(self.frame_buffer)
            
            recorder = VideoRecorder(
                track_id=track_id,
                class_name=class_name,
                frame_shape=frame_shape,
                fps=self.fps,
                track_dir=track_dir,
                pre_buffer_frames=pre_event_frames,
                verification_mode=(self.verification_manager is not None)
            )
            self.active_recordings[track_id] = recorder
            
        except Exception as e:
            logger.error(f"❌ Error starting recording for track {track_id}: {e}")
    
    def stop_recording(self, track_id: int):
        """Stop a recording and save metadata."""
        if track_id in self.active_recordings:
            try:
                # Store metadata path for potential post-recording updates
                recorder = self.active_recordings[track_id]
                metadata_path = recorder.metadata_path
                self.recent_metadata_paths[track_id] = str(metadata_path)
                
                # Cleanup old paths to avoid memory leak (keep last 50)
                if len(self.recent_metadata_paths) > 50:
                    oldest_id = next(iter(self.recent_metadata_paths))
                    self.recent_metadata_paths.pop(oldest_id, None)

                recorder.stop()
                del self.active_recordings[track_id]
                if track_id in self.last_seen:
                    del self.last_seen[track_id]
                    
            except Exception as e:
                logger.error(f"❌ Error stopping recording for track {track_id}: {e}")
    
    def cancel_recording(self, track_id: int):
        """Cancel a recording and delete files via VerificationManager."""
        if track_id in self.active_recordings:
            try:
                self.active_recordings[track_id].cancel()
                del self.active_recordings[track_id]
                if track_id in self.last_seen:
                    del self.last_seen[track_id]
                
                # Remove from recent paths
                self.recent_metadata_paths.pop(track_id, None)
                
                # Delete via verification manager
                if self.verification_manager:
                    self.verification_manager.cancel_pending(track_id)
                    
            except Exception as e:
                logger.error(f"❌ Error cancelling recording for track {track_id}: {e}")

    def rename_recording(self, track_id: int, new_class_name: str):
        """
        Update the class name for a recording.
        
        Args:
            track_id: Track ID
            new_class_name: AI-verified class name
        """
        with self.lock:
            if track_id in self.active_recordings:
                logger.info(f"✏️ Renaming track {track_id} → {new_class_name}")
                self.active_recordings[track_id].update_class_name(new_class_name)
    
    def cleanup_disappeared_tracks(self, current_track_ids: set, current_time: float):
        """Stop recordings for tracks that have disappeared."""
        disappeared_tracks = []
        
        for track_id, recorder in self.active_recordings.items():
            if track_id not in current_track_ids:
                last_seen_time = self.last_seen.get(track_id, current_time)
                time_since_seen = current_time - last_seen_time
                
                if time_since_seen >= self.disappear_timeout:
                    disappeared_tracks.append(track_id)
        
        for track_id in disappeared_tracks:
            self.stop_recording(track_id)
    
    def get_stats(self) -> Dict:
        """Get current recording statistics."""
        with self.lock:
            stats = {
                "active_recordings": len(self.active_recordings),
                "track_ids": list(self.active_recordings.keys()),
                "buffer_size": len(self.frame_buffer),
                "type": "FFmpeg/H.264"
            }
            
            if self.verification_manager:
                vm_stats = self.verification_manager.get_stats()
                stats.update({
                    "pending": vm_stats.get('pending', 0),
                    "verified": vm_stats.get('verified', 0),
                    "rejected": vm_stats.get('rejected', 0)
                })
            
            return stats
    
    def stop_all(self):
        """Stop all active recordings."""
        with self.lock:
            track_ids = list(self.active_recordings.keys())
            for track_id in track_ids:
                self.stop_recording(track_id)
