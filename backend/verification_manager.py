"""
Verification Manager - Handles two-phase commit for wildlife recordings.

Phase 1: Record to pending/ directory
Phase 2: After AI verification, move to verified/{species}/ or rejected/

This ensures videos are properly organized by verified species name.
"""

import os
import shutil
import json
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerificationManager")


class VerificationManager:
    """
    Manages the lifecycle of recordings from pending to verified/rejected.
    Implements two-phase commit pattern for reliable file organization.
    """
    
    def __init__(
        self,
        base_dir: Optional[str] = None,
        retention_days: int = 7,
        cleanup_interval: int = 3600
    ):
        """
        Initialize verification manager.
        
        Args:
            base_dir: Base directory for all recordings
            retention_days: How long to keep rejected recordings
            cleanup_interval: Seconds between cleanup runs
        """
        self.base_dir = Path(base_dir) if base_dir else Config.EVENTS_DIR

        self.pending_dir = self.base_dir / "pending"
        self.verified_dir = self.base_dir / "verified"
        self.rejected_dir = self.base_dir / "rejected"
        
        self.retention_days = retention_days
        self.cleanup_interval = cleanup_interval
        
        # Create directory structure
        self._initialize_directories()
        
        # Track pending operations
        self.pending_operations: Dict[int, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Start cleanup thread
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("✓ VerificationManager initialized")
        logger.info(f"  📂 Pending: {self.pending_dir}")
        logger.info(f"  ✅ Verified: {self.verified_dir}")
        logger.info(f"  ❌ Rejected: {self.rejected_dir}")
    
    def _initialize_directories(self):
        """Create the directory structure if it doesn't exist."""
        for directory in [self.pending_dir, self.verified_dir, self.rejected_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        logger.info("✓ Directory structure initialized")
    
    def create_pending_recording(
        self, 
        track_id: int, 
        yolo_class: str,
        timestamp: Optional[datetime] = None
    ) -> Path:
        """
        Create a pending recording directory.
        
        Args:
            track_id: Unique tracking ID
            yolo_class: YOLO's detected class name
            timestamp: Recording start time (default: now)
            
        Returns:
            Path to the pending track directory
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        track_dir = self.pending_dir / f"track_{track_id}"
        track_dir.mkdir(parents=True, exist_ok=True)
        
        # Store pending operation metadata
        with self.lock:
            self.pending_operations[track_id] = {
                'track_dir': str(track_dir),
                'yolo_class': yolo_class,
                'created_at': timestamp.isoformat(),
                'status': 'recording'
            }
        
        logger.info(f"📹 Created pending recording: track_{track_id}")
        return track_dir
    
    def get_pending_dir(self, track_id: int) -> Optional[Path]:
        """Get the pending directory path for a track."""
        track_dir = self.pending_dir / f"track_{track_id}"
        return track_dir if track_dir.exists() else None
    
    def verify_and_move(
        self,
        track_id: int,
        verified_class_name: str,
        ai_info: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """
        Move a pending recording to the verified directory after AI confirmation.
        
        Args:
            track_id: Track ID
            verified_class_name: AI-verified species name (e.g., "Red_Fox")
            ai_info: Complete AI information dictionary
            
        Returns:
            Path to final verified directory, or None if failed
        """
        source_dir = self.pending_dir / f"track_{track_id}"
        
        if not source_dir.exists():
            logger.error(f"❌ Cannot verify track {track_id}: Source directory not found")
            return None
        
        try:
            # Create species-specific directory
            species_dir = self.verified_dir / verified_class_name
            species_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped subdirectory
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            dest_dir = species_dir / f"{timestamp}_track_{track_id}"
            
            # Move directory atomically
            shutil.move(str(source_dir), str(dest_dir))
            
            # Update metadata.json with verification info
            metadata_path = dest_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata['verification_status'] = 'verified'
                metadata['verified_at'] = datetime.now().isoformat()
                metadata['verified_class_name'] = verified_class_name
                
                if ai_info:
                    metadata['ai_info'] = ai_info
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Remove from pending operations
            with self.lock:
                self.pending_operations.pop(track_id, None)
            
            logger.info(f"✅ Verified: track_{track_id} → {verified_class_name}")
            logger.info(f"   Moved to: {dest_dir}")
            return dest_dir
            
        except Exception as e:
            logger.error(f"❌ Error verifying track {track_id}: {e}")
            return None
    
    def reject_and_move(
        self,
        track_id: int,
        reason: str = "Not an animal"
    ) -> Optional[Path]:
        """
        Move a pending recording to the rejected directory.
        
        Args:
            track_id: Track ID
            reason: Reason for rejection
            
        Returns:
            Path to rejected directory, or None if failed
        """
        source_dir = self.pending_dir / f"track_{track_id}"
        
        if not source_dir.exists():
            logger.warning(f"⚠ Cannot reject track {track_id}: Source directory not found")
            return None
        
        try:
            dest_dir = self.rejected_dir / f"track_{track_id}"
            
            # Move directory
            shutil.move(str(source_dir), str(dest_dir))
            
            # Update metadata
            metadata_path = dest_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata['verification_status'] = 'rejected'
                metadata['rejected_at'] = datetime.now().isoformat()
                metadata['rejection_reason'] = reason
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Remove from pending operations
            with self.lock:
                self.pending_operations.pop(track_id, None)
            
            logger.info(f"❌ Rejected: track_{track_id} (Reason: {reason})")
            return dest_dir
            
        except Exception as e:
            logger.error(f"❌ Error rejecting track {track_id}: {e}")
            return None
    
    def mark_as_error(
        self,
        track_id: int,
        error_message: str
    ) -> bool:
        """
        Mark a pending recording as error (AI failed multiple times).
        Keeps it in pending/ for manual review.
        
        Args:
            track_id: Track ID
            error_message: Error description
            
        Returns:
            True if successful
        """
        source_dir = self.pending_dir / f"track_{track_id}"
        
        if not source_dir.exists():
            logger.warning(f"⚠ Cannot mark track {track_id} as error: Not found")
            return False
        
        try:
            # Update metadata to indicate error
            metadata_path = source_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata['verification_status'] = 'error'
                metadata['error_at'] = datetime.now().isoformat()
                metadata['error_message'] = error_message
                
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Update pending operations
            with self.lock:
                if track_id in self.pending_operations:
                    self.pending_operations[track_id]['status'] = 'error'
                    self.pending_operations[track_id]['error_message'] = error_message
            
            logger.warning(f"⚠ Marked as error: track_{track_id} ({error_message})")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error marking track {track_id}: {e}")
            return False
    
    def cancel_pending(self, track_id: int) -> bool:
        """
        Cancel a pending recording and delete its files.
        Used when the object disappears before AI analysis.
        
        Args:
            track_id: Track ID
            
        Returns:
            True if successful
        """
        source_dir = self.pending_dir / f"track_{track_id}"
        
        if not source_dir.exists():
            return False
        
        try:
            shutil.rmtree(source_dir)
            
            # Remove from pending operations
            with self.lock:
                self.pending_operations.pop(track_id, None)
            
            logger.info(f"🗑️  Cancelled and deleted: track_{track_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cancelling track {track_id}: {e}")
            return False
    
    def get_pending_tracks(self) -> list:
        """Get list of all pending track directories."""
        if not self.pending_dir.exists():
            return []
        
        pending = []
        for track_dir in self.pending_dir.glob("track_*"):
            if track_dir.is_dir():
                metadata_file = track_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        metadata['track_dir'] = str(track_dir)
                        pending.append(metadata)
                    except Exception as e:
                        logger.error(f"Error reading metadata for {track_dir}: {e}")
        
        return pending
    
    def get_verified_tracks(self, species: Optional[str] = None) -> list:
        """
        Get list of verified tracks, optionally filtered by species.
        
        Args:
            species: Filter by species name (e.g., "Red_Fox")
            
        Returns:
            List of verified track metadata
        """
        verified = []
        
        if species:
            # Search specific species directory
            species_dir = self.verified_dir / species
            if species_dir.exists():
                search_dirs = [species_dir]
            else:
                return []
        else:
            # Search all species directories
            search_dirs = [d for d in self.verified_dir.iterdir() if d.is_dir()]
        
        for species_dir in search_dirs:
            for track_dir in species_dir.glob("*_track_*"):
                if track_dir.is_dir():
                    metadata_file = track_dir / "metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            metadata['track_dir'] = str(track_dir)
                            metadata['species'] = species_dir.name
                            verified.append(metadata)
                        except Exception as e:
                            logger.error(f"Error reading metadata for {track_dir}: {e}")
        
        return verified
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about recordings."""
        stats = {
            'pending': 0,
            'verified': 0,
            'rejected': 0,
            'verified_by_species': {},
            'oldest_pending': None,
            'newest_verified': None
        }
        
        # Count pending
        if self.pending_dir.exists():
            stats['pending'] = len(list(self.pending_dir.glob("track_*")))
        
        # Count rejected
        if self.rejected_dir.exists():
            stats['rejected'] = len(list(self.rejected_dir.glob("track_*")))
        
        # Count verified by species
        if self.verified_dir.exists():
            for species_dir in self.verified_dir.iterdir():
                if species_dir.is_dir():
                    count = len(list(species_dir.glob("*_track_*")))
                    if count > 0:
                        stats['verified_by_species'][species_dir.name] = count
                        stats['verified'] += count
        
        return stats
    
    def _cleanup_loop(self):
        """Background thread to cleanup old rejected recordings."""
        logger.info("🧹 Cleanup thread started")
        
        while self.running:
            try:
                self._cleanup_old_rejected()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
            
            # Sleep for cleanup_interval
            time.sleep(self.cleanup_interval)
    
    def _cleanup_old_rejected(self):
        """Delete rejected recordings older than retention_days."""
        if not self.rejected_dir.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        deleted_count = 0
        
        for track_dir in self.rejected_dir.glob("track_*"):
            if not track_dir.is_dir():
                continue
            
            # Check creation time
            created_time = datetime.fromtimestamp(track_dir.stat().st_ctime)
            
            if created_time < cutoff_time:
                try:
                    shutil.rmtree(track_dir)
                    deleted_count += 1
                    logger.info(f"🗑️  Cleaned up old rejected: {track_dir.name}")
                except Exception as e:
                    logger.error(f"Error deleting {track_dir}: {e}")
        
        if deleted_count > 0:
            logger.info(f"🧹 Cleaned up {deleted_count} old rejected recordings")
    
    def stop(self):
        """Stop the verification manager and cleanup thread."""
        self.running = False
        if self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=2)
        logger.info("✓ VerificationManager stopped")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()


# Utility functions for integration

def get_verified_path(base_dir: str, track_id: int, species_name: str) -> Path:
    """
    Get the expected verified path for a track.
    
    Args:
        base_dir: Base events directory
        track_id: Track ID
        species_name: Verified species name
        
    Returns:
        Path to verified track directory
    """
    base = Path(base_dir) / "verified" / species_name
    
    # Find the directory (it will have a timestamp prefix)
    for track_dir in base.glob(f"*_track_{track_id}"):
        return track_dir
    
    return None


def get_all_species(base_dir: str) -> list:
    """
    Get list of all verified species.
    
    Args:
        base_dir: Base events directory
        
    Returns:
        List of species names
    """
    verified_dir = Path(base_dir) / "verified"
    
    if not verified_dir.exists():
        return []
    
    return [d.name for d in verified_dir.iterdir() if d.is_dir()]
