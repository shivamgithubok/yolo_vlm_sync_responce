"""
Database module for tracking object persistence using SQLite.
Enhanced with verification status tracking and better data integrity.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import threading
import os


class DatabaseManager:
    """Manages SQLite database for tracking objects with verification support."""
    
    def __init__(self, db_path: str = "tracking_data.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self.lock = threading.Lock()
        self._initialize_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _initialize_database(self):
        """Create database tables if they don't exist."""
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Create tracking_objects table with verification columns
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tracking_objects (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        track_id INTEGER NOT NULL,
                        class_name TEXT NOT NULL,
                        yolo_original_class TEXT,
                        verification_status TEXT DEFAULT 'pending',
                        first_seen TIMESTAMP NOT NULL,
                        last_seen TIMESTAMP NOT NULL,
                        ai_processed_at TIMESTAMP,
                        ai_error_count INTEGER DEFAULT 0,
                        ai_error_message TEXT,
                        video_path TEXT,
                        metadata_path TEXT,
                        ai_is_animal BOOLEAN,
                        ai_is_person BOOLEAN,
                        ai_common_name TEXT,
                        ai_scientific_name TEXT,
                        ai_description TEXT,
                        ai_safety_info TEXT,
                        ai_conservation_status TEXT,
                        ai_is_dangerous BOOLEAN,
                        ai_diet TEXT,
                        ai_lifespan TEXT,
                        ai_height_cm TEXT,
                        ai_weight_kg TEXT,
                        ai_color TEXT,
                        ai_predators TEXT,
                        ai_average_speed_kmh TEXT,
                        ai_countries_found TEXT,
                        ai_family TEXT,
                        ai_gestation_period_days TEXT,
                        ai_top_speed_kmh TEXT,
                        ai_social_structure TEXT,
                        ai_offspring_per_birth TEXT,
                        frame_snapshot TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for faster queries
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_track_id 
                    ON tracking_objects(track_id)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_verification_status 
                    ON tracking_objects(verification_status)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_scientific_name 
                    ON tracking_objects(ai_scientific_name)
                """)
                
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_last_seen 
                    ON tracking_objects(last_seen)
                """)
                
                # Migration: Add new columns to existing databases
                cursor.execute("PRAGMA table_info(tracking_objects)")
                existing_columns = {row[1] for row in cursor.fetchall()}
                
                # Define all new columns (including verification columns)
                new_columns = {
                    'yolo_original_class': 'TEXT',
                    'verification_status': 'TEXT DEFAULT "pending"',
                    'ai_processed_at': 'TIMESTAMP',
                    'ai_error_count': 'INTEGER DEFAULT 0',
                    'ai_error_message': 'TEXT',
                    'video_path': 'TEXT',
                    'metadata_path': 'TEXT',
                    'ai_is_animal': 'BOOLEAN',
                    'ai_is_person': 'BOOLEAN',
                    'ai_common_name': 'TEXT',
                    'ai_scientific_name': 'TEXT',
                    'ai_description': 'TEXT',
                    'ai_safety_info': 'TEXT',
                    'ai_conservation_status': 'TEXT',
                    'ai_is_dangerous': 'BOOLEAN',
                    'ai_diet': 'TEXT',
                    'ai_lifespan': 'TEXT',
                    'ai_height_cm': 'TEXT',
                    'ai_weight_kg': 'TEXT',
                    'ai_color': 'TEXT',
                    'ai_predators': 'TEXT',
                    'ai_average_speed_kmh': 'TEXT',
                    'ai_countries_found': 'TEXT',
                    'ai_family': 'TEXT',
                    'ai_gestation_period_days': 'TEXT',
                    'ai_top_speed_kmh': 'TEXT',
                    'ai_social_structure': 'TEXT',
                    'ai_offspring_per_birth': 'TEXT'
                }
                
                for col_name, col_type in new_columns.items():
                    if col_name not in existing_columns:
                        cursor.execute(f"""
                            ALTER TABLE tracking_objects 
                            ADD COLUMN {col_name} {col_type}
                        """)
                        print(f"✓ Added column: {col_name}")
                
                # Migrate existing data: Set yolo_original_class if NULL
                cursor.execute("""
                    UPDATE tracking_objects 
                    SET yolo_original_class = class_name 
                    WHERE yolo_original_class IS NULL
                """)
                
                # Set verification_status based on AI data presence
                cursor.execute("""
                    UPDATE tracking_objects 
                    SET verification_status = CASE 
                        WHEN ai_is_animal = 1 THEN 'verified'
                        WHEN ai_is_animal = 0 THEN 'rejected'
                        WHEN ai_error_count > 0 THEN 'error'
                        ELSE 'pending'
                    END
                    WHERE verification_status IS NULL OR verification_status = ''
                """)
                
                conn.commit()
                print("✓ Database initialized successfully")
            except Exception as e:
                print(f"✗ Error initializing database: {e}")
                raise
            finally:
                conn.close()
    
    def create_tracking_object(
        self, 
        track_id: int, 
        class_name: str, 
        first_seen: datetime,
        ai_info: Optional[Dict] = None,
        frame_snapshot: Optional[str] = None,
        video_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ) -> Optional[int]:
        """
        Create a new tracking object entry.
        
        Args:
            track_id: Unique tracking ID
            class_name: Detected class name (from YOLO)
            first_seen: Timestamp when first detected
            ai_info: Wildlife information dictionary (optional, usually None initially)
            frame_snapshot: Base64 encoded frame snapshot
            video_path: Path to video file
            metadata_path: Path to metadata.json file
            
        Returns:
            Database row ID if successful, None otherwise
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Extract individual AI fields if provided
                ai_is_animal = ai_info.get('is_animal') if ai_info else None
                ai_is_person = ai_info.get('is_person') if ai_info else None
                ai_common_name = ai_info.get('commonName') if ai_info else None
                ai_scientific_name = ai_info.get('scientificName') if ai_info else None
                ai_description = ai_info.get('description') if ai_info else None
                ai_safety_info = ai_info.get('safetyInfo') if ai_info else None
                ai_conservation_status = ai_info.get('conservationStatus') if ai_info else None
                ai_is_dangerous = ai_info.get('isDangerous') if ai_info else None
                ai_diet = ai_info.get('diet') if ai_info else None
                ai_lifespan = ai_info.get('lifespan') if ai_info else None
                ai_height_cm = ai_info.get('height_cm') if ai_info else None
                ai_weight_kg = ai_info.get('weight_kg') if ai_info else None
                ai_color = ai_info.get('color') if ai_info else None
                ai_predators = ai_info.get('predators') if ai_info else None
                ai_average_speed_kmh = ai_info.get('average_speed_kmh') if ai_info else None
                ai_countries_found = ai_info.get('countries_found') if ai_info else None
                ai_family = ai_info.get('family') if ai_info else None
                ai_gestation_period_days = ai_info.get('gestation_period_days') if ai_info else None
                ai_top_speed_kmh = ai_info.get('top_speed_kmh') if ai_info else None
                ai_social_structure = ai_info.get('social_structure') if ai_info else None
                ai_offspring_per_birth = ai_info.get('offspring_per_birth') if ai_info else None
                
                # Determine initial verification status
                if ai_info:
                    if ai_is_animal:
                        verification_status = 'verified'
                    elif ai_is_person:
                        verification_status = 'rejected'
                    else:
                        verification_status = 'rejected'
                else:
                    verification_status = 'pending'
                
                cursor.execute("""
                    INSERT INTO tracking_objects (
                        track_id, class_name, yolo_original_class, verification_status,
                        first_seen, last_seen, 
                        video_path, metadata_path,
                        ai_is_animal, ai_is_person,
                        ai_common_name, ai_scientific_name, ai_description,
                        ai_safety_info, ai_conservation_status, ai_is_dangerous,
                        ai_diet, ai_lifespan, ai_height_cm, ai_weight_kg,
                        ai_color, ai_predators, ai_average_speed_kmh,
                        ai_countries_found, ai_family, ai_gestation_period_days,
                        ai_top_speed_kmh, ai_social_structure, ai_offspring_per_birth,
                        frame_snapshot
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    track_id, class_name, class_name, verification_status,
                    first_seen, first_seen,
                    video_path, metadata_path,
                    ai_is_animal, ai_is_person,
                    ai_common_name, ai_scientific_name, ai_description,
                    ai_safety_info, ai_conservation_status, ai_is_dangerous,
                    ai_diet, ai_lifespan, ai_height_cm, ai_weight_kg,
                    ai_color, ai_predators, ai_average_speed_kmh,
                    ai_countries_found, ai_family, ai_gestation_period_days,
                    ai_top_speed_kmh, ai_social_structure, ai_offspring_per_birth,
                    frame_snapshot
                ))
                
                conn.commit()
                return cursor.lastrowid
            except sqlite3.IntegrityError as e:
                print(f"✗ Track {track_id} already exists: {e}")
                return None
            except Exception as e:
                print(f"✗ Error creating tracking object: {e}")
                return None
            finally:
                conn.close()
    
    def update_ai_info(self, track_id: int, ai_info: Dict[str, Any]) -> bool:
        """
        Update AI information for a tracking object.
        
        Args:
            track_id: Tracking ID
            ai_info: Complete AI information dictionary
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Extract all AI fields
                ai_is_animal = ai_info.get('is_animal')
                ai_is_person = ai_info.get('is_person')
                
                # Determine verification status
                if ai_is_animal:
                    verification_status = 'verified'
                elif ai_is_person:
                    verification_status = 'rejected'
                else:
                    verification_status = 'rejected'
                
                cursor.execute("""
                    UPDATE tracking_objects SET
                        verification_status = ?,
                        ai_processed_at = ?,
                        ai_error_count = 0,
                        ai_error_message = NULL,
                        ai_is_animal = ?,
                        ai_is_person = ?,
                        ai_common_name = ?,
                        ai_scientific_name = ?,
                        ai_description = ?,
                        ai_safety_info = ?,
                        ai_conservation_status = ?,
                        ai_is_dangerous = ?,
                        ai_diet = ?,
                        ai_lifespan = ?,
                        ai_height_cm = ?,
                        ai_weight_kg = ?,
                        ai_color = ?,
                        ai_predators = ?,
                        ai_average_speed_kmh = ?,
                        ai_countries_found = ?,
                        ai_family = ?,
                        ai_gestation_period_days = ?,
                        ai_top_speed_kmh = ?,
                        ai_social_structure = ?,
                        ai_offspring_per_birth = ?
                    WHERE track_id = ?
                """, (
                    verification_status,
                    datetime.now(),
                    ai_is_animal,
                    ai_is_person,
                    ai_info.get('commonName'),
                    ai_info.get('scientificName'),
                    ai_info.get('description'),
                    ai_info.get('safetyInfo'),
                    ai_info.get('conservationStatus'),
                    ai_info.get('isDangerous'),
                    ai_info.get('diet'),
                    ai_info.get('lifespan'),
                    ai_info.get('height_cm'),
                    ai_info.get('weight_kg'),
                    ai_info.get('color'),
                    ai_info.get('predators'),
                    ai_info.get('average_speed_kmh'),
                    ai_info.get('countries_found'),
                    ai_info.get('family'),
                    ai_info.get('gestation_period_days'),
                    ai_info.get('top_speed_kmh'),
                    ai_info.get('social_structure'),
                    ai_info.get('offspring_per_birth'),
                    track_id
                ))
                
                conn.commit()
                success = cursor.rowcount > 0
                return success
            except Exception as e:
                print(f"✗ Error updating AI info for track {track_id}: {e}")
                return False
            finally:
                conn.close()
    
    def update_class_name(self, track_id: int, new_class_name: str) -> bool:
        """
        Update the class name for a tracking object.
        This is called when AI refines the YOLO detection.
        
        Args:
            track_id: Tracking ID
            new_class_name: New class name (from AI)
            
        Returns:
            True if successful
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE tracking_objects 
                    SET class_name = ?
                    WHERE track_id = ?
                """, (new_class_name, track_id))
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                print(f"✗ Error updating class name: {e}")
                return False
            finally:
                conn.close()
    
    def update_paths(self, track_id: int, video_path: Optional[str] = None, 
                     metadata_path: Optional[str] = None) -> bool:
        """
        Update file paths for a tracking object.
        
        Args:
            track_id: Tracking ID
            video_path: Path to video file
            metadata_path: Path to metadata.json file
            
        Returns:
            True if successful
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                updates = []
                params = []
                
                if video_path is not None:
                    updates.append("video_path = ?")
                    params.append(video_path)
                
                if metadata_path is not None:
                    updates.append("metadata_path = ?")
                    params.append(metadata_path)
                
                if not updates:
                    return False
                
                params.append(track_id)
                query = f"UPDATE tracking_objects SET {', '.join(updates)} WHERE track_id = ?"
                
                cursor.execute(query, params)
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                print(f"✗ Error updating paths: {e}")
                return False
            finally:
                conn.close()
    
    def increment_error_count(self, track_id: int, error_message: str) -> bool:
        """
        Increment error count for failed AI processing.
        
        Args:
            track_id: Tracking ID
            error_message: Error message to store
            
        Returns:
            True if successful
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE tracking_objects 
                    SET ai_error_count = ai_error_count + 1,
                        ai_error_message = ?,
                        ai_processed_at = ?
                    WHERE track_id = ?
                """, (error_message, datetime.now(), track_id))
                
                # If error count >= 3, mark as 'error' status
                cursor.execute("""
                    UPDATE tracking_objects 
                    SET verification_status = 'error'
                    WHERE track_id = ? AND ai_error_count >= 3
                """, (track_id,))
                
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                print(f"✗ Error incrementing error count: {e}")
                return False
            finally:
                conn.close()
    
    def update_last_seen(self, track_id: int, timestamp: datetime) -> bool:
        """Update the last seen timestamp for a tracking object."""
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE tracking_objects 
                    SET last_seen = ?
                    WHERE track_id = ?
                """, (timestamp, track_id))
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                print(f"✗ Error updating last seen: {e}")
                return False
            finally:
                conn.close()
    
    def delete_track(self, track_id: int) -> bool:
        """Delete a tracking object from the database."""
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tracking_objects WHERE track_id = ?", (track_id,))
                conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                print(f"✗ Error deleting track: {e}")
                return False
            finally:
                conn.close()
    
    def get_tracking_object(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Get a tracking object by track ID."""
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM tracking_objects WHERE track_id = ?", (track_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return self._row_to_dict(row)
            except Exception as e:
                print(f"✗ Error getting tracking object: {e}")
                return None
            finally:
                conn.close()
    
    def get_pending_tracks(self, max_errors: int = 3) -> List[Dict[str, Any]]:
        """
        Get all tracks pending AI verification.
        
        Args:
            max_errors: Maximum error count to include
            
        Returns:
            List of pending track dictionaries
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tracking_objects 
                    WHERE verification_status = 'pending' 
                    AND ai_error_count < ?
                    ORDER BY first_seen ASC
                """, (max_errors,))
                
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
            except Exception as e:
                print(f"✗ Error getting pending tracks: {e}")
                return []
            finally:
                conn.close()
    
    def get_verified_tracks(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all verified animal tracks."""
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                query = """
                    SELECT * FROM tracking_objects 
                    WHERE verification_status = 'verified'
                    ORDER BY last_seen DESC
                """
                if limit:
                    query += f" LIMIT {limit}"
                
                cursor.execute(query)
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
            except Exception as e:
                print(f"✗ Error getting verified tracks: {e}")
                return []
            finally:
                conn.close()
    
    def get_tracks_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get tracks by verification status."""
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM tracking_objects 
                    WHERE verification_status = ?
                    ORDER BY last_seen DESC
                """, (status,))
                
                rows = cursor.fetchall()
                return [self._row_to_dict(row) for row in rows]
            except Exception as e:
                print(f"✗ Error getting tracks by status: {e}")
                return []
            finally:
                conn.close()
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a database row to a dictionary with proper AI info structure."""
        data = dict(row)
        
        # Build ai_info from individual columns
        if data.get('ai_is_animal') is not None or data.get('ai_common_name'):
            data['ai_info'] = {
                'is_animal': data.get('ai_is_animal'),
                'is_person': data.get('ai_is_person'),
                'detected_class': data.get('yolo_original_class'),
                'commonName': data.get('ai_common_name'),
                'scientificName': data.get('ai_scientific_name'),
                'description': data.get('ai_description'),
                'safetyInfo': data.get('ai_safety_info'),
                'conservationStatus': data.get('ai_conservation_status'),
                'isDangerous': data.get('ai_is_dangerous'),
                'diet': data.get('ai_diet'),
                'lifespan': data.get('ai_lifespan'),
                'height_cm': data.get('ai_height_cm'),
                'weight_kg': data.get('ai_weight_kg'),
                'color': data.get('ai_color'),
                'predators': data.get('ai_predators'),
                'average_speed_kmh': data.get('ai_average_speed_kmh'),
                'countries_found': data.get('ai_countries_found'),
                'family': data.get('ai_family'),
                'gestation_period_days': data.get('ai_gestation_period_days'),
                'top_speed_kmh': data.get('ai_top_speed_kmh'),
                'social_structure': data.get('ai_social_structure'),
                'offspring_per_birth': data.get('ai_offspring_per_birth')
            }
        else:
            data['ai_info'] = None
        
        # Remove individual AI column fields from output
        for key in list(data.keys()):
            if key.startswith('ai_') and key != 'ai_info':
                data.pop(key, None)
        
        return data
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                
                # Total tracks
                cursor.execute("SELECT COUNT(*) FROM tracking_objects")
                total = cursor.fetchone()[0]
                
                # Tracks by status
                cursor.execute("""
                    SELECT verification_status, COUNT(*) as count 
                    FROM tracking_objects 
                    GROUP BY verification_status
                """)
                by_status = {row[0]: row[1] for row in cursor.fetchall()}
                
                # Tracks by class (verified only)
                cursor.execute("""
                    SELECT class_name, COUNT(*) as count 
                    FROM tracking_objects 
                    WHERE verification_status = 'verified'
                    GROUP BY class_name
                """)
                by_class = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    "total_tracks": total,
                    "tracks_by_status": by_status,
                    "verified_tracks_by_class": by_class
                }
            except Exception as e:
                print(f"✗ Error getting stats: {e}")
                return {}
            finally:
                conn.close()
    
    def get_recent_animal_history(self, limit: int = 2) -> List[Dict[str, Any]]:
        """Get the most recent confirmed animal detections."""
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT class_name, ai_common_name, ai_scientific_name
                    FROM tracking_objects 
                    WHERE verification_status = 'verified'
                    AND ai_is_animal = 1
                    ORDER BY last_seen DESC 
                    LIMIT ?
                """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        "class_name": row['class_name'],
                        "common_name": row['ai_common_name'],
                        "scientific_name": row['ai_scientific_name']
                    })
                
                return results
            except Exception as e:
                print(f"✗ Error getting recent animal history: {e}")
                return []
            finally:
                conn.close()
    
    def get_unique_species_history(self, minutes: int = 10) -> List[Dict[str, Any]]:
        """
        Get all verified animal detections from the last X minutes.
        
        Args:
            minutes: Time window in minutes
            
        Returns:
            List of all verified animal sightings in the time window
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT class_name, ai_scientific_name, ai_common_name, 
                           ai_description, ai_safety_info,
                           ai_conservation_status, ai_is_dangerous, ai_is_animal,
                           frame_snapshot, last_seen, track_id,
                           ai_diet, ai_lifespan, ai_height_cm, ai_weight_kg,
                           ai_color, ai_predators, ai_average_speed_kmh,
                           ai_countries_found, ai_family, ai_gestation_period_days,
                           ai_top_speed_kmh, ai_social_structure, ai_offspring_per_birth
                    FROM tracking_objects 
                    WHERE verification_status = 'verified'
                    AND ai_is_animal = 1
                    AND ai_scientific_name IS NOT NULL
                    AND last_seen >= datetime('now', '-' || ? || ' minutes')
                    ORDER BY last_seen DESC
                """, (minutes,))
                
                results = []
                
                for row in cursor.fetchall():
                    results.append({
                        "track_id": row['track_id'],
                        "class_name": row['class_name'],
                        "ai_info": {
                            'is_animal': row['ai_is_animal'],
                            'is_person': False,
                            'detected_class': row['class_name'],
                            'commonName': row['ai_common_name'],
                            'scientificName': row['ai_scientific_name'],
                            'description': row['ai_description'],
                            'safetyInfo': row['ai_safety_info'],
                            'conservationStatus': row['ai_conservation_status'],
                            'isDangerous': row['ai_is_dangerous'],
                            'diet': row['ai_diet'],
                            'lifespan': row['ai_lifespan'],
                            'height_cm': row['ai_height_cm'],
                            'weight_kg': row['ai_weight_kg'],
                            'color': row['ai_color'],
                            'predators': row['ai_predators'],
                            'average_speed_kmh': row['ai_average_speed_kmh'],
                            'countries_found': row['ai_countries_found'],
                            'family': row['ai_family'],
                            'gestation_period_days': row['ai_gestation_period_days'],
                            'top_speed_kmh': row['ai_top_speed_kmh'],
                            'social_structure': row['ai_social_structure'],
                            'offspring_per_birth': row['ai_offspring_per_birth']
                        },
                        "frame_snapshot": row['frame_snapshot'],
                        "last_seen": row['last_seen']
                    })
                
                return results
            except Exception as e:
                print(f"✗ Error getting species history: {e}")
                return []
            finally:
                conn.close()

    def get_recent_verified_track(self, class_name: str, minutes: int = 5) -> Optional[Dict[str, Any]]:
        """
        Get the most recent verified track for a specific class within the last N minutes.
        
        Args:
            class_name: The YOLO class name to search for
            minutes: Time window in minutes
            
        Returns:
            Dictionary with track info if found, else None
        """
        with self.lock:
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                # Find the most recent verified track of this class
                # that was seen within the last 'minutes'
                cursor.execute("""
                    SELECT track_id, class_name, verification_status, last_seen, 
                           ai_common_name, ai_scientific_name, frame_snapshot
                    FROM tracking_objects 
                    WHERE (class_name = ? OR yolo_original_class = ?)
                    AND verification_status = 'verified'
                    AND last_seen >= datetime('now', '-' || ? || ' minutes')
                    ORDER BY last_seen DESC
                    LIMIT 1
                """, (class_name, class_name, minutes))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
            except Exception as e:
                print(f"✗ Error getting recent verified track: {e}")
                return None
            finally:
                conn.close()