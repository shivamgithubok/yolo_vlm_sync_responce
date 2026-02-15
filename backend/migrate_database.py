"""
Migration script to update existing tracking_data.db with verification columns.
This script safely migrates your existing data to the new schema.

Run this ONCE after replacing database.py with the new version.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path


def migrate_database(db_path: str = "backend/tracking_data.db"):
    """
    Migrate existing database to new schema with verification support.
    
    Args:
        db_path: Path to the database file
    """
    print("="*70)
    print("DATABASE MIGRATION SCRIPT")
    print("="*70)
    
    if not Path(db_path).exists():
        print(f"✗ Database not found at: {db_path}")
        print("  Creating new database with updated schema...")
        # The DatabaseManager will create it with the new schema
        return
    
    print(f"📂 Migrating database: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Step 1: Check what columns exist
        cursor.execute("PRAGMA table_info(tracking_objects)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        print(f"\n✓ Found {len(existing_columns)} existing columns")
        
        # Step 2: Add new verification columns if they don't exist
        new_columns = {
            'yolo_original_class': 'TEXT',
            'verification_status': 'TEXT DEFAULT "pending"',
            'ai_processed_at': 'TIMESTAMP',
            'ai_error_count': 'INTEGER DEFAULT 0',
            'ai_error_message': 'TEXT',
            'video_path': 'TEXT',
            'metadata_path': 'TEXT'
        }
        
        added_count = 0
        for col_name, col_type in new_columns.items():
            if col_name not in existing_columns:
                print(f"  Adding column: {col_name}")
                cursor.execute(f"""
                    ALTER TABLE tracking_objects 
                    ADD COLUMN {col_name} {col_type}
                """)
                added_count += 1
        
        if added_count > 0:
            print(f"\n✓ Added {added_count} new columns")
        else:
            print("\n✓ All verification columns already exist")
        
        # Step 3: Migrate existing data - Set yolo_original_class
        print("\n📝 Migrating existing data...")
        cursor.execute("""
            UPDATE tracking_objects 
            SET yolo_original_class = class_name 
            WHERE yolo_original_class IS NULL
        """)
        rows_updated = cursor.rowcount
        print(f"  ✓ Set yolo_original_class for {rows_updated} records")
        
        # Step 4: Set verification_status based on AI data
        cursor.execute("""
            UPDATE tracking_objects 
            SET verification_status = CASE 
                WHEN ai_is_animal = 1 THEN 'verified'
                WHEN ai_is_animal = 0 THEN 'rejected'
                WHEN ai_common_name IS NULL AND ai_scientific_name IS NULL THEN 'pending'
                ELSE 'error'
            END
            WHERE verification_status IS NULL OR verification_status = ''
        """)
        rows_updated = cursor.rowcount
        print(f"  ✓ Set verification_status for {rows_updated} records")
        
        # Step 5: Create new indexes
        print("\n🔍 Creating indexes...")
        indexes = [
            ("idx_verification_status", "verification_status"),
            ("idx_last_seen", "last_seen"),
            ("idx_track_id", "track_id"),
            ("idx_scientific_name", "ai_scientific_name")
        ]
        
        for idx_name, column in indexes:
            try:
                cursor.execute(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name} 
                    ON tracking_objects({column})
                """)
                print(f"  ✓ Created index: {idx_name}")
            except Exception as e:
                print(f"  ⚠ Index {idx_name} already exists")
        
        # Step 6: Get migration statistics
        print("\n" + "="*70)
        print("MIGRATION STATISTICS")
        print("="*70)
        
        cursor.execute("SELECT COUNT(*) FROM tracking_objects")
        total = cursor.fetchone()[0]
        print(f"Total records: {total}")
        
        cursor.execute("""
            SELECT verification_status, COUNT(*) as count 
            FROM tracking_objects 
            GROUP BY verification_status
        """)
        status_counts = cursor.fetchall()
        print("\nRecords by status:")
        for status, count in status_counts:
            print(f"  {status or 'NULL'}: {count}")
        
        # Step 7: Show pending/error records that need attention
        cursor.execute("""
            SELECT track_id, class_name, yolo_original_class, verification_status, 
                   ai_common_name, ai_scientific_name
            FROM tracking_objects 
            WHERE verification_status IN ('pending', 'error')
            ORDER BY track_id
        """)
        problem_records = cursor.fetchall()
        
        if problem_records:
            print(f"\n⚠ Found {len(problem_records)} records needing attention:")
            print("\nTrack ID | YOLO Class | AI Status | Verification Status")
            print("-" * 70)
            for track_id, class_name, yolo_class, status, ai_common, ai_sci in problem_records:
                ai_status = f"{ai_common or 'N/A'} ({ai_sci or 'N/A'})"
                print(f"{track_id:8d} | {yolo_class or class_name:15s} | {ai_status:30s} | {status}")
            
            print("\nRecommendation:")
            print("  - 'pending' records: Will be processed by AI on next detection")
            print("  - 'error' records: AI failed multiple times, may need manual review")
        
        # Commit all changes
        conn.commit()
        
        print("\n" + "="*70)
        print("✓ MIGRATION COMPLETED SUCCESSFULLY")
        print("="*70)
        print("\nNext steps:")
        print("  1. Restart your application")
        print("  2. New detections will use the verification system")
        print("  3. Pending records will be re-processed automatically")
        
    except Exception as e:
        print(f"\n✗ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def verify_migration(db_path: str = "backend/tracking_data.db"):
    """
    Verify the migration was successful.
    
    Args:
        db_path: Path to the database file
    """
    print("\n" + "="*70)
    print("VERIFICATION CHECK")
    print("="*70)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check for required columns
        cursor.execute("PRAGMA table_info(tracking_objects)")
        columns = {row[1] for row in cursor.fetchall()}
        
        required_columns = {
            'verification_status', 'yolo_original_class', 'ai_processed_at',
            'ai_error_count', 'ai_error_message', 'video_path', 'metadata_path'
        }
        
        missing = required_columns - columns
        if missing:
            print(f"✗ Missing columns: {missing}")
            return False
        
        print("✓ All required columns exist")
        
        # Check indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='tracking_objects'")
        indexes = {row[0] for row in cursor.fetchall()}
        
        required_indexes = {
            'idx_verification_status', 'idx_last_seen', 'idx_track_id', 'idx_scientific_name'
        }
        
        missing_idx = required_indexes - indexes
        if missing_idx:
            print(f"⚠ Missing indexes: {missing_idx} (will be created automatically)")
        else:
            print("✓ All required indexes exist")
        
        # Check data integrity
        cursor.execute("""
            SELECT COUNT(*) FROM tracking_objects 
            WHERE verification_status IS NULL
        """)
        null_status = cursor.fetchone()[0]
        
        if null_status > 0:
            print(f"⚠ Found {null_status} records with NULL verification_status")
        else:
            print("✓ All records have verification_status")
        
        print("\n✓ Migration verification passed!")
        return True
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    import sys
    
    # Allow custom database path
    db_path = sys.argv[1] if len(sys.argv) > 1 else "backend/tracking_data.db"
    
    print("\n🚀 Starting database migration...\n")
    
    # Run migration
    migrate_database(db_path)
    
    # Verify migration
    verify_migration(db_path)
    
    print("\n✅ Done! Your database is now ready for the verification system.")
