import psycopg2
from psycopg2.extras import execute_values
import json
import logging
from typing import List, Dict
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

logger = logging.getLogger(__name__)


def get_db_connection():
    """Create PostgreSQL connection."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )


def init_database():
    """Initialize database schema."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create inference results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inference_results (
            id SERIAL PRIMARY KEY,
            video_id VARCHAR(255) NOT NULL,
            frame_number INTEGER NOT NULL,
            timestamp FLOAT,
            motion_score FLOAT,
            predictions JSONB,
            embedding FLOAT[],
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(video_id, frame_number)
        );
        
        CREATE INDEX IF NOT EXISTS idx_video_id ON inference_results(video_id);
        CREATE INDEX IF NOT EXISTS idx_created_at ON inference_results(created_at);
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    logger.info("Database initialized")


def save_results(messages: List[Dict], predictions: List[Dict]):
    """Save inference results to PostgreSQL."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Prepare data
    data = []
    for msg, pred in zip(messages, predictions):
        embedding = pred.get("embedding", None)
        
        data.append((
            msg["video_id"],
            msg["frame_number"],
            msg["timestamp"],
            msg["motion_score"],
            json.dumps(pred),
            embedding,
            pred.get("confidence", None)
        ))
    
    # Insert data
    execute_values(
        cursor,
        """
        INSERT INTO inference_results 
        (video_id, frame_number, timestamp, motion_score, predictions, embedding, confidence)
        VALUES %s
        ON CONFLICT (video_id, frame_number) DO UPDATE SET
            predictions = EXCLUDED.predictions,
            embedding = EXCLUDED.embedding,
            confidence = EXCLUDED.confidence
        """,
        data
    )
    
    conn.commit()
    cursor.close()
    conn.close()


def get_stats():
    """Get processing statistics from database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_frames,
            COUNT(DISTINCT video_id) as total_videos,
            AVG(confidence) as avg_confidence,
            MIN(created_at) as first_processed,
            MAX(created_at) as last_processed
        FROM inference_results
    """)
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return {
        "total_frames": result[0],
        "total_videos": result[1],
        "avg_confidence": float(result[2]) if result[2] else None,
        "first_processed": result[3].isoformat() if result[3] else None,
        "last_processed": result[4].isoformat() if result[4] else None
    }
