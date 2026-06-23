import sqlite3
import os
from datetime import datetime
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DB_PATH = os.getenv("DB_PATH", "data/predictions.db")

def get_connection():
    """Returns a database connection (PostgreSQL or SQLite) depending on environment config."""
    if DATABASE_URL:
        import psycopg2
        return psycopg2.connect(DATABASE_URL)
    else:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        return sqlite3.connect(DB_PATH)

def init_db():
    """Initializes the database and creates the logs table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()
    
    if DATABASE_URL:
        # PostgreSQL Schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                timestamp VARCHAR(50),
                beds INTEGER,
                baths INTEGER,
                area DOUBLE PRECISION,
                luxury_score INTEGER,
                has_view INTEGER,
                has_maids_room INTEGER,
                is_freehold INTEGER,
                district VARCHAR(255),
                furnished VARCHAR(100),
                predicted_price DOUBLE PRECISION,
                client_host VARCHAR(100),
                latency_ms DOUBLE PRECISION DEFAULT 0.0
            )
        """)
    else:
        # SQLite Schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                beds INTEGER,
                baths INTEGER,
                area REAL,
                luxury_score INTEGER,
                has_view INTEGER,
                has_maids_room INTEGER,
                is_freehold INTEGER,
                district TEXT,
                furnished TEXT,
                predicted_price REAL,
                client_host TEXT,
                latency_ms REAL DEFAULT 0.0
            )
        """)
        
        # Ensure backward compatibility by adding latency_ms if it's missing in existing DB
        cursor.execute("PRAGMA table_info(prediction_logs)")
        columns = [col[1] for col in cursor.fetchall()]
        if columns and "latency_ms" not in columns:
            cursor.execute("ALTER TABLE prediction_logs ADD COLUMN latency_ms REAL DEFAULT 0.0")
        
    conn.commit()
    conn.close()

def log_prediction(beds: int, baths: int, area: float, luxury_score: int, 
                    has_view: int, has_maids_room: int, is_freehold: int, 
                    district: str, furnished: str, predicted_price: float, client_host: str, latency_ms: float = 0.0):
    """Logs a single prediction request and output to the database."""
    conn = get_connection()
    cursor = conn.cursor()
    
    placeholder = "%s" if DATABASE_URL else "?"
    query = f"""
        INSERT INTO prediction_logs (
            timestamp, beds, baths, area, luxury_score, has_view, has_maids_room, 
            is_freehold, district, furnished, predicted_price, client_host, latency_ms
        ) VALUES ({', '.join([placeholder]*13)})
    """
    
    cursor.execute(query, (
        datetime.utcnow().isoformat(),
        beds, baths, area, luxury_score, has_view, has_maids_room, 
        is_freehold, district, furnished, predicted_price, client_host, latency_ms
    ))
    conn.commit()
    conn.close()

