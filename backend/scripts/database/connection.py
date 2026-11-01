import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# Project paths
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT_ROOT = BACKEND_ROOT.parent

load_dotenv(BACKEND_ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL")
# Fail fast when the database is down; the default libpq wait is about two minutes.
CONNECT_TIMEOUT = int(os.getenv("DB_CONNECT_TIMEOUT", "5"))


def get_connection():
    if not DATABASE_URL:
        return None
    try:
        return psycopg.connect(DATABASE_URL, connect_timeout=CONNECT_TIMEOUT)
    except Exception as e:
        print(f"Connection error: {e}")
        return None
