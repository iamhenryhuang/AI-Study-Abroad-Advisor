import os

import psycopg

import settings  # noqa: F401  (loads backend/.env)

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
