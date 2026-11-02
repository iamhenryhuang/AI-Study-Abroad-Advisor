"""Project paths and environment loading, shared by every backend module.

Importing this module loads backend/.env. Modules that read environment variables at import time
import it first, so the result does not depend on the working directory or on the import order.
"""
from pathlib import Path

from dotenv import load_dotenv

BACKEND_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_ROOT.parent

load_dotenv(BACKEND_ROOT / ".env")
