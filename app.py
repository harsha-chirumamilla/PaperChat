"""
Hugging Face Space - FastAPI Application for RAG Pipeline
This provides a REST API for the paper chat RAG system
"""

import os
import uvicorn
from src.api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
