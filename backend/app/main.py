import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
app = FastAPI(
    title="AI Nutrition Coach API",
    version="0.1.0",
    description="RAG-powered nutrition assistant backend built with FastAPI.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:8501,http://localhost:3000").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root() -> dict:
    return {"message": "AI Nutrition Coach API", "docs_url": "/docs"}
