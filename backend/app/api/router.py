from fastapi import APIRouter

from app.api.routes import datasets, health, predict

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(datasets.router)
api_router.include_router(predict.router)

