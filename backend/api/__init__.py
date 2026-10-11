"""API 路由包。"""
from .routes_algorithm import router as algorithm_router
from .routes_pipeline import router as pipeline_router
from .routes_query import router as query_router

__all__ = ["pipeline_router", "algorithm_router", "query_router"]
