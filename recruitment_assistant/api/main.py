"""
招聘助手 - FastAPI 应用入口
============================
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from .routers.recruitment import router as recruitment_router

app = FastAPI(
    title="招聘助手 API",
    description="基于 LangGraph 的全流程招聘助手，支持简历筛选、面试生成、录用建议",
    version="1.0.0",
)

# CORS 配置（允许前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    """健康检查接口"""
    return {"status": "ok", "service": "recruitment-assistant"}


# 注册 API 路由（必须在静态文件挂载之前）
app.include_router(recruitment_router)

# 前端首页（直接返回 index.html）
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
def serve_frontend():
    """返回前端首页"""
    return FileResponse(os.path.join(frontend_dir, "index.html"))
