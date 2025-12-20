from typing import AsyncGenerator, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import DateTime, String, Integer, select, delete
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from datetime import datetime, timezone
from typing import Any
import json

app = FastAPI(title="Search_for_anomalies_in_data_FastAPI_Service")


# Создание асинхронного движка и сессии SQLAlchemy 

# BASE_DIR = Path(__file__).parent
# DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR}/test.db"
# engine: AsyncEngine = create_async_engine(
#     DATABASE_URL,
#     echo=True,
#     future=True
# )

# AsyncSessionLocal = async_sessionmaker(
#     engine,
#     class_=AsyncSession,
#     expire_on_commit=False,
#     autocommit=False,
#     autoflush=False
# )

# sqlalchemy модель

# class Base(DeclarativeBase):
#     """
#     Базовый класс для всех моделей SQLAlchemy
#     """
#     pass

# class Log_Requests(Base):
#     """
#     Реализация запросов /stats и /history подразумевает наличие модели (SQLAlchemy) для 
#     хранения логов
#     """

#     __tablename__ = 'Log_Requests'

#     id: Mapped[int] = mapped_column(
#         Integer,
#         primary_key=True, 
#         autoincrement=True
#         )

#     ts: Mapped[str] = mapped_column(
#         DateTime(timezone=True),
#         nullable=False
#         )


# Надо будет ещё сделать Pydantic валидацию 


class ForwardIn(BaseModel):
    features: dict[str, Any] = Field(..., description="One transaction features")

# модель предсказания
# def prediction(features: dict[str, Any]) -> dict[str, Any]:
    """
    Пока оставляю так, реальнгую модель подключу позже
    """

@app.post("/forward")
async def forward(request: Request, image: Optional[UploadFile] = None):
    """
    - 400 bad request при неверном формате инпута
    - 403 когда модель не смогла обработать данные (предобработка сломалась)
    - Если всё успешно то возвращаю JSON и 200
    """

    ct = (request.headers.get("content-type") or "").lower()

    # Если всё ок
    if "application/json" in ct:
        raw = await request.body()

        # парсинг
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return JSONResponse(status_code=400, content="bad request")

        try:
            inp = ForwardIn(**payload)
        except Exception:
            return JSONResponse(status_code=400, content="bad request")

        # Вызов модели
        try:
            result = prediction(inp.features)  # модель надо будет подключиь из пикла
        except Exception:
            return JSONResponse(status_code=403, content="модель не смогла обработать данные")

        return JSONResponse(status_code=200, content=result)

    # у нас имаджес в проекте нет, так что просто предусмотрм это как 400
    if "multipart/form-data" in ct:
        if image is None:
            return JSONResponse(status_code=400, content="bad request")
        return JSONResponse(status_code=400, content="bad request")

    return JSONResponse(status_code=400, content="bad request")