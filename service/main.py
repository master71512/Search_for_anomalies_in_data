# пример 1

# from fastapi import FastAPI, HTTPException
# import uvicorn
# from pydantic import BaseModel

# app = FastAPI()

# books = [
#     {
#         "id": 1,
#         "title": "Асинхронное программирование в Python",
#         "author": "Мотье Бал",
#     },
#     {
#         "id": 2,
#         "title": "Изучаем Python",
#         "author": "Марк Лутц",
#     }
# ]

# @app.get("/books",
#          tags=["Книги"],
#          summary="Получить список всех книг",
#          )
# def read_books():
#     return books



# @app.get("/books{book_id}",
#          tags=["Книги"],
#          summary="Получить конкретную книгу по ID",
#          )
# def get_book(book_id: int):
#     for book in books:
#         if book["id"] == book_id:
#             return book
#     raise HTTPException(status_code=404, detail="Book not found")



# class NewBook(BaseModel):
#     title: str
#     author: str

# @app.post("/books", tags=["Книги"])
# def create_book(new_book: NewBook):
#     books.append({
#         "id": len(books) + 1,
#         "title": new_book.title,
#         "author": new_book.author
#     })
#     return { 
#         "success": True,
#         "message": "Book added successfully"
#         }


# if __name__ == "__main__":
#     uvicorn.run("main:app", reload=True)
 
#------------------------------------------------------------------
# пример 2

# from fastapi import FastAPI, Depends
# from typing import Annotated
# from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
# from sqlalchemy import select
# from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
# from pydantic import BaseModel


# app = FastAPI()

# engine = create_async_engine("sqlite+aiosqlite:///books.db", echo=True)

# new_session = async_sessionmaker(engine, expire_on_commit=False) 

# async def get_session():
#     async with new_session() as session:
#         yield session

# SessionDep = Annotated[AsyncSession, Depends(get_session)]

# class Base(DeclarativeBase):
#     pass


# class BookModel(Base):
#     __tablename__ = "books"

#     from sqlalchemy import Integer, String, Column

#     id: Mapped[int] = mapped_column(primary_key=True)
#     title: Mapped[str]
#     author: Mapped[str]

# @app.post('/setup_database')
# async def seed_database():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.drop_all)
#         await conn.run_sync(Base.metadata.create_all)
#     return {"ok": True}


# class BookAddSchema(BaseModel):
#     title: str
#     author: str

# class BookSchema(BookAddSchema):
#     id: int

# @app.post('/books')
# async def add_book(data: BookAddSchema, session: SessionDep):
#     new_book = BookModel(
#         title=data.title,
#         author=data.author
#     )
#     session.add(new_book)
#     await session.commit()
#     return {"ok": True}


# @app.get('/books')
# async def get_books(session: SessionDep):
#     query = select(BookModel)
#     result = await session.execute(query)
#     return result.scalars().all()




#--------------------------------------------------
# пример 3

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import StreamingResponse, FileResponse

# app = FastAPI()

# @app.post("/files")
# async def upload_file(uploaded_file: UploadFile):
#     file = uploaded_file.file
#     file_name = uploaded_file.filename
#     with open(f"1_{file_name}", 'wb') as f:
#         f.write(file.read())


# @app.post("/multiple_files")
# async def upload_file(uploaded_files: list[UploadFile]):
#     for uploaded_files in uploaded_files:
#         file = uploaded_files.file
#         file_name = uploaded_files.filename
#         with open(f"1_{file_name}", 'wb') as f:
#             f.write(file.read())


# @app.get("/files/filename")
# async def get_file(file_name: str):
#     return FileResponse(path=file_name, filename=file_name)

#--------------------------------------------------

from typing import AsyncGenerator, List, Optional
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import String, Integer, select, delete
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


# Database setup
BASE_DIR = Path(__file__).parent
DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR}/test.db"
engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=True,
    future=True
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)

# sqlalchemy модель

class Base(DeclarativeBase):
    """
    Базовый класс для всех моделей SQLAlchemy
    """
    pass

