# Доступ к сервисам на VM

Docker Compose публикует сервисы для команды на всех сетевых интерфейсах VM.

## Публичные адреса

- FastAPI backend: `http://201.51.9.227:8000`
- документация FastAPI: `http://201.51.9.227:8000/docs`
- MLflow UI: `http://201.51.9.227:5050`

## Порты

Порты настраиваются в `.env`:

- `BACKEND_PORT=8000`
- `MLFLOW_PORT=5050`
- `PUBLIC_HOST=201.51.9.227`

## Запуск или обновление сервисов

Выполнить на VM:

```bash
docker compose up -d --build backend-service mlflow-service
```

Firewall или security group VM должны разрешать входящий TCP-трафик на порты `8000` и `5050`.
