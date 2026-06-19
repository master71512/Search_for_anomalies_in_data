### FastAPI ML service

Сервис выполняет инференс CatBoost-модели для детектирования фрода.

Модель: `CatBoostClassifier`, загружается через MLflow artifacts.

Таргет: `isFraud` (`fraud=1`, `not_fraud=0`).

#### Запуск

- создать и активировать venv;
- установить зависимости: `pip install -r requirements.txt`;
- убедиться, что доступны `data/raw/Fraud.csv` и MLflow artifacts модели;
- запустить сервис: `uvicorn service.main:app --reload`.

#### API. POST /forward

Вход: `Content-Type: application/json`.

Сервис ожидает сырые признаки транзакции без `Date`, `nameOrig` и `isFraud`. Предобработка выполняется внутри сервиса так же, как в `notebooks/2_Linear_model.ipynb`: `StandardScaler`, OHE и target encoding для `City`.

**Пример JSON:**

```json
{
  "features": {
    "amount": 1234.56,
    "oldbalanceOrg": 5000,
    "newbalanceOrig": 3765.44,
    "City": "Achalpur, India",
    "type": "PAYMENT",
    "Card Type": "Mass",
    "Exp Type": "Food",
    "Gender": "M"
  }
}
```

В headers указываются:

- `X-Threshold`: порог перехода вероятности, по умолчанию `0.5`;
- `X-Return-Proba`: возвращать вероятность или `null`.

**Response 200:**

```json
{
  "is_anomaly": false,
  "proba": 0.19,
  "threshold": 0.5
}
```

**Error 400:** `bad request`

**Error 403:** `model failed`
