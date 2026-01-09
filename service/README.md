**FastAPI ML service. Выполняет инференс обученной ML-модели для детектирования фрода**

Модель: RandomForestClassifier
Таргет: isFraud (fraud=1, not_fraud=0)

**Запуск**
 - созать и активировать venv
 - pip install -r requirements.txt
 - заупстить Linear_model.ipynb. В конце наутбука находится ячейка сохраняющая модель в pkl 
 - uvicorn service.main:app --reload

**API. POST /forward**

Вход > Content-Type: application/json > JSON с предобработанными признаками (как в X_train)

**Пример JSONa**
{
  "features": {
    "amount": 1234.56,
    "oldbalanceOrg": 5000,
    "newbalanceOrig": 3765.44,
    "type_CASH_OUT": 0,
    "type_DEBIT": 0,
    "type_PAYMENT": 1,
    "type_TRANSFER": 0,
    "Card Type_Gold": 0,
    "Card Type_Mass": 1,
    "Card Type_Platinum": 0,
    "Card Type_Signature": 0,
    "Card Type_Silver": 0,
    "Exp Type_Entertainment": 0,
    "Exp Type_Food": 1,
    "Exp Type_Fuel": 0,
    "Exp Type_Grocery": 0,
    "Exp Type_Health_Fitness": 0,
    "Exp Type_Home": 0,
    "Exp Type_Personal_Care": 0,
    "Exp Type_Travel": 0,
    "Gender_M": 1,
    "City_TargetEncoded": 0.42
  }
}

В Headers указываются **X-Threshold** (порог перехода вероятности, 0.5 по дефолту) и **X-Return-Proba** (отобразить в ответе вероятности или нет)

**Response 200**
{
  "is_anomaly": false,
  "proba": 0.19,
  "threshold": 0.5
}


**Ошибки**
400 > bad request (неверный формат)

403 > модель не смогла обработать данные

**!!!Примечание!!!**

Предобработка выполняется вне сервиса, ожидаются уже предобработанные признаки.
