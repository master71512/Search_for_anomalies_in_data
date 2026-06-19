from __future__ import annotations
from pathlib import Path
from typing import Optional
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, Body, Request, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI(
    title='Search_for_anomalies_in_data_FastAPI_Service',
    description=(
        'ML-сервис для инференса модели RandomForest по данным подготовленным аналогично трейну \n\n'
        'Целевая переменная isFraud (0 not fraud, 1 fraud). В ответе is_anomaly=True соответствует класс 1'
    ),
)

FEATURES = [
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'type_CASH_OUT',
    'type_DEBIT',
    'type_PAYMENT',
    'type_TRANSFER',
    'Card Type_Gold',
    'Card Type_Mass',
    'Card Type_Platinum',
    'Card Type_Signature',
    'Card Type_Silver',
    'Exp Type_Entertainment',
    'Exp Type_Food',
    'Exp Type_Fuel',
    'Exp Type_Grocery',
    'Exp Type_Health_Fitness',
    'Exp Type_Home',
    'Exp Type_Personal_Care',
    'Exp Type_Travel',
    'Gender_M',
    'City_TargetEncoded',
]



# Pydantic валидация
class FeaturesIn(BaseModel):
    '''
    Это уже ПРЕДОБРАБОТАННЫЕ признаки как в X_train после OHE/target encoding
    Бинарные значение передабются как 0/1
    '''

    model_config = ConfigDict(extra='forbid')  # запрет лишних полей

    amount: float = Field(..., description='Сумма операции')
    oldbalanceOrg: float = Field(..., description='Баланс отправителя ДО операции')
    newbalanceOrig: float = Field(..., description='Баланс отправителя ПОСЛЕ операции')

    type_CASH_OUT: float = Field(..., description='OHE: тип операции CASH_OUT')
    type_DEBIT: float = Field(..., description='OHE: тип операции DEBIT')
    type_PAYMENT: float = Field(..., description='OHE: тип операции PAYMENT')
    type_TRANSFER: float = Field(..., description='OHE: тип операции TRANSFER')

    Card_Type_Gold: float = Field(..., alias='Card Type_Gold', description='OHE: Card Type Gold')
    Card_Type_Mass: float = Field(..., alias='Card Type_Mass', description='OHE: Card Type Mass')
    Card_Type_Platinum: float = Field(..., alias='Card Type_Platinum', description='OHE: Card Type Platinum')
    Card_Type_Signature: float = Field(..., alias='Card Type_Signature', description='OHE: Card Type Signature')
    Card_Type_Silver: float = Field(..., alias='Card Type_Silver', description='OHE: Card Type Silver')

    Exp_Type_Entertainment: float = Field(..., alias='Exp Type_Entertainment', description='OHE: Exp Type Entertainment')
    Exp_Type_Food: float = Field(..., alias='Exp Type_Food', description='OHE: Exp Type Food')
    Exp_Type_Fuel: float = Field(..., alias='Exp Type_Fuel', description='OHE: Exp Type Fuel')
    Exp_Type_Grocery: float = Field(..., alias='Exp Type_Grocery', description='OHE: Exp Type Grocery')
    Exp_Type_Health_Fitness: float = Field(..., alias='Exp Type_Health_Fitness', description='OHE: Exp Type Health_Fitness')
    Exp_Type_Home: float = Field(..., alias='Exp Type_Home', description='OHE: Exp Type Home')
    Exp_Type_Personal_Care: float = Field(..., alias='Exp Type_Personal_Care', description='OHE: Exp Type Personal_Care')
    Exp_Type_Travel: float = Field(..., alias='Exp Type_Travel', description='OHE: Exp Type Travel')

    Gender_M: float = Field(..., description='OHE: Gender_M (0/1)')
    City_TargetEncoded: float = Field(..., description='Target encoding для города (float)')

    # чтобы можно было отправлять ключи как в FEATURES (с пробелами)
    model_config = ConfigDict(extra='forbid', populate_by_name=True)


class ForwardIn(BaseModel):
    '''
    Тело запроса -> одна транзакция
    '''
    model_config = ConfigDict(extra='forbid')
    features: FeaturesIn = Field(
        ...,
        description='Набор признаков (уже предобработанных как в обучении)'
    )


class ForwardOut(BaseModel):
    '''
    Ответ модели
    '''
    is_anomaly: bool = Field(..., description='True это 1 это fraud')
    proba: Optional[float] = Field(None, description='Вероятность что фрод, если включен X-Return-Proba')
    threshold: float = Field(..., description='Заданный порог перехода вероятности')


# 400, bad request
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return PlainTextResponse('bad request', status_code=400)

# загрузка модели
_MODEL = None

def load_model():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    model_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "minio_data"
        / "mlflow-bucket"
        / "mlflow"
        / "fe15dbe84f8f4ad48f24487999bec679"
        / "artifacts"
        / "model"
    )

    if not model_path.exists():
        raise FileNotFoundError(f"CatBoost MLflow model not found: {model_path}")

    _MODEL = mlflow.pyfunc.load_model(str(model_path))
    return _MODEL

def prediction(features: dict, threshold: float) -> dict:
    '''
    ValueError - 400, не подходящий формат
    Other Exception - 403, модель не сработала
    '''
    model = load_model()

    if set(features.keys()) != set(FEATURES):
        raise ValueError('wrong feature set')

    try:
        row = {name: float(features[name]) for name in FEATURES}
    except Exception:
        raise ValueError('bad feature types')

    X = pd.DataFrame([row], columns=FEATURES)

    if hasattr(model, 'predict_proba'):
        proba_1 = float(model.predict_proba(X)[0][1])  # class 1 => fraud
        is_anomaly = bool(proba_1 >= threshold)
        return {'is_anomaly': is_anomaly, 'proba': proba_1}

    pred = int(model.predict(X)[0])  # 1 => fraud
    return {'is_anomaly': bool(pred), 'proba': None}


@app.post('/forward', response_model=ForwardOut, 
    responses={
            400: {
                'description': 'bad request',
                'content': {'text/plain': {'example': 'bad request'}},
            },
            403: {
                'description': 'model failed',
                'content': {'text/plain': {'example': 'модель не смогла обработать данные'}},
            },
        }, summary='Прогон одного объекта через модель',
)
async def forward(
    request: Request,
    inp: ForwardIn = Body(..., description='JSON с полем features (ожидаются 22 признака)'),
    x_threshold: float = Header(
        0.5,
        alias='X-Threshold',
        ge=0.0,
        le=1.0,
        description='Порог по вероятности для fraud',
    ),
    x_return_proba: bool = Header(
        True,
        alias='X-Return-Proba',
        description='Если false, то поле proba не возвращается.',
    ),
):
    ct = (request.headers.get('content-type') or '').lower()
    if 'application/json' not in ct:
        return PlainTextResponse('bad request', status_code=400)

    # Инференс
    try:
        features_dict = inp.features.model_dump(by_alias=True)  # ключи как в FEATURES (с пробелами)
        result = prediction(features_dict, threshold=x_threshold)
    except ValueError:
        return PlainTextResponse('bad request', status_code=400)
    except Exception:
        return PlainTextResponse('модель не смогла обработать данные', status_code=403)

    # форматирование ответа
    out = {
        'is_anomaly': result['is_anomaly'],
        'threshold': float(x_threshold),
        'proba': result.get('proba'),
    }
    if not x_return_proba:
        out['proba'] = None  # можно и удалить ключ, но тогда response_model будет “прыгать”
    return JSONResponse(status_code=200, content=out)


@app.get('/')
async def root():
    return {'ok': True, 'docs': '/docs'}


