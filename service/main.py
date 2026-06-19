from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal, Optional

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

app = FastAPI(
    title='Search_for_anomalies_in_data_FastAPI_Service',
    description=(
        'ML-сервис для инференса CatBoost-модели. Перед вызовом модели сервис полностью повторяет этапы '
        'предобработки данных из ноутбука notebooks/2_Linear_model.ipynb. Целевая переменная - isFraud: значение '
        'is_anomaly=True соответствует классу 1 (фрод), а is_anomaly=False - классу 0 (нормальная транзакция).'
    ),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'Fraud.csv'
MODEL_ARTIFACT_PATH = (
    PROJECT_ROOT
    / 'data'
    / 'minio_data'
    / 'mlflow-bucket'
    / 'mlflow'
    / 'fe15dbe84f8f4ad48f24487999bec679'
    / 'artifacts'
    / 'model'
)
MATERIALIZED_MODEL_PATH = PROJECT_ROOT / '.runtime' / 'catboost_mlflow_model'

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
NUMERIC_FEATURES = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
OHE_FEATURES = FEATURES[3:-1]

TYPE_BASELINE = 'CASH_IN'
CARD_TYPE_BASELINE = 'Classic'
EXP_TYPE_BASELINE = 'Bills'
GENDER_BASELINE = 'F'

_MODEL = None
_PREPROCESSING = None


class FeaturesIn(BaseModel):
    """Сырые признаки транзакции до предобработки из ноутбука."""

    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    amount: float = Field(..., allow_inf_nan=False, description='Сумма операции')
    oldbalanceOrg: float = Field(..., allow_inf_nan=False, description='Баланс отправителя до операции')
    newbalanceOrig: float = Field(..., allow_inf_nan=False, description='Баланс отправителя после операции')
    City: str = Field(..., min_length=1, description='Город транзакции')
    transaction_type: Literal['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'] = Field(
        ...,
        alias='type',
        description='Тип операции',
    )
    card_type: Literal['Classic', 'Gold', 'Mass', 'Platinum', 'Signature', 'Silver'] = Field(
        ...,
        alias='Card Type',
        description='Тип карты',
    )
    exp_type: Literal[
        'Bills',
        'Entertainment',
        'Food',
        'Fuel',
        'Grocery',
        'Health_Fitness',
        'Home',
        'Personal_Care',
        'Travel',
    ] = Field(..., alias='Exp Type', description='Тип расхода')
    Gender: Literal['F', 'M'] = Field(..., description='Пол клиента')


class ForwardIn(BaseModel):
    """Тело запроса для одной транзакции."""

    model_config = ConfigDict(extra='forbid')
    features: FeaturesIn = Field(..., description='Сырые признаки транзакции')


class ForwardOut(BaseModel):
    """Ответ модели."""

    is_anomaly: bool = Field(..., description='Значение true означает класс 1, то есть фрод')
    proba: Optional[float] = Field(None, description='Вероятность фрода, если включен X-Return-Proba')
    threshold: float = Field(..., description='Порог принятия решения')


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return PlainTextResponse('некорректный запрос', status_code=400)


def _read_minio_inline_file(path: Path, marker: bytes) -> bytes:
    data = (path / 'xl.meta').read_bytes()
    start = data.find(marker)
    if start == -1:
        raise FileNotFoundError(f'Не удалось материализовать файл MLflow-артефакта: {path}')
    return data[start:]


def _materialize_minio_model(model_path: Path) -> Path:
    if (model_path / 'MLmodel').is_file():
        return model_path

    if (MATERIALIZED_MODEL_PATH / 'MLmodel').is_file() and (MATERIALIZED_MODEL_PATH / 'model.cb').is_file():
        return MATERIALIZED_MODEL_PATH

    runtime_root = MATERIALIZED_MODEL_PATH.parent.resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    tmp_path = runtime_root / f'{MATERIALIZED_MODEL_PATH.name}.tmp'

    if not str(tmp_path.resolve()).startswith(str(runtime_root)):
        raise RuntimeError(f'Небезопасный путь для материализованной модели: {tmp_path}')

    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True)

    inline_files = {
        'MLmodel': b'artifact_path:',
        'conda.yaml': b'channels:',
        'python_env.yaml': b'python:',
        'requirements.txt': b'mlflow==',
    }
    for file_name, marker in inline_files.items():
        (tmp_path / file_name).write_bytes(_read_minio_inline_file(model_path / file_name, marker))

    model_parts = list((model_path / 'model.cb').glob('*/part.1'))
    if len(model_parts) != 1:
        raise FileNotFoundError(f'Не удалось материализовать CatBoost model.cb из: {model_path / "model.cb"}')

    model_bytes = model_parts[0].read_bytes()
    model_start = model_bytes.find(b'CBM1')
    if model_start == -1:
        raise FileNotFoundError(f'Не найден маркер CatBoost CBM в файле: {model_parts[0]}')
    (tmp_path / 'model.cb').write_bytes(model_bytes[model_start:])

    if MATERIALIZED_MODEL_PATH.exists():
        shutil.rmtree(MATERIALIZED_MODEL_PATH)
    tmp_path.rename(MATERIALIZED_MODEL_PATH)
    return MATERIALIZED_MODEL_PATH


def load_model():
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(f'CatBoost MLflow-модель не найдена: {MODEL_ARTIFACT_PATH}')

    model_path = _materialize_minio_model(MODEL_ARTIFACT_PATH)
    _MODEL = mlflow.pyfunc.load_model(str(model_path))
    return _MODEL


def load_preprocessing():
    global _PREPROCESSING

    if _PREPROCESSING is not None:
        return _PREPROCESSING

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f'Обучающие данные для предобработки не найдены: {RAW_DATA_PATH}')

    df = pd.read_csv(RAW_DATA_PATH)
    df_new = df.drop(['Date', 'nameOrig'], axis=1)
    df_new_encoded = pd.get_dummies(
        df_new,
        columns=['type', 'Card Type', 'Exp Type', 'Gender'],
        drop_first=True,
    )

    X = df_new_encoded.drop('isFraud', axis=1)
    y = df_new_encoded['isFraud']
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    scaler.fit(X_train[NUMERIC_FEATURES])

    feature = 'City'
    target = 'isFraud'
    min_samples_leaf = 50
    smoothing = 20
    global_mean = float(y_train.mean())

    temp_df = X_train[[feature]].copy()
    temp_df[target] = y_train
    agg = temp_df.groupby(feature)[target].agg(['count', 'mean'])
    agg.columns = ['counts', 'mean_target']
    agg['lambda'] = 1 / (1 + np.exp((min_samples_leaf - agg['counts']) / smoothing))
    agg['smoothed_target'] = agg['lambda'] * agg['mean_target'] + (1 - agg['lambda']) * global_mean

    _PREPROCESSING = {
        'scaler_mean': dict(zip(NUMERIC_FEATURES, scaler.mean_)),
        'scaler_scale': dict(zip(NUMERIC_FEATURES, scaler.scale_)),
        'city_encoding_map': agg['smoothed_target'].to_dict(),
        'global_mean': global_mean,
    }
    return _PREPROCESSING


def build_model_input(features: FeaturesIn) -> pd.DataFrame:
    preprocessing = load_preprocessing()
    raw = features.model_dump(by_alias=True)

    row = {}
    for name in NUMERIC_FEATURES:
        row[name] = (
            float(raw[name]) - float(preprocessing['scaler_mean'][name])
        ) / float(preprocessing['scaler_scale'][name])

    row.update({name: False for name in OHE_FEATURES})

    if raw['type'] != TYPE_BASELINE:
        row[f"type_{raw['type']}"] = True
    if raw['Card Type'] != CARD_TYPE_BASELINE:
        row[f"Card Type_{raw['Card Type']}"] = True
    if raw['Exp Type'] != EXP_TYPE_BASELINE:
        row[f"Exp Type_{raw['Exp Type']}"] = True
    if raw['Gender'] != GENDER_BASELINE:
        row[f"Gender_{raw['Gender']}"] = True

    row['City_TargetEncoded'] = float(
        preprocessing['city_encoding_map'].get(raw['City'], preprocessing['global_mean'])
    )

    X = pd.DataFrame([row], columns=FEATURES)
    X[NUMERIC_FEATURES + ['City_TargetEncoded']] = X[NUMERIC_FEATURES + ['City_TargetEncoded']].astype('float64')
    X[OHE_FEATURES] = X[OHE_FEATURES].astype('bool')
    return X


def _get_raw_catboost_model(model):
    model_impl = getattr(model, '_model_impl', None)
    if model_impl is not None and hasattr(model_impl, 'get_raw_model'):
        return model_impl.get_raw_model()
    raise RuntimeError('Загруженная MLflow pyfunc-модель не предоставляет исходную CatBoost-модель')


def prediction(features: FeaturesIn, threshold: float) -> dict:
    model = load_model()
    X = build_model_input(features)
    raw_model = _get_raw_catboost_model(model)

    proba_1 = float(raw_model.predict_proba(X)[0][1])
    return {'is_anomaly': bool(proba_1 >= threshold), 'proba': proba_1}


@app.post(
    '/forward',
    response_model=ForwardOut,
    responses={
        400: {
            'description': 'Некорректный запрос',
            'content': {'text/plain': {'example': 'некорректный запрос'}},
        },
        403: {
            'description': 'Ошибка модели',
            'content': {'text/plain': {'example': 'модель не смогла обработать данные'}},
        },
    },
    summary='Прогон одной транзакции через модель',
)
async def forward(
    request: Request,
    inp: ForwardIn = Body(..., description='JSON-тело с сырыми признаками транзакции'),
    x_threshold: float = Header(
        0.5,
        alias='X-Threshold',
        ge=0.0,
        le=1.0,
        description='Порог вероятности для определения фрода',
    ),
    x_return_proba: bool = Header(
        True,
        alias='X-Return-Proba',
        description='Если значение равно false, поле proba возвращается как null.',
    ),
):
    content_type = (request.headers.get('content-type') or '').lower()
    if 'application/json' not in content_type:
        return PlainTextResponse('некорректный запрос', status_code=400)

    try:
        result = prediction(inp.features, threshold=x_threshold)
    except ValueError:
        return PlainTextResponse('некорректный запрос', status_code=400)
    except Exception:
        return PlainTextResponse('модель не смогла обработать данные', status_code=403)

    out = {
        'is_anomaly': result['is_anomaly'],
        'threshold': float(x_threshold),
        'proba': result.get('proba'),
    }
    if not x_return_proba:
        out['proba'] = None
    return JSONResponse(status_code=200, content=out)


@app.get('/')
async def root():
    return {'ok': True, 'docs': '/docs'}
