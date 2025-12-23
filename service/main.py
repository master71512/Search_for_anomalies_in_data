from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, Body, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

app = FastAPI(title="Search_for_anomalies_in_data_FastAPI_Service")

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
    "Card Type_Gold",
    "Card Type_Mass",
    "Card Type_Platinum",
    "Card Type_Signature",
    "Card Type_Silver",
    "Exp Type_Entertainment",
    "Exp Type_Food",
    "Exp Type_Fuel",
    "Exp Type_Grocery",
    "Exp Type_Health_Fitness",
    "Exp Type_Home",
    "Exp Type_Personal_Care",
    "Exp Type_Travel",
    "Gender_M",
    "City_TargetEncoded",
]


class ForwardIn(BaseModel):
    features: dict[str, Any] = Field(..., description="One transaction features (preprocessed)")


# --- превращаем стандартный 422 (ошибка валидации) в 400 'bad request'
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=400, content="bad request")


_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_path = Path(__file__).resolve().parents[1] / "models" / "random_forest_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        _MODEL = pickle.load(f)

    return _MODEL


def _parse_bool_header(val: str | None, default: bool) -> bool:
    if val is None:
        return default
    v = val.strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    return default


def _parse_threshold(val: str | None, default: float = 0.5) -> float:
    if val is None:
        return default
    try:
        t = float(val)
    except Exception:
        return default
    if t < 0.0:
        return 0.0
    if t > 1.0:
        return 1.0
    return t


def prediction(features: dict[str, Any], threshold: float) -> dict[str, Any]:
    """
    ValueError -> неверный формат (400)
    Other Exception -> модель не смогла (403)
    """
    model = _load_model()

    expected = set(FEATURES)
    keys = set(features.keys())

    if keys != expected:
        raise ValueError("wrong feature set")

    row = {}
    for name in FEATURES:
        row[name] = float(features[name])

    X = pd.DataFrame([row], columns=FEATURES)

    if hasattr(model, "predict_proba"):
        proba_1 = float(model.predict_proba(X)[0][1])  # class=1 => fraud
        is_anomaly = bool(proba_1 >= threshold)
        return {"is_anomaly": is_anomaly, "proba": proba_1}

    pred = int(model.predict(X)[0])  # 1 => fraud
    return {"is_anomaly": bool(pred), "proba": None}


@app.post("/forward")
async def forward(
    inp: ForwardIn = Body(...),
    request: Request = None,
):
    """
    - неверный формат -> 400 'bad request'
    - модель не смогла -> 403 'модель не смогла обработать данные'
    - успех -> 200 JSON
    """
    # headers: дополнительные параметры
    threshold = _parse_threshold(request.headers.get("x-threshold"), default=0.5)
    return_proba = _parse_bool_header(request.headers.get("x-return-proba"), default=True)

    try:
        result = prediction(inp.features, threshold=threshold)
    except ValueError:
        return JSONResponse(status_code=400, content="bad request")
    except Exception:
        return JSONResponse(status_code=403, content="модель не смогла обработать данные")

    if not return_proba:
        result.pop("proba", None)

    result["threshold"] = threshold
    return JSONResponse(status_code=200, content=result)


@app.get("/")
async def root():
    return {"ok": True, "docs": "/docs"}
