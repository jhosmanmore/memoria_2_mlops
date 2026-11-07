"""Pruebas unitarias para la API REST de inferencia (src/api.py)."""
from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient

from src.api import app, estado_servicio

client = TestClient(app)


def test_raiz():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "documentacion" in data


def test_health_sin_modelo():
    estado_servicio["modelo"] = None
    estado_servicio["version"] = None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_predict_sin_modelo():
    estado_servicio["modelo"] = None
    payload = {
        "marca": "Toyota",
        "tipo_vehiculo": "SUV",
        "año_modelo": 2022,
        "kilometraje": 35000.0,
        "precio_lista": 28500.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503


def test_predict_con_modelo_mock():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([25450.75])
    estado_servicio["modelo"] = mock_model
    estado_servicio["version"] = "2"
    estado_servicio["alias"] = "champion"

    payload = {
        "marca": "Toyota",
        "tipo_vehiculo": "SUV",
        "año_modelo": 2022,
        "kilometraje": 35000.0,
        "precio_lista": 28500.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["precio_predicho_usd"] == 25450.75
    assert data["modelo_version"] == "2"
    assert data["modelo_alias"] == "champion"


def test_predict_batch_con_modelo_mock():
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([25000.0, 18500.0])
    estado_servicio["modelo"] = mock_model
    estado_servicio["version"] = "2"
    estado_servicio["alias"] = "champion"

    payload = [
        {
            "marca": "Toyota",
            "tipo_vehiculo": "SUV",
            "año_modelo": 2022,
            "kilometraje": 35000.0,
            "precio_lista": 28500.0,
        },
        {
            "marca": "Renault",
            "tipo_vehiculo": "Sedán",
            "año_modelo": 2018,
            "kilometraje": 80000.0,
            "precio_lista": 19000.0,
        },
    ]
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["precio_predicho_usd"] == 25000.0
    assert data[1]["precio_predicho_usd"] == 18500.0
