"""Script para validar el funcionamiento en vivo de la API de FastAPI con el modelo Champion."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from fastapi.testclient import TestClient
from src.api import app


def probar_api():
    print("=== Probando API con modelo Champion ===")
    with TestClient(app) as client:
        # 1. Health
        res_health = client.get("/health")
        print(f"GET /health: {res_health.status_code} -> {res_health.json()}")
        assert res_health.status_code == 200

        # 2. Model Info
        res_info = client.get("/model-info")
        print(f"GET /model-info: {res_info.status_code} -> {res_info.json()}")
        assert res_info.status_code == 200

        # 3. Predict individual
        ejemplo_vehiculo = {
            "marca": "Toyota",
            "tipo_vehiculo": "SUV",
            "año_modelo": 2022,
            "kilometraje": 35000.0,
            "precio_lista": 28500.0,
        }
        res_pred = client.post("/predict", json=ejemplo_vehiculo)
        print(f"POST /predict: {res_pred.status_code} -> {res_pred.json()}")
        assert res_pred.status_code == 200
        pred = res_pred.json()
        print(f"  -> Precio predicho: ${pred['precio_predicho_usd']:,.2f} USD")
        print(f"  -> Versión Champion: v{pred['modelo_version']} (@{pred['modelo_alias']})")

        # 4. Predict lote
        lote_ejemplo = [
            ejemplo_vehiculo,
            {
                "marca": "Renault",
                "tipo_vehiculo": "Sedán",
                "año_modelo": 2019,
                "kilometraje": 70000.0,
                "precio_lista": 18000.0,
            },
        ]
        res_batch = client.post("/predict/batch", json=lote_ejemplo)
        print(f"POST /predict/batch: {res_batch.status_code} -> {len(res_batch.json())} predicciones")
        assert res_batch.status_code == 200

    print("=== ¡Todas las pruebas de la API pasaron exitosamente! ===")


if __name__ == "__main__":
    probar_api()
