"""API REST con FastAPI para servir el modelo Champion de precio de venta de vehículos.

Requisitos de la actividad:
  - Carga el modelo por su alias 'champion' desde MLflow Model Registry.
  - Expone un endpoint POST /predict para predecir el precio con los datos de un cliente.
  - Expone endpoints auxiliares /health y /model-info para verificar la versión activa.
  - Al reiniciar la API tras una actualización del champion, sirve la nueva versión sin tocar código.

Ejecución:
  $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
  $env:PYTHONIOENCODING = "utf-8"
  $env:PYTHONUTF8 = "1"
  poetry run uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
"""
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

from src.config import cargar_config

# Estado global del servicio
estado_servicio: dict[str, Any] = {
    "modelo": None,
    "version": None,
    "alias": None,
    "run_id": None,
    "nombre_modelo": None,
    "error_carga": None,
}


def cargar_modelo_champion() -> None:
    """Carga el modelo champion desde MLflow Model Registry y actualiza el estado global."""
    config = cargar_config()
    nombre_modelo = config.get("mlflow", {}).get("nombre_modelo", "precio-vehiculos-model")
    alias_champion = config.get("mlflow", {}).get("alias_champion", "champion")
    estado_servicio["nombre_modelo"] = nombre_modelo
    estado_servicio["alias"] = alias_champion

    client = MlflowClient()
    try:
        model_version_info = client.get_model_version_by_alias(nombre_modelo, alias_champion)
        version_str = str(model_version_info.version)
        run_id_str = str(model_version_info.run_id)

        modelo_uri = f"models:/{nombre_modelo}@{alias_champion}"
        modelo_cargado = mlflow.sklearn.load_model(modelo_uri)

        estado_servicio["modelo"] = modelo_cargado
        estado_servicio["version"] = version_str
        estado_servicio["run_id"] = run_id_str
        estado_servicio["error_carga"] = None
        print(f"[API] Modelo Champion cargado exitosamente: {nombre_modelo}@{alias_champion} (v{version_str})")
    except Exception as exc:
        estado_servicio["modelo"] = None
        estado_servicio["version"] = None
        estado_servicio["error_carga"] = str(exc)
        print(f"[API] ADVERTENCIA: No se pudo cargar el modelo champion: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida de la aplicación: inicializa el modelo al arrancar."""
    cargar_modelo_champion()
    yield


app = FastAPI(
    title="API de Predicción de Precios de Vehículos Usados",
    description=(
        "Servicio REST que expone el modelo Champion registrado en MLflow Model Registry. "
        "Permite estimar el precio final de venta a partir de las características del vehículo."
    ),
    version="0.2.0",
    lifespan=lifespan,
)


# -----------------------------------------------------------------------------
# Esquemas Pydantic
# -----------------------------------------------------------------------------
class DatosVehiculo(BaseModel):
    marca: str = Field(
        ...,
        examples=["Toyota"],
        description="Marca del vehículo (Toyota, Chevrolet, Renault, Mazda, Nissan, Kia, Ford)",
    )
    tipo_vehiculo: str = Field(
        ...,
        examples=["SUV"],
        description="Tipo de carrocería (Sedán, SUV, Hatchback, Pick-up)",
    )
    año_modelo: int = Field(
        ...,
        ge=2000,
        le=2030,
        examples=[2022],
        description="Año de fabricación del modelo",
    )
    kilometraje: float = Field(
        ...,
        ge=0.0,
        examples=[35000.0],
        description="Kilometraje recorrido en kilómetros",
    )
    precio_lista: float = Field(
        ...,
        gt=0.0,
        examples=[28500.0],
        description="Precio de lista oficial o sugerido en USD",
    )


class RespuestaPrediccion(BaseModel):
    precio_predicho_usd: float = Field(..., description="Precio de venta estimado en USD")
    modelo_version: str = Field(..., description="Versión del modelo Champion que generó la predicción")
    modelo_alias: str = Field(..., description="Alias utilizado para servir el modelo ('champion')")
    timestamp: str = Field(..., description="Marca de tiempo UTC de la respuesta")


class InformacionModelo(BaseModel):
    nombre_modelo: str
    modelo_alias: str
    modelo_version: str | None
    run_id: str | None
    estado: str
    error: str | None = None


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/", tags=["Información"])
def inicio():
    """Ruta raíz de bienvenida con resumen del estado del servicio."""
    return {
        "mensaje": "Bienvenido a la API de Inferencia de Precios de Vehículos Usados",
        "modelo": estado_servicio.get("nombre_modelo"),
        "alias": estado_servicio.get("alias"),
        "version_activa": estado_servicio.get("version"),
        "documentacion": "/docs",
    }


@app.get("/health", tags=["Monitoreo"])
def health():
    """Endpoint de comprobación de salud del servicio y disponibilidad del modelo."""
    if estado_servicio.get("modelo") is None:
        return {
            "status": "degraded",
            "mensaje": "Servicio activo pero el modelo Champion aún no está disponible.",
            "detalle": estado_servicio.get("error_carga"),
        }
    return {
        "status": "healthy",
        "modelo": estado_servicio.get("nombre_modelo"),
        "version_champion": estado_servicio.get("version"),
    }


@app.get("/model-info", response_model=InformacionModelo, tags=["Información"])
def model_info():
    """Devuelve los metadatos del modelo champion actualmente en memoria."""
    return InformacionModelo(
        nombre_modelo=estado_servicio.get("nombre_modelo") or "desconocido",
        modelo_alias=estado_servicio.get("alias") or "champion",
        modelo_version=estado_servicio.get("version"),
        run_id=estado_servicio.get("run_id"),
        estado="cargado" if estado_servicio.get("modelo") is not None else "no_cargado",
        error=estado_servicio.get("error_carga"),
    )


@app.post("/predict", response_model=RespuestaPrediccion, tags=["Inferencia"])
def predict(datos: DatosVehiculo):
    """Calcula la predicción del precio de venta para un vehículo individual."""
    modelo = estado_servicio.get("modelo")
    if modelo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El modelo Champion no está disponible en la API. Asegúrese de que MLflow "
                "esté en ejecución y ejecute el pipeline de entrenamiento para generar el champion."
            ),
        )

    # Construir DataFrame compatible con el pipeline de scikit-learn
    df_input = pd.DataFrame([datos.model_dump()])

    try:
        prediccion = float(modelo.predict(df_input)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error durante el cálculo de la predicción: {exc}",
        ) from exc

    return RespuestaPrediccion(
        precio_predicho_usd=round(prediccion, 2),
        modelo_version=str(estado_servicio.get("version")),
        modelo_alias=str(estado_servicio.get("alias")),
        timestamp=datetime.now().isoformat(),
    )


@app.post("/predict/batch", response_model=list[RespuestaPrediccion], tags=["Inferencia"])
def predict_batch(lote: list[DatosVehiculo]):
    """Calcula predicciones para una lista de vehículos."""
    modelo = estado_servicio.get("modelo")
    if modelo is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="El modelo Champion no está disponible en la API.",
        )

    if not lote:
        return []

    df_input = pd.DataFrame([item.model_dump() for item in lote])
    try:
        predicciones = modelo.predict(df_input)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error al predecir lote: {exc}",
        ) from exc

    ahora = datetime.now().isoformat()
    return [
        RespuestaPrediccion(
            precio_predicho_usd=round(float(p), 2),
            modelo_version=str(estado_servicio.get("version")),
            modelo_alias=str(estado_servicio.get("alias")),
            timestamp=ahora,
        )
        for p in predicciones
    ]
