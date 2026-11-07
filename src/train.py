"""Construcción, entrenamiento y evaluación del modelo de precio de venta."""
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import obtener_features

MODELOS = {
    "random_forest": RandomForestRegressor,
    "linear_regression": LinearRegression,
}


def _resolver_features(config_modelo: dict[str, Any], features_info: dict[str, Any] | None = None) -> tuple[list[str], list[str]]:
    """Determina las variables categóricas y numéricas a utilizar."""
    if features_info is not None:
        return features_info["categoricas"], features_info["numericas"]

    # Soporte para configuración directa o lectura desde features.yaml
    if "variables_categoricas" in config_modelo and "variables_numericas" in config_modelo:
        return config_modelo["variables_categoricas"], config_modelo["variables_numericas"]

    info = obtener_features()
    return info["categoricas"], info["numericas"]


def construir_pipeline(
    config_modelo: dict[str, Any],
    categoricas: list[str] | None = None,
    numericas: list[str] | None = None,
) -> Pipeline:
    """Preprocesado flexible (one-hot + escalado) y regresor en un objeto Pipeline de scikit-learn."""
    tipo = config_modelo["tipo"]
    if tipo not in MODELOS:
        raise ValueError(f"Modelo no soportado: {tipo}. Opciones: {list(MODELOS)}")

    if categoricas is None or numericas is None:
        categoricas, numericas = _resolver_features(config_modelo)

    transformadores = []
    if categoricas:
        transformadores.append(
            ("categoricas", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categoricas)
        )
    if numericas:
        transformadores.append(
            ("numericas", StandardScaler(), numericas)
        )

    if not transformadores:
        raise ValueError("No se definieron transformadores: lista de variables vacía.")

    preprocesado = ColumnTransformer(transformers=transformadores)
    hiperparametros = config_modelo.get("hiperparametros", {}) if tipo == "random_forest" else {}
    return Pipeline([("preprocesado", preprocesado), ("modelo", MODELOS[tipo](**hiperparametros))])


def dividir_datos(
    df: pd.DataFrame,
    config_modelo: dict[str, Any],
    features_info: dict[str, Any] | None = None,
):
    """Separa variables explicativas y objetivo, y divide en entrenamiento y prueba."""
    categoricas, numericas = _resolver_features(config_modelo, features_info)
    variables = categoricas + numericas
    X = df[variables]
    y = df[config_modelo["objetivo"]]
    return train_test_split(
        X, y, test_size=config_modelo["test_size"], random_state=config_modelo["semilla_split"]
    )


def calcular_metricas(y_real, y_pred) -> dict[str, float]:
    """Métricas de regresión. MAE y RMSE en las mismas unidades que el precio (USD)."""
    return {
        "mae": float(mean_absolute_error(y_real, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_real, y_pred))),
        "r2": float(r2_score(y_real, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_real, y_pred)),
    }


def entrenar_modelo(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config_modelo: dict[str, Any],
    features_info: dict[str, Any] | None = None,
) -> Pipeline:
    """Instancia y entrena el pipeline sobre el conjunto de entrenamiento."""
    categoricas, numericas = _resolver_features(config_modelo, features_info)
    pipeline = construir_pipeline(config_modelo, categoricas=categoricas, numericas=numericas)
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluar_modelo(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calcula las métricas de desempeño sobre el conjunto de prueba."""
    y_pred = pipeline.predict(X_test)
    return calcular_metricas(y_test, y_pred)


def entrenar(df: pd.DataFrame, config_modelo: dict[str, Any], features_info: dict[str, Any] | None = None):
    """Función de compatibilidad: divide datos, entrena y evalúa."""
    X_train, X_test, y_train, y_test = dividir_datos(df, config_modelo, features_info)
    pipeline = entrenar_modelo(X_train, y_train, config_modelo, features_info)
    metricas = evaluar_modelo(pipeline, X_test, y_test)
    return pipeline, metricas, X_train, X_test
