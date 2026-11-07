"""Gestión y selección dinámica de características (configs/features.yaml)."""
from typing import Any

from src.config import cargar_features_config

OPCIONES_SELECCION = {"ALL", "NUMERIC", "CUSTOM"}


def obtener_features(config_features: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resuelve la lista de variables categóricas y numéricas según el alias 'seleccion'.

    Opciones:
      - ALL: todas las variables numéricas y categóricas.
      - NUMERIC: únicamente variables numéricas (categóricas = []).
      - CUSTOM: subconjunto especificado en el bloque 'custom'.

    Retorna un diccionario con:
      - alias: 'ALL' | 'NUMERIC' | 'CUSTOM'
      - categoricas: list[str]
      - numericas: list[str]
      - todas: list[str] (unión ordenada)
      - n_features: int (total de variables explicativas)
    """
    if config_features is None:
        config_features = cargar_features_config()

    seleccion = str(config_features.get("seleccion", "ALL")).strip().upper()
    if seleccion not in OPCIONES_SELECCION:
        raise ValueError(
            f"Opción de selección no soportada: '{seleccion}'. Opciones válidas: {sorted(OPCIONES_SELECCION)}"
        )

    bloque_todas = config_features.get("todas", {})
    bloque_custom = config_features.get("custom", {})

    if seleccion == "ALL":
        categoricas = list(bloque_todas.get("categoricas", []))
        numericas = list(bloque_todas.get("numericas", []))
    elif seleccion == "NUMERIC":
        categoricas = []
        numericas = list(bloque_todas.get("numericas", []))
    elif seleccion == "CUSTOM":
        categoricas = list(bloque_custom.get("categoricas", []))
        numericas = list(bloque_custom.get("numericas", []))

    todas = categoricas + numericas
    if not todas:
        raise ValueError(f"La selección '{seleccion}' no produjo ninguna característica.")

    return {
        "alias": seleccion,
        "categoricas": categoricas,
        "numericas": numericas,
        "todas": todas,
        "n_features": len(todas),
    }
