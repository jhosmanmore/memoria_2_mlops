"""Lectura de la configuración del proyecto (configs/config.yaml y configs/features.yaml)."""
from pathlib import Path

import yaml

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_CONFIG = RAIZ_PROYECTO / "configs" / "config.yaml"
RUTA_FEATURES = RAIZ_PROYECTO / "configs" / "features.yaml"


def cargar_config(ruta: str | Path = RUTA_CONFIG) -> dict:
    """Devuelve el contenido de config.yaml como diccionario."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cargar_features_config(ruta: str | Path = RUTA_FEATURES) -> dict:
    """Devuelve el contenido de features.yaml como diccionario."""
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def ruta_absoluta(ruta_relativa: str | Path) -> Path:
    """Convierte una ruta relativa a la raíz del repositorio en absoluta."""
    ruta = Path(ruta_relativa)
    return ruta if ruta.is_absolute() else RAIZ_PROYECTO / ruta
