"""Punto de entrada de la API en services/api/main.py.

Permite ejecutar la API según la estructura de servicios de referencia:
  poetry run uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
from pathlib import Path

# Añadir la raíz del repositorio al PYTHONPATH para importar src
RAIZ = Path(__file__).resolve().parent.parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src.api import app  # noqa: E402, F401

if __name__ == "__main__":
    import uvicorn
    from src.config import cargar_config

    config_api = cargar_config().get("api", {})
    host = config_api.get("host", "0.0.0.0")
    port = int(config_api.get("port", 8000))
    uvicorn.run("services.api.main:app", host=host, port=port, reload=True)
