"""Registro y despliegue de los flows de entrenamiento e inferencia en Prefect.

Requisitos de la actividad:
  - La pipeline de entrenamiento y la de inferencia deben aparecer como despliegues separados.
  - La inferencia debe tener un horario diario (cron).
  - El entrenamiento puede lanzarse a mano (sin horario obligatorio o bajo demanda).
  - Ambos deben comprobarse en la UI de Prefect (http://localhost:4200/deployments).

Ejecución:
  $env:PREFECT_API_URL = "http://localhost:4200/api"
  $env:PYTHONIOENCODING = "utf-8"
  $env:PYTHONUTF8 = "1"
  poetry run python flows/desplegar.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Permite importar src/ y flows/

from prefect.client.schemas.schedules import CronSchedule  # noqa: E402
from prefect.deployments import deploy  # noqa: E402

from flows.entrenamiento_flow import entrenamiento_flow  # noqa: E402
from flows.inferencia_flow import inferencia_flow  # noqa: E402
from src.config import cargar_config  # noqa: E402


def crear_despliegues():
    """Construye y registra los dos despliegues en el servidor de Prefect."""
    config = cargar_config()
    cfg_despliegues = config.get("despliegues", {})
    cron_expr = cfg_despliegues.get("cron_inferencia", "0 2 * * *")
    nombre_entrenamiento = cfg_despliegues.get("nombre_entrenamiento", "entrenamiento-despliegue")
    nombre_inferencia = cfg_despliegues.get("nombre_inferencia", "inferencia-diaria-despliegue")

    print(f"[Despliegue] Configurando despliegue de entrenamiento: '{nombre_entrenamiento}' (Manual)")
    despliegue_entrenamiento = entrenamiento_flow.to_deployment(
        name=nombre_entrenamiento,
        tags=["entrenamiento", "mlops", "precio-vehiculos"],
        description="Pipeline de ingesta, entrenamiento, evaluación y registro del Champion en MLflow.",
    )

    print(f"[Despliegue] Configurando despliegue de inferencia: '{nombre_inferencia}' (Cron: {cron_expr})")
    despliegue_inferencia = inferencia_flow.to_deployment(
        name=nombre_inferencia,
        schedule=CronSchedule(cron=cron_expr),
        tags=["inferencia", "mlops", "diario", "precio-vehiculos"],
        description=f"Pipeline de inferencia diaria por lotes con el modelo champion (horario cron: {cron_expr}).",
    )

    # En Prefect 3.x, apply() registra el despliegue localmente en la API sin exigir imagen o bucket
    id_entrenamiento = despliegue_entrenamiento.apply()
    id_inferencia = despliegue_inferencia.apply()

    ids = [id_entrenamiento, id_inferencia]

    print("[Despliegue] Despliegues creados exitosamente en Prefect:")
    print(f"  - Entrenamiento: {id_entrenamiento} ('{nombre_entrenamiento}')")
    print(f"  - Inferencia:    {id_inferencia} ('{nombre_inferencia}') [Horario: {cron_expr}]")
    print("Verifique ambos en la interfaz de Prefect: http://localhost:4200/deployments")
    return ids


if __name__ == "__main__":
    crear_despliegues()
