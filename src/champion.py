"""Lógica de evaluación, comparación y promoción del modelo Champion en MLflow Model Registry."""
from typing import Any

from mlflow.tracking import MlflowClient


def evaluar_candidato_a_champion(
    client: MlflowClient,
    nombre_modelo: str,
    metricas: dict[str, float],
    config_evaluacion: dict[str, Any],
    alias_champion: str = "champion",
) -> dict[str, Any]:
    """Evalúa si un modelo candidato cumple los criterios para ser promovido a Champion.

    Reglas:
      1. Debe superar el umbral mínimo configurado (ej. R² >= 0.85).
      2. Si no hay un champion previo, pasa automáticamente a ser champion si cumple (1).
      3. Si ya existe un champion, el candidato debe mejorar la métrica del champion
         por al menos 'mejora_minima'.
    """
    metrica_nombre = config_evaluacion.get("metrica_champion", "r2")
    umbral_minimo = float(config_evaluacion.get("umbral_minimo", 0.85))
    mejora_minima = float(config_evaluacion.get("mejora_minima", 0.0001))
    modo = str(config_evaluacion.get("modo", "max")).strip().lower()

    if metrica_nombre not in metricas:
        raise KeyError(
            f"La métrica requerida '{metrica_nombre}' no fue encontrada en las métricas del modelo: {list(metricas)}"
        )

    valor_candidato = float(metricas[metrica_nombre])

    # 1. Validación de umbral mínimo
    pasa_umbral = (
        valor_candidato >= umbral_minimo if modo == "max" else valor_candidato <= umbral_minimo
    )
    if not pasa_umbral:
        return {
            "es_champion": False,
            "motivo": (
                f"No supera el umbral mínimo de {metrica_nombre} "
                f"(obtenido: {valor_candidato:.4f}, requerido: {umbral_minimo:.4f})"
            ),
            "metrica_nombre": metrica_nombre,
            "valor_candidato": valor_candidato,
            "valor_champion_previo": None,
            "version_champion_previa": None,
            "mejora": None,
        }

    # 2. Consultar champion actual
    champion_actual = None
    try:
        champion_actual = client.get_model_version_by_alias(nombre_modelo, alias_champion)
    except Exception:
        champion_actual = None

    if champion_actual is None:
        return {
            "es_champion": True,
            "motivo": (
                f"Primer modelo registrado que supera el umbral mínimo "
                f"({metrica_nombre}={valor_candidato:.4f} >= {umbral_minimo:.4f})"
            ),
            "metrica_nombre": metrica_nombre,
            "valor_candidato": valor_candidato,
            "valor_champion_previo": None,
            "version_champion_previa": None,
            "mejora": None,
        }

    # 3. Comparar con el champion vigente
    try:
        run_champion = client.get_run(champion_actual.run_id)
        valor_previo = float(run_champion.data.metrics.get(metrica_nombre, float("-inf") if modo == "max" else float("inf")))
    except Exception:
        valor_previo = None

    if valor_previo is None:
        return {
            "es_champion": True,
            "motivo": f"Champion actual v{champion_actual.version} no tiene métrica registrada; se promueve nuevo modelo.",
            "metrica_nombre": metrica_nombre,
            "valor_candidato": valor_candidato,
            "valor_champion_previo": None,
            "version_champion_previa": champion_actual.version,
            "mejora": None,
        }

    mejora = (valor_candidato - valor_previo) if modo == "max" else (valor_previo - valor_candidato)
    es_mejor = mejora >= mejora_minima

    if es_mejor:
        motivo = (
            f"Supera al champion v{champion_actual.version} en {metrica_nombre} "
            f"(nuevo: {valor_candidato:.4f} vs vigente: {valor_previo:.4f}, mejora: {mejora:+.4f} >= {mejora_minima:.4f})"
        )
    else:
        motivo = (
            f"No supera al champion v{champion_actual.version} en {metrica_nombre} "
            f"(nuevo: {valor_candidato:.4f} vs vigente: {valor_previo:.4f}, mejora: {mejora:+.4f} < {mejora_minima:.4f})"
        )

    return {
        "es_champion": es_mejor,
        "motivo": motivo,
        "metrica_nombre": metrica_nombre,
        "valor_candidato": valor_candidato,
        "valor_champion_previo": valor_previo,
        "version_champion_previa": champion_actual.version,
        "mejora": mejora,
    }


def asignar_alias_champion(
    client: MlflowClient,
    nombre_modelo: str,
    version: str,
    alias_champion: str = "champion",
) -> None:
    """Asigna el alias champion a la versión indicada en MLflow Model Registry."""
    client.set_registered_model_alias(
        name=nombre_modelo,
        alias=alias_champion,
        version=version,
    )
