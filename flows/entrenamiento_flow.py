"""Pipeline de entrenamiento en Prefect para predecir precio de venta de vehículos usados.

Pasos obligatorios de la actividad:
  ingesta -> entrenamiento -> evaluación -> registro

Requisitos cubiertos:
  - Cada ejecución crea un run en MLflow con el tag fase = entrenamiento.
  - Registra el alias de selección de variables ('features') y el número de variables ('n_features').
  - Compara el modelo nuevo con el champion actual en MLflow Model Registry.
  - Solo promueve a 'champion' si supera el umbral mínimo y supera al champion vigente.
  - En cada momento hay un único champion. Si no mejora, el run queda registrado pero el alias no cambia.

Ejecución:
  $env:PREFECT_API_URL = "http://localhost:4200/api"
  $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
  $env:PYTHONIOENCODING = "utf-8"
  $env:PYTHONUTF8 = "1"
  poetry run python flows/entrenamiento_flow.py
"""
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Permite importar src/

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

from src.champion import asignar_alias_champion, evaluar_candidato_a_champion  # noqa: E402
from src.config import RUTA_CONFIG, RUTA_FEATURES, cargar_config, ruta_absoluta  # noqa: E402
from src.features import obtener_features  # noqa: E402
from src.load_data import cargar_datos, huella_archivo  # noqa: E402
from src.train import dividir_datos, entrenar_modelo, evaluar_modelo  # noqa: E402


# -----------------------------------------------------------------------------
# 1. PASO DE INGESTA
# -----------------------------------------------------------------------------
@task(name="ingesta", cache_policy=NO_CACHE)
def tarea_ingesta(config_datos: dict[str, Any]) -> pd.DataFrame:
    """Lee el CSV del dataset, valida columnas y genera estadísticas iniciales."""
    logger = get_run_logger()
    df = cargar_datos(config_datos)
    ruta_datos = ruta_absoluta(config_datos["ruta"])
    md5 = huella_archivo(ruta_datos)
    logger.info(f"[Ingesta] Dataset cargado: {len(df):,} filas x {df.shape[1]} columnas")
    logger.info(f"[Ingesta] Archivo: {config_datos['ruta']} | MD5: {md5}")
    logger.info(f"[Ingesta] Rango fechas: {df['fecha_venta'].min().date()} a {df['fecha_venta'].max().date()}")
    return df


# -----------------------------------------------------------------------------
# 2. PASO DE ENTRENAMIENTO
# -----------------------------------------------------------------------------
@task(name="entrenamiento", cache_policy=NO_CACHE)
def tarea_entrenamiento(
    df: pd.DataFrame,
    config_modelo: dict[str, Any],
    features_info: dict[str, Any],
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Divide los datos según features.yaml y entrena el pipeline de scikit-learn."""
    logger = get_run_logger()
    logger.info(
        f"[Entrenamiento] Selección de variables: '{features_info['alias']}' "
        f"({features_info['n_features']} variables: {features_info['todas']})"
    )

    X_train, X_test, y_train, y_test = dividir_datos(df, config_modelo, features_info)
    logger.info(f"[Entrenamiento] Train size: {len(X_train):,}, Test size: {len(X_test):,}")

    pipeline = entrenar_modelo(X_train, y_train, config_modelo, features_info)
    logger.info(f"[Entrenamiento] Pipeline entrenado con modelo tipo: '{config_modelo['tipo']}'")

    return pipeline, X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# 3. PASO DE EVALUACIÓN
# -----------------------------------------------------------------------------
@task(name="evaluacion", cache_policy=NO_CACHE)
def tarea_evaluacion(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evalúa el pipeline sobre el conjunto de test y calcula métricas."""
    logger = get_run_logger()
    metricas = evaluar_modelo(pipeline, X_test, y_test)
    logger.info(
        f"[Evaluación] Métricas obtenidas: "
        f"R2={metricas['r2']:.4f} | RMSE=${metricas['rmse']:,.2f} | "
        f"MAE=${metricas['mae']:,.2f} | MAPE={metricas['mape']:.2%}"
    )
    return metricas


# -----------------------------------------------------------------------------
# 4. PASO DE REGISTRO Y COMPARACIÓN CON EL CHAMPION
# -----------------------------------------------------------------------------
@task(name="registro", cache_policy=NO_CACHE)
def tarea_registro(
    pipeline: Pipeline,
    metricas: dict[str, float],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    df: pd.DataFrame,
    config: dict[str, Any],
    features_info: dict[str, Any],
) -> dict[str, Any]:
    """Registra en MLflow con tag fase=entrenamiento y gestiona el alias champion."""
    logger = get_run_logger()
    config_datos = config["datos"]
    config_modelo = config["modelo"]
    config_eval = config.get("evaluacion", {})
    config_mlflow = config["mlflow"]

    nombre_experimento = config_mlflow.get("experimento", "precio-vehiculos")
    nombre_modelo = config_mlflow.get("nombre_modelo", "precio-vehiculos-model")
    alias_champion = config_mlflow.get("alias_champion", "champion")

    mlflow.set_experiment(nombre_experimento)
    with mlflow.start_run(run_name=config_mlflow.get("nombre_run", "rf-entrenamiento")) as run:
        run_id = run.info.run_id

        # Tags obligatorios de la actividad
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.set_tag("features", features_info["alias"])
        mlflow.set_tag("dataset_md5", huella_archivo(ruta_absoluta(config_datos["ruta"])))

        # Parámetros requeridos por la consigna
        mlflow.log_params(
            {
                "features": features_info["alias"],
                "n_features": features_info["n_features"],
                "variables_lista": ",".join(features_info["todas"]),
                "n_filas": len(df),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "test_size": config_modelo["test_size"],
                "tipo_modelo": config_modelo["tipo"],
                **{f"hp_{k}": v for k, v in config_modelo.get("hiperparametros", {}).items()},
            }
        )

        # Registro de métricas calculadas en el paso de evaluación
        mlflow.log_metrics(metricas)

        # Trazabilidad de datos de entrada y configs
        mlflow.log_input(
            mlflow.data.from_pandas(
                df,
                source=str(config_datos["ruta"]),
                name="car_sales_dataset",
                targets=config_modelo["objetivo"],
            ),
            context="training",
        )
        if RUTA_CONFIG.exists():
            mlflow.log_artifact(str(RUTA_CONFIG), artifact_path="config")
        if RUTA_FEATURES.exists():
            mlflow.log_artifact(str(RUTA_FEATURES), artifact_path="config")

        # Registro del modelo serializado
        ejemplo = X_test.head(5)
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="modelo",
            signature=infer_signature(ejemplo, pipeline.predict(ejemplo)),
            input_example=ejemplo,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        # Registrar en el MLflow Model Registry
        client = MlflowClient()
        reg_version = mlflow.register_model(model_uri=model_info.model_uri, name=nombre_modelo)
        version_str = str(reg_version.version)

        # Comparar con el Champion actual
        decision = evaluar_candidato_a_champion(
            client=client,
            nombre_modelo=nombre_modelo,
            metricas=metricas,
            config_evaluacion=config_eval,
            alias_champion=alias_champion,
        )

        if decision["es_champion"]:
            asignar_alias_champion(client, nombre_modelo, version_str, alias_champion)
            mlflow.set_tag("promovido_a_champion", "true")
            client.set_model_version_tag(nombre_modelo, version_str, "alias", alias_champion)
            client.set_model_version_tag(nombre_modelo, version_str, "decision", "promovido")
            logger.info(
                f"[Champion] ¡NUEVO CHAMPION ASIGNADO! Versión {version_str} -> alias '{alias_champion}'. "
                f"Motivo: {decision['motivo']}"
            )
        else:
            mlflow.set_tag("promovido_a_champion", "false")
            client.set_model_version_tag(nombre_modelo, version_str, "decision", "no_promovido")
            logger.info(
                f"[Champion] Modelo registrado como versión {version_str}, pero NO reemplaza al champion actual. "
                f"Motivo: {decision['motivo']}"
            )

        # Crear artefacto Markdown con el resumen de la ejecución
        resumen_md = f"""# Resumen de Pipeline de Entrenamiento

**Estado Champion:** {'PROMOVIDO A CHAMPION' if decision['es_champion'] else 'NO PROMOVIDO (CHAMPION VIGENTE SE MANTIENE)'}

### Configuración del Experimento
- **Experimento MLflow:** `{nombre_experimento}`
- **Run ID:** `{run_id}`
- **Fase:** `entrenamiento`
- **Selección de Variables (`features`):** `{features_info['alias']}` ({features_info['n_features']} variables)
- **Modelo:** `{config_modelo['tipo']}`
- **Versión en Registry:** `v{version_str}`

### Desempeño
| Métrica | Valor |
|---|---|
| R² | {metricas['r2']:.4f} |
| RMSE | ${metricas['rmse']:,.2f} USD |
| MAE | ${metricas['mae']:,.2f} USD |
| MAPE | {metricas['mape']:.2%} |

### Decisión de Promoción
- **Métrica Evaluada:** `{decision['metrica_nombre']}` = {decision['valor_candidato']:.4f}
- **Champion Anterior:** v{decision.get('version_champion_previa') or 'Ninguno'} ({decision.get('valor_champion_previo') if decision.get('valor_champion_previo') is not None else 'N/A'})
- **Detalle:** {decision['motivo']}
"""
        create_markdown_artifact(
            key="resumen-entrenamiento",
            markdown=resumen_md,
            description="Resumen de ejecución y evaluación champion del modelo de precio de venta",
        )

        return {
            "run_id": run_id,
            "version": version_str,
            "es_champion": decision["es_champion"],
            "decision": decision,
            "metricas": metricas,
        }


# -----------------------------------------------------------------------------
# FLOW PRINCIPAL DE PREFECT
# -----------------------------------------------------------------------------
@flow(name="entrenamiento-precio-vehiculos", log_prints=True)
def entrenamiento_flow() -> dict[str, Any]:
    """Pipeline completo de entrenamiento con pasos ingesta -> entrenamiento -> evaluación -> registro."""
    config = cargar_config()
    features_info = obtener_features()

    # 1. Ingesta
    df = tarea_ingesta(config["datos"])

    # 2. Entrenamiento
    pipeline, X_train, X_test, y_train, y_test = tarea_entrenamiento(df, config["modelo"], features_info)

    # 3. Evaluación
    metricas = tarea_evaluacion(pipeline, X_test, y_test)

    # 4. Registro y comparación Champion
    resultado = tarea_registro(pipeline, metricas, X_train, X_test, df, config, features_info)

    print(
        f"Pipeline finalizado con éxito. Versión {resultado['version']} | "
        f"Champion: {resultado['es_champion']} | R² = {metricas['r2']:.4f}"
    )
    return resultado


if __name__ == "__main__":
    entrenamiento_flow()
