"""Pipeline de inferencia por lotes en Prefect para el modelo de precios de vehículos.

Requisitos de la actividad:
  - Carga el modelo por su alias 'champion' (no por número de versión fijo).
  - Toma un lote de datos y calcula predicciones (simulación de nuevos clientes o datos recientes).
  - Guarda las predicciones en un archivo CSV con la fecha y hora de ejecución.
  - Registra un run en MLflow con el tag fase = inferencia.

Ejecución:
  $env:PREFECT_API_URL = "http://localhost:4200/api"
  $env:MLFLOW_TRACKING_URI = "http://localhost:5000"
  $env:PYTHONIOENCODING = "utf-8"
  $env:PYTHONUTF8 = "1"
  poetry run python flows/inferencia_flow.py
"""
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Permite importar src/

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from prefect import flow, get_run_logger, task  # noqa: E402
from prefect.artifacts import create_markdown_artifact  # noqa: E402
from prefect.cache_policies import NO_CACHE  # noqa: E402

from src.config import cargar_config, ruta_absoluta  # noqa: E402
from src.load_data import cargar_datos  # noqa: E402


# -----------------------------------------------------------------------------
# 1. CARGA DEL MODELO CHAMPION DESDE MLFLOW
# -----------------------------------------------------------------------------
@task(name="cargar-modelo-champion", cache_policy=NO_CACHE)
def tarea_cargar_champion(nombre_modelo: str, alias_champion: str = "champion") -> tuple[Any, str, str]:
    """Carga el modelo registrado en MLflow utilizando su alias 'champion'."""
    logger = get_run_logger()
    client = MlflowClient()

    try:
        model_version_info = client.get_model_version_by_alias(nombre_modelo, alias_champion)
    except Exception as e:
        raise RuntimeError(
            f"No se encontró un modelo registrado con alias '{alias_champion}' en '{nombre_modelo}'. "
            f"Asegúrese de ejecutar primero el pipeline de entrenamiento. Detalle: {e}"
        ) from e

    version = str(model_version_info.version)
    run_id = str(model_version_info.run_id)
    modelo_uri = f"models:/{nombre_modelo}@{alias_champion}"

    logger.info(f"[Inferencia] Cargando modelo champion desde URI: {modelo_uri}")
    logger.info(f"[Inferencia] Versión del champion: v{version} (Run ID: {run_id})")

    # Carga usando el alias oficial
    modelo = mlflow.sklearn.load_model(modelo_uri)
    return modelo, version, run_id


# -----------------------------------------------------------------------------
# 2. PREPARACIÓN DEL LOTE DE ENTRADA
# -----------------------------------------------------------------------------
@task(name="obtener-lote-inferencia", cache_policy=NO_CACHE)
def tarea_obtener_lote(config: dict[str, Any]) -> pd.DataFrame:
    """Prepara o simula el lote de clientes nuevos para inferencia."""
    logger = get_run_logger()
    tamano_lote = int(config.get("inferencia", {}).get("tamano_lote", 50))

    df_base = cargar_datos(config["datos"])
    # Tomamos una muestra aleatoria para simular solicitudes recientes
    sample = df_base.sample(n=min(tamano_lote, len(df_base)), random_state=int(datetime.now().timestamp()) % 10000).copy()

    # Eliminamos el target si está presente para simular entorno de producción
    if "precio_venta" in sample.columns:
        sample = sample.drop(columns=["precio_venta"])

    # Agregamos identificador de cliente/solicitud
    sample["id_cliente"] = [f"CLI-{1000 + i}" for i in range(len(sample))]
    sample["fecha_solicitud"] = datetime.now().isoformat()

    logger.info(f"[Inferencia] Lote preparado con {len(sample)} registros para predicción.")
    return sample


# -----------------------------------------------------------------------------
# 3. PREDICCIÓN CON EL MODELO CHAMPION
# -----------------------------------------------------------------------------
@task(name="calcular-predicciones", cache_policy=NO_CACHE)
def tarea_calcular_predicciones(
    modelo: Any,
    lote: pd.DataFrame,
    version_champion: str,
    alias_champion: str,
) -> pd.DataFrame:
    """Ejecuta la inferencia sobre el lote y formatea los resultados."""
    logger = get_run_logger()

    # Excluimos columnas auxiliares que no son variables del modelo
    cols_a_excluir = {"id_cliente", "fecha_solicitud", "fecha_venta"}
    cols_features = [c for c in lote.columns if c not in cols_a_excluir]

    X = lote[cols_features]
    logger.info(f"[Inferencia] Variables suministradas al modelo: {list(X.columns)}")

    predicciones = modelo.predict(X)
    predicciones = np.round(predicciones, 2)

    df_resultado = lote.copy()
    df_resultado["precio_predicho_usd"] = predicciones
    df_resultado["modelo_version"] = version_champion
    df_resultado["modelo_alias"] = alias_champion
    df_resultado["fecha_inferencia"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    logger.info(
        f"[Inferencia] Predicciones calculadas: Min=${predicciones.min():,.2f}, "
        f"Media=${predicciones.mean():,.2f}, Max=${predicciones.max():,.2f}"
    )
    return df_resultado


# -----------------------------------------------------------------------------
# 4. GUARDADO DE RESULTADOS CON FECHA
# -----------------------------------------------------------------------------
@task(name="guardar-predicciones", cache_policy=NO_CACHE)
def tarea_guardar_predicciones(df_resultado: pd.DataFrame, config_inferencia: dict[str, Any]) -> Path:
    """Guarda el archivo CSV con las predicciones identificadas con la fecha de ejecución."""
    logger = get_run_logger()
    dir_salida = ruta_absoluta(config_inferencia.get("ruta_salida", "data/predicciones"))
    dir_salida.mkdir(parents=True, exist_ok=True)

    prefijo = config_inferencia.get("prefijo_archivo", "predicciones")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_archivo = f"{prefijo}_{timestamp_str}.csv"
    ruta_archivo = dir_salida / nombre_archivo

    df_resultado.to_csv(ruta_archivo, index=False, encoding="utf-8")
    logger.info(f"[Inferencia] Archivo de predicciones guardado en: {ruta_archivo}")
    return ruta_archivo


# -----------------------------------------------------------------------------
# 5. REGISTRO DEL RUN EN MLFLOW Y ARTEFACTO PREFECT
# -----------------------------------------------------------------------------
@task(name="registrar-run-inferencia", cache_policy=NO_CACHE)
def tarea_registrar_inferencia(
    ruta_archivo: Path,
    df_resultado: pd.DataFrame,
    version_champion: str,
    alias_champion: str,
    config: dict[str, Any],
) -> str:
    """Crea el run en MLflow con tag fase=inferencia y genera el artefacto en Prefect."""
    logger = get_run_logger()
    config_mlflow = config["mlflow"]
    experimento = config_mlflow.get("experimento", "precio-vehiculos")
    mlflow.set_experiment(experimento)

    timestamp_run = datetime.now().strftime("%Y%m%d-%H%M%S")
    with mlflow.start_run(run_name=f"inferencia-batch-{timestamp_run}") as run:
        run_id = run.info.run_id

        # Tags requeridos por la consigna
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_alias", alias_champion)
        mlflow.set_tag("modelo_version", version_champion)

        # Parámetros y métricas de la inferencia
        mlflow.log_params(
            {
                "tamano_lote": len(df_resultado),
                "archivo_generado": ruta_archivo.name,
                "modelo_cargado": f"{config_mlflow.get('nombre_modelo')}@{alias_champion}",
            }
        )
        precios = df_resultado["precio_predicho_usd"]
        mlflow.log_metrics(
            {
                "prediccion_precio_promedio": float(precios.mean()),
                "prediccion_precio_min": float(precios.min()),
                "prediccion_precio_max": float(precios.max()),
                "prediccion_precio_std": float(precios.std()),
            }
        )

        # Registro del archivo de predicciones como artefacto de MLflow
        mlflow.log_artifact(str(ruta_archivo), artifact_path="predicciones")

        logger.info(f"[Inferencia] Run de MLflow registrado: {run_id} con tag fase=inferencia")

        # Artefacto Markdown en Prefect
        muestra_tabla = df_resultado[["id_cliente", "marca", "año_modelo", "kilometraje", "precio_predicho_usd"]].head(5)
        filas_md = [
            "| ID Cliente | Marca | Año Modelo | Kilometraje | Precio Predicho (USD) |",
            "|---|---|---|---|---|",
        ]
        for _, fila in muestra_tabla.iterrows():
            filas_md.append(
                f"| {fila['id_cliente']} | {fila['marca']} | {int(fila['año_modelo'])} | "
                f"{fila['kilometraje']:,.0f} | ${fila['precio_predicho_usd']:,.2f} |"
            )
        tabla_md = "\n".join(filas_md)

        resumen_md = f"""# Resumen de Pipeline de Inferencia Diaria

- **Fase:** `inferencia`
- **Modelo Utilizado:** `{config_mlflow.get('nombre_modelo')}@{alias_champion}` (Versión `v{version_champion}`)
- **Total Registros Predichos:** {len(df_resultado):,}
- **Archivo Generado:** `{ruta_archivo.name}`
- **MLflow Run ID:** `{run_id}`
- **Fecha de Ejecución:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Estadísticas del Lote
- **Precio Promedio Predicho:** ${precios.mean():,.2f} USD
- **Precio Mínimo Predicho:** ${precios.min():,.2f} USD
- **Precio Máximo Predicho:** ${precios.max():,.2f} USD

### Muestra de Predicciones (Primeros 5 registros)
{tabla_md}
"""
        create_markdown_artifact(
            key="resumen-inferencia",
            markdown=resumen_md,
            description="Resumen de la corrida de inferencia por lotes con el modelo champion",
        )

        return run_id


# -----------------------------------------------------------------------------
# FLOW PRINCIPAL DE INFERENCIA
# -----------------------------------------------------------------------------
@flow(name="inferencia-precio-vehiculos", log_prints=True)
def inferencia_flow() -> dict[str, Any]:
    """Pipeline de inferencia por lotes: carga champion, predice lote, guarda CSV y registra en MLflow."""
    config = cargar_config()
    nombre_modelo = config["mlflow"].get("nombre_modelo", "precio-vehiculos-model")
    alias_champion = config["mlflow"].get("alias_champion", "champion")

    # 1. Cargar modelo champion
    modelo, version_champion, run_id_modelo = tarea_cargar_champion(nombre_modelo, alias_champion)

    # 2. Obtener lote
    lote = tarea_obtener_lote(config)

    # 3. Calcular predicciones
    df_resultado = tarea_calcular_predicciones(modelo, lote, version_champion, alias_champion)

    # 4. Guardar archivo con fecha
    ruta_archivo = tarea_guardar_predicciones(df_resultado, config.get("inferencia", {}))

    # 5. Registrar en MLflow
    run_id_inf = tarea_registrar_inferencia(ruta_archivo, df_resultado, version_champion, alias_champion, config)

    print(
        f"Inferencia completada con éxito. Archivo: {ruta_archivo.name} | "
        f"Modelo: v{version_champion} (@{alias_champion}) | Run MLflow: {run_id_inf}"
    )

    return {
        "ruta_archivo": str(ruta_archivo),
        "total_predicciones": len(df_resultado),
        "version_champion": version_champion,
        "run_id_inferencia": run_id_inf,
    }


if __name__ == "__main__":
    inferencia_flow()
