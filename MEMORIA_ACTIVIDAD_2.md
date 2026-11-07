# MEMORIA ACTIVIDAD 2 – MLOps
## Pipeline y empaquetado

**Maestría en Ciencia de Datos – Universidad Santo Tomás**  
**Asignatura:** MLOps  
**Caso de Uso:** Predicción del precio de venta final (`precio_venta`, USD) de vehículos usados multimarca (Problema de Regresión Supervisada).  
**Integrantes:**
- Jhosman Ricardo Moreno Meauri
- Brayan Steven Vera Moreno

---

## 1. Resumen Actividad y Arquitectura

En esta Actividad 2 se implementó una arquitectura de MLOps para la orquestación, evaluación, registro y servicio del modelo de predicción de precios de vehículos usados realizando:

1. **Prefect 3.8.8:** Orquestación de dos pipelines independientes:
   - **Pipeline de Entrenamiento:** Flujo de 4 pasos modulares (`ingesta` → `entrenamiento` → `evaluación` → `registro`).
   - **Pipeline de Inferencia:** Flujo por lotes programado para ejecutarse diariamente (cron `0 2 * * *`).
2. **MLflow 3.17.0:**
   - *Experiment Tracking:* Registro detallado de runs con tags `fase = entrenamiento` y `fase = inferencia`, parámetros de selección de variables (`features`, `n_features`), métricas de regresión ($R^2$, RMSE, MAE, MAPE), artefactos y firmas de entrada/salida.
   - *Model Registry & Champion Logic:* Comparación del nuevo modelo con el modelo vigente. Solo se promueve y asigna el alias `champion` si supera el umbral mínimo configurado ($R^2 \ge 0.85$) y si mejora al champion vigente por al menos un margen delta. Si no lo supera, el run queda registrado pero el alias `champion` no cambia.
3. **FastAPI & Uvicorn:**
   - Servicio REST en `services/api/main.py` y `src/api.py` que consume el modelo resuelto dinámicamente mediante el alias `champion` (`models:/precio-vehiculos-model@champion`).
   - Comprobación: al actualizar el champion en MLflow y reiniciar la API, ésta sirve la nueva versión del modelo sin modificar líneas de código.
4. **Configuración en YAML:**
   - `configs/config.yaml`: Centralización de hiperparámetros, umbrales de decisión, nombres de experimento, rutas y parámetros de despliegue.
   - `configs/features.yaml`: Selección dinámica del subconjunto de variables (`ALL`, `NUMERIC`, `CUSTOM`).

```text
PROYECTO_DE_EQUIPO/
├── configs/
│   ├── config.yaml            # Parámetros desacoplados (modelo, umbrales, MLflow, inferencia)
│   └── features.yaml          # Selección de características (ALL, NUMERIC, CUSTOM)
├── data/
│   ├── car_sales_dataset.csv  # Dataset base (20,000 registros sintéticos reproducibles)
│   └── predicciones/          # Salida de inferencia: predicciones_YYYYMMDD_HHMMSS.csv
├── flows/
│   ├── entrenamiento_flow.py  # Flow Prefect: ingesta -> entrenamiento -> evaluación -> registro
│   ├── inferencia_flow.py     # Flow Prefect: inferencia diaria por lotes con champion
│   ├── desplegar.py           # Creación y registro de despliegues en Prefect
│   └── prueba_flow.py         # Verificación inicial de conectividad
├── services/
│   ├── api/
│   │   └── main.py            # API REST FastAPI para servir el modelo champion
│   ├── docker-compose.yml     # Contenedores de MLflow (5000) y Prefect (4200)
│   └── README.md
├── src/
│   ├── api.py                 # Lógica de la API y esquemas Pydantic
│   ├── champion.py            # Lógica de comparación y promoción del modelo Champion
│   ├── config.py              # Cargador de configuraciones YAML
│   ├── features.py            # Resolución dinámica de variables según alias
│   ├── generate_data.py       # Generador de datos reproducibles
│   ├── load_data.py           # Ingesta y validación de columnas
│   └── train.py               # Preprocesado (ColumnTransformer), entrenamiento y métricas
├── tests/
│   ├── test_api.py            # Pruebas de endpoints FastAPI (/health, /model-info, /predict)
│   ├── test_champion.py       # Pruebas de reglas de decisión de champion
│   ├── test_datos.py          # Pruebas de reproducibilidad y datos
│   ├── test_features.py       # Pruebas de alias ALL, NUMERIC y CUSTOM
│   └── test_train.py          # Pruebas del pipeline de entrenamiento
└── scripts/
    └── verificar_api.py       # Script cliente para probar la API contra el modelo champion
```

---

## 2. Entregables de la actividad

### Entregable 1: Pipeline de Entrenamiento en Prefect (`flows/entrenamiento_flow.py`)
El pipeline implementa de forma explícita los 4 pasos solicitados:

1. **`ingesta` (`tarea_ingesta`):**
   - Carga el archivo `data/car_sales_dataset.csv` validando que contenga todas las columnas requeridas (`fecha_venta`, `marca`, `tipo_vehiculo`, `año_modelo`, `kilometraje`, `precio_lista`, `precio_venta`).
   - Calcula y registra el hash MD5 del dataset para trazabilidad absoluta.
2. **`entrenamiento` (`tarea_entrenamiento`):**
   - Resuelve el conjunto de variables según `configs/features.yaml` (`ALL`, `NUMERIC` o `CUSTOM`).
   - Divide los datos en 80% train / 20% test con semilla fija.
   - Construye un `ColumnTransformer` que aplica `OneHotEncoder` a las categóricas y `StandardScaler` a las numéricas.
   - Ajusta el modelo (`RandomForestRegressor` o `LinearRegression`).
3. **`evaluacion` (`tarea_evaluacion`):**
   - Genera predicciones sobre el conjunto de test y calcula métricas clave:
     - $R^2$ (coeficiente de determinación): métrica principal de negocio.
     - RMSE (Root Mean Squared Error en USD).
     - MAE (Mean Absolute Error en USD).
     - MAPE (Mean Absolute Percentage Error).
4. **`registro` (`tarea_registro`):**
   - Inicia un run en MLflow con tag obligatorio `fase: entrenamiento`.
   - Registra tags adicionales: `features` (alias usado), `dataset_md5` y `promovido_a_champion` (`true`/`false`).
   - Registra parámetros: `features`, `n_features`, `variables_lista`, tamaño del dataset, tamaño de particiones e hiperparámetros.
   - Registra métricas de evaluación.
   - Loguea artefactos de configuración (`config.yaml` y `features.yaml`) y el dataset de entrada.
   - Registra el modelo en el Model Registry de MLflow (`precio-vehiculos-model`).
   - Evalúa contra el champion actual mediante `src/champion.py`. Si califica, asigna el alias `champion`.
   - Publica en Prefect el artefacto visual Markdown `resumen-entrenamiento`.

---

### Entregable 2: Pipeline de Inferencia en Prefect (`flows/inferencia_flow.py`)
El pipeline de inferencia por lotes:

1. **Carga por alias `champion`:** Utiliza `models:/precio-vehiculos-model@champion`, sin acoplarse a un número de versión específico.
2. **Obtención del lote:** Simula un lote de clientes entrantes (`id_cliente: CLI-1000...`) de tamaño configurable (`tamano_lote: 50`) descartando la variable objetivo (`precio_venta`) para simular condiciones reales de producción.
3. **Cálculo de predicciones:** Ejecuta el pipeline del modelo sobre el lote y calcula el precio predicho en USD.
4. **Guardado con fecha:** Escribe el resultado en `data/predicciones/predicciones_YYYYMMDD_HHMMSS.csv` incluyendo metadatos (`id_cliente`, variables del vehículo, `precio_predicho_usd`, `modelo_version`, `modelo_alias`, `fecha_inferencia`).
5. **Registro en MLflow:**
   - Inicia run con tag `fase: inferencia`, `modelo_alias: champion`, `modelo_version: <version>`.
   - Registra parámetros del lote y métricas agregadas (precio promedio, mínimo y máximo predicho).
   - Sube el CSV generado como artefacto de MLflow.
   - Publica en Prefect el artefacto visual `resumen-inferencia`.

---

### Entregable 3: Conexión mediante MLflow y lógica de Champion

La lógica está encapsulada en `src/champion.py`:
- **Regla 1 (Umbral mínimo):** El modelo candidato debe alcanzar un valor mínimo de la métrica configurada ($R^2 \ge 0.85$ por defecto). Si no lo alcanza, es descartado para champion.
- **Regla 2 (Primer modelo):** Si en MLflow Model Registry no existe un modelo previo con alias `champion`, el primer candidato que cumpla la Regla 1 recibe automáticamente el alias.
- **Regla 3 (Competencia con Champion vigente):** Si ya existe un champion, se consulta su run de entrenamiento en MLflow para extraer su desempeño. El nuevo modelo **solo lo reemplaza** si:

$$
R^2_{\text{nuevo}} - R^2_{\text{champion}} \ge \text{mejora}
$$

  Si no supera al champion, el run y la nueva versión quedan registrados en MLflow para auditoría, pero el alias `champion` **permanece intacto** en la versión anterior.
- **Unicidad:** En MLflow, los alias asignados a un modelo registrado son únicos; asignar `champion` a una nueva versión desasigna automáticamente la versión anterior.

#### Evidencia de comparación de versiones registrada en pruebas:
| Versión | Configuración | $R^2$ Obtenido | Decisión Champion | Motivo Registrado en MLflow / Prefect |
|---|---|---|---|---|
| **v1** | RF (`n_estimators: 150, max_depth: 12`) | **0.9938** | **Promovido a Champion** | Primer modelo que supera el umbral mínimo ($0.9938 \ge 0.8500$). |
| **v2** | RF (`n_estimators: 250, max_depth: 14`) | **0.9936** | **NO Promovido** | No supera al champion v1 ($0.9936 < 0.9938$, mejora $-0.0002 < 0.0000$). Champion sigue siendo v1. |
| **v3** | RF (`n_estimators: 200, max_depth: 20`) | **0.9933** | **NO Promovido** | No supera al champion v1 ($0.9933 < 0.9938$, mejora $-0.0004 < 0.0000$). Champion sigue siendo v1. |
| **v4** | RF (`n_estimators: 220, max_depth: 11`) | **0.9939** | **Promovido a Champion** | Supera al champion v1 ($0.9939 > 0.9938$, mejora $+0.0001 \ge 0.00001$). Alias `champion` apunta ahora a v4. |

---

### Entregable 4: API de inferencia con FastAPI (`services/api/main.py`)

La API sirve el modelo cargándolo directamente mediante su alias `champion`.

#### Endpoints implementados:
- `GET /`: Bienvenida y estado general.
- `GET /health`: Monitoreo del estado (`healthy`) y versión activa del champion.
- `GET /model-info`: Metadatos del modelo champion cargado en memoria (`run_id`, versión, estado).
- `POST /predict`: Inferencia para un cliente individual con validación mediante Pydantic (`DatosVehiculo`).
- `POST /predict/batch`: Inferencia para una lista de clientes en lote.

#### Ejemplo de predicción:
**Solicitud (`POST /predict`):**
```json
{
  "marca": "Toyota",
  "tipo_vehiculo": "SUV",
  "año_modelo": 2022,
  "kilometraje": 35000.0,
  "precio_lista": 28500.0
}
```

**Respuesta Obtenida:**
```json
{
  "precio_predicho_usd": 27310.84,
  "modelo_version": "4",
  "modelo_alias": "champion",
  "timestamp": "2026-11-07T16:44:06.411854"
}
```

#### Comprobación del funcionamiento:
1. Con el champion en **v1**, la API servía predicciones con `"modelo_version": "1"`.
2. Se entrenó la versión **v4** que superó a v1, convirtiéndose en el nuevo `champion`.
3. Al reiniciar la API (`scripts/verificar_api.py` o recarga de Uvicorn), la API comenzó inmediatamente a responder con `"modelo_version": "4"` sin tocar ningún archivo ni código fuente.

---

### Entregable 5: Configuración separada en YAML

#### `configs/features.yaml`:
Permite configurar el subconjunto de variables sin alterar el código:
```yaml
seleccion: ALL   # Opciones: ALL | NUMERIC | CUSTOM

todas:
  categoricas:
    - marca
    - tipo_vehiculo
  numericas:
    - año_modelo
    - kilometraje
    - precio_lista

custom:
  categoricas:
    - marca
  numericas:
    - año_modelo
    - precio_lista
```
- Si `seleccion: ALL`: usa todas las 5 variables (`n_features: 5`).
- Si `seleccion: NUMERIC`: excluye categóricas y usa solo las 3 variables numéricas (`n_features: 3`).
- Si `seleccion: CUSTOM`: usa el subconjunto definido en `custom`.
En cada run de MLflow se registran los parámetros `features` y `n_features` para comparar el impacto en las métricas.

#### `configs/config.yaml`:
Centraliza todos los parámetros que pueden variar en el ciclo de vida:
```yaml
datos:
  ruta: data/car_sales_dataset.csv
  num_registros: 20000
  semilla: 42
  columnas_esperadas:
    - fecha_venta
    - marca
    - tipo_vehiculo
    - año_modelo
    - kilometraje
    - precio_lista
    - precio_venta

modelo:
  objetivo: precio_venta
  tipo: random_forest
  hiperparametros:
    n_estimators: 220
    max_depth: 11
    random_state: 42
  test_size: 0.2
  semilla_split: 42

evaluacion:
  metrica_champion: r2
  umbral_minimo: 0.85
  mejora_minima: 0.00001
  modo: max

mlflow:
  experimento: precio-vehiculos
  nombre_modelo: precio-vehiculos-model
  alias_champion: champion
  nombre_run: rf-tuning-v4-nuevo-champion

inferencia:
  tamano_lote: 50
  ruta_salida: data/predicciones
  prefijo_archivo: predicciones

api:
  host: 0.0.0.0
  port: 8000
  titulo: API de Inferencia - Precio de Vehículos Usados
  version: 0.2.0

despliegues:
  cron_inferencia: "0 2 * * *"
  nombre_entrenamiento: entrenamiento-despliegue
  nombre_inferencia: inferencia-diaria-despliegue
```

---

### Entregable 6: Despliegues en Prefect (`flows/desplegar.py`)

El script registra los dos despliegues independientes en el servidor de Prefect:
- **Despliegue de Entrenamiento (`entrenamiento-despliegue`):** Lanzable a mano o bajo demanda.
- **Despliegue de Inferencia (`inferencia-diaria-despliegue`):** Programado con horario cron diario (`0 2 * * *` = todos los días a las 02:00 AM).

Los despliegues quedan visibles en la UI de prefect (`http://localhost:4200/deployments`):
- Deployment ID Entrenamiento: `df299c04-4939-427c-a55e-f0c6ca617f4b`
- Deployment ID Inferencia: `8217d7ee-1d37-4cf0-a88a-5bf5cd2a64a5`

---

## 3. Ejecutar pasos de la actividad para replicar todo en local


### Paso 1: Configurar el entorno y dependencias
Desde la raíz del repositorio:

```powershell
# 1. Asegurar configuración local de Poetry para usar virtualenv interno del proyecto
poetry config virtualenvs.in-project true --local
poetry config keyring.enabled false --local

# 2. Instalar todas las dependencias bloqueadas
poetry install

# 3. Comprobar que los paquetes esenciales están instalados correctamente
poetry run python -c "import prefect, mlflow, sklearn, pandas, numpy, yaml, fastapi, uvicorn; print('Todos los paquetes OK')"
```

### Paso 2: Ejecutar pruebas unitarias
Ejecutar pytest para verificar que los componentes (datos, entrenamiento, features, champion y API) funcionen sin errores:

```powershell
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
poetry run pytest
```
*Resultado esperado:* **23 passed**.

### Paso 3: Levantar los servicios (MLflow y Prefect)
En una terminal con Docker Desktop abierto:

```powershell
cd services
docker compose up -d
docker compose ps
cd ..
```
*Comprobación de URLs:*
- MLflow Tracking & Registry: [http://localhost:5000](http://localhost:5000)
- Prefect UI & API: [http://localhost:4200](http://localhost:4200)

### Paso 4: Definir variables de entorno en cada terminal de trabajo

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
```

> **Nota: ** Las variables `$env:PYTHONIOENCODING = "utf-8"` y `$env:PYTHONUTF8 = "1"` evitan el error `UnicodeEncodeError: 'charmap'`.

### Paso 5: Validar conectividad con el flow de prueba
```powershell
poetry run python flows/prueba_flow.py
```
*Resultado esperado:* Prefect registra el run en estado `Completed` y MLflow registra el experimento `prueba-validacion`.

### Paso 6: Ejecutar el pipeline de entrenamiento
```powershell
poetry run python flows/entrenamiento_flow.py
```
*Comprobación:*
- En la terminal: Muestra logs de los 4 pasos (`ingesta`, `entrenamiento`, `evaluacion`, `registro`).
- En MLflow ([http://localhost:5000](http://localhost:5000)):
  - Experimento `precio-vehiculos` con run tag `fase = entrenamiento`.
  - En la pestaña **Models**: Modelo registrado `precio-vehiculos-model` con versión 1 y alias **`champion`**.
- En Prefect ([http://localhost:4200](http://localhost:4200)):
  - Run del flow `entrenamiento-precio-vehiculos` en estado `Completed`.
  - Pestaña **Artifacts**: Artefacto Markdown `resumen-entrenamiento` con tabla de métricas y decisión.

### Paso 7: Ejecutar el Pipeline de Inferencia
```powershell
poetry run python flows/inferencia_flow.py
```
*Comprobación:*
- Carga el modelo mediante `models:/precio-vehiculos-model@champion`.
- Genera el archivo CSV con fecha en `data/predicciones/predicciones_YYYYMMDD_HHMMSS.csv`.
- En MLflow: Aparece un run en `precio-vehiculos` con tag `fase = inferencia` y el CSV como artefacto.
- En Prefect: Aparece el run en estado `Completed` y el artefacto visual `resumen-inferencia`.

### Paso 8: Desplegar los flows en prefect
```powershell
poetry run python flows/desplegar.py
```
*Comprobación:*
- Entrar a [http://localhost:4200/deployments](http://localhost:4200/deployments).
- aparecen dos despliegues:
  1. `entrenamiento-precio-vehiculos/entrenamiento-despliegue` (Manual).
  2. `inferencia-precio-vehiculos/inferencia-diaria-despliegue` (Horario: `0 2 * * *`).

### Paso 9: Iniciar y probar la API de inferencia (FastAPI)

#### Iniciar el servidor de la API:
En una terminal con las variables de entorno configuradas:
```powershell
poetry run uvicorn services.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Probar la API desde otra terminal:
```powershell
poetry run python scripts/verificar_api.py
```

O directamente mediante `curl`:
```powershell
curl -X POST "http://localhost:8000/predict" `
     -H "Content-Type: application/json" `
     -d '{"marca": "Toyota", "tipo_vehiculo": "SUV", "año_modelo": 2022, "kilometraje": 35000.0, "precio_lista": 28500.0}'
```
*Respuesta esperada:* Código 200 con el precio estimado y metadatos del champion (`modelo_version`, `modelo_alias`).

### Paso 10: Comprobar el funcionamiento (Actualización del champion y reinicio de la API)
1. Modificar un hiperparámetro en `configs/config.yaml` (por ejemplo, cambiar `n_estimators` o `max_depth`) o cambiar `seleccion` en `configs/features.yaml`.
2. Volver a ejecutar el flow de entrenamiento:
   ```powershell
   poetry run python flows/entrenamiento_flow.py
   ```
3. Si el nuevo modelo supera al champion anterior, el alias `champion` se reasigna a la nueva versión en MLflow Model Registry.
4. Reiniciar la API (o dejar que `--reload` recargue).
5. Volver a consultar `GET /health` o `POST /predict`. El servicio responde con la nueva versión del modelo de manera transparente sin tocar ningún archivo de código.

---

