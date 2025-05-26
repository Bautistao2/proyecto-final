# Proyecto de Predicción de Eneagrama

Este proyecto implementa un sistema de predicción de tipos de eneagrama y alas basado en aprendizaje automático, con una arquitectura de microservicios en Docker.

## Estructura del Proyecto

El proyecto está dividido en tres componentes principales:

```
├── model/                  # Entrenamiento y definición del modelo ML
├── web-predictor/          # Aplicación web (backend y frontend)
└── docker-compose.yml      # Configuración de orquestación de servicios
```

## Componentes Principales

### Modelo de Predicción

El componente `model/` contiene el código para entrenar y generar el modelo de predicción de eneagrama:

- **eneagrama_predictor/**: Módulo principal con la lógica del modelo
  - `optimized_enneagram_system.py`: Implementación del sistema de predicción
  - `data_preprocessing.py`: Preparación de datos para el entrenamiento
  - `data_augmentation.py`: Técnicas para aumentar y balancear el conjunto de datos
- **results/**: Carpeta donde se guardan los modelos entrenados (archivos .joblib)
- **dockerfile**: Configuración para construir la imagen Docker del modelo

### Aplicación Web (web-predictor)

La aplicación web consta de un backend y un frontend:

#### Backend (`web-predictor/backend/`)

API RESTful desarrollada con FastAPI que:
- Carga el modelo de predicción
- Expone endpoints para realizar predicciones
- Proporciona información sobre las preguntas del test

Archivos clave:
- `src/services/model_service.py`: Servicio para cargar y utilizar el modelo
- `src/api/routes/predictor.py`: Endpoints para realizar predicciones
- `src/core/settings.py`: Configuración de la aplicación
- `docker-entrypoint.sh`: Script que verifica el entorno Docker al inicio

#### Frontend (`web-predictor/frontend/`)

Interfaz de usuario desarrollada con React y TypeScript que:
- Presenta las preguntas del test al usuario
- Envía las respuestas al backend para procesamiento
- Muestra los resultados de la predicción de manera visual

## Configuración de Docker

El archivo `docker-compose.yml` define cuatro servicios:

1. **model**: Servicio para entrenar y generar modelos
   - Monta volúmenes para guardar los resultados

2. **web**: Aplicación web básica (no incluida en la descripción)

3. **backend**: API para servir predicciones
   - Expone el puerto 8000
   - Monta los volúmenes necesarios para acceder al modelo
   - Configurado para cargar cualquier modelo .joblib disponible

4. **frontend**: Interfaz de usuario
   - Expone el puerto 3000
   - Depende del backend para funcionar

## Funcionalidades Principales

### 1. Predicción de Eneagrama

El sistema puede predecir:
- El tipo principal de eneagrama (1-9)
- El ala del eneagrama
- Probabilidades para cada tipo y ala posible

### 2. Carga Dinámica de Modelos

- El backend puede cargar automáticamente cualquier modelo con extensión `.joblib`
- Prioriza el modelo más reciente por fecha de modificación
- Verifica múltiples ubicaciones para encontrar modelos disponibles

### 3. Diagnóstico y Monitoreo

- Script `environment_check.py` para verificar la configuración del entorno
- Mensajes de log detallados sobre la carga del modelo
- Verificación de dependencias y rutas al inicio

## Cómo Ejecutar el Proyecto

1. **Requisitos Previos**
   - Docker y Docker Compose instalados
   - Git para clonar el repositorio

2. **Iniciar la aplicación**
   ```bash
   docker-compose up
   ```

3. **Acceder a la aplicación**
   - Frontend: http://localhost:3000
   - API Backend: http://localhost:8000/api/v1
   - Documentación API: http://localhost:8000/docs

## Desarrollo y Extensión

### Entrenar un Nuevo Modelo

Para entrenar un nuevo modelo:

1. Modifica los parámetros en `model/eneagrama_predictor/main.py` si es necesario
2. Ejecuta el servicio de modelo:
   ```bash
   docker-compose run model python eneagrama_predictor/main.py
   ```
3. El nuevo modelo se guardará en `model/results/`

### Añadir Nuevas Características

- **Backend**: Extiende las rutas en `web-predictor/backend/src/api/routes/`
- **Frontend**: Añade nuevos componentes en `web-predictor/frontend/src/components/`

## Notas Técnicas

- El modelo utiliza un sistema optimizado de aprendizaje automático para predecir tipos de eneagrama
- La aplicación está diseñada para ser escalable y fácilmente desplegable mediante Docker
- El backend busca automáticamente cualquier modelo .joblib disponible, lo que facilita las actualizaciones

## Solución de Problemas

- Si hay problemas para cargar el modelo, verifica los logs del contenedor backend:
  ```bash
  docker-compose logs backend
  ```
- Para diagnosticar problemas con los volúmenes, revisa el reporte de entorno al inicio
