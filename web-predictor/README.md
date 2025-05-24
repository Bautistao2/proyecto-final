# Enneagram Predictor Web Application

Una aplicación web moderna para la predicción de eneagramas basada en un modelo de machine learning.

## Descripción

Este proyecto consiste en una aplicación web para realizar un test de eneagrama y obtener una predicción del tipo de personalidad según este sistema. Incluye:

- **Backend**: Una API REST construida con FastAPI que expone endpoints para obtener preguntas y realizar predicciones usando un modelo de machine learning.
- **Frontend**: Una aplicación React moderna construida con Chakra UI para una experiencia de usuario intuitiva.

## Estructura del Proyecto

```
web-predictor/
├── backend/               # API FastAPI
│   ├── src/               # Código fuente del backend
│   │   ├── api/           # Definición de rutas y endpoints
│   │   ├── core/          # Configuraciones y dependencias
│   │   ├── data/          # Datos estáticos (preguntas)
│   │   ├── models/        # Modelos ML pre-entrenados
│   │   ├── schemas/       # Definición de esquemas Pydantic
│   │   ├── services/      # Servicios (modelo de predicción)
│   │   └── main.py        # Punto de entrada de la aplicación
│   └── requirements.txt   # Dependencias Python
├── frontend/              # Aplicación React
│   └── src/               # Código fuente del frontend
│       ├── components/    # Componentes reutilizables
│       ├── pages/         # Páginas principales
│       ├── services/      # Servicios (API)
│       └── theme.ts       # Configuración del tema
└── start_app.ps1          # Script para iniciar la aplicación
```

## Requisitos

- Python 3.8+
- Node.js 14+
- npm 6+

## Instalación

### Backend

1. Navega a la carpeta del backend:
   ```
   cd backend
   ```

2. Crea un entorno virtual (opcional pero recomendado):
   ```
   python -m venv env
   env\Scripts\activate
   ```

3. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

### Frontend

1. Navega a la carpeta del frontend:
   ```
   cd frontend
   ```

2. Instala las dependencias:
   ```
   npm install
   ```

## Ejecución

### Método 1: Script de inicio (recomendado)

Ejecuta el script `start_app.ps1` desde PowerShell para iniciar tanto el backend como el frontend:

```
.\start_app.ps1
```

### Método 2: Iniciar servicios por separado

#### Backend

```
cd backend
python -m uvicorn src.main:app --reload --port 8000
```

#### Frontend

```
cd frontend
npm start
```

## Uso de la aplicación

1. Abre tu navegador y navega a `http://localhost:3000`.
2. En la página principal, haz clic en "Comenzar Test".
3. Responde todas las preguntas del test (80 preguntas en total).
4. Envía tus respuestas y obtén tu tipo de eneagrama y ala.

## API Documentation

La documentación de la API está disponible en `http://localhost:8000/docs` una vez que el backend está en ejecución.

## Implementación en producción

Para implementar en producción, se recomienda:

1. Configurar un servidor WSGI como Gunicorn para el backend
2. Crear una build estática del frontend con `npm run build`
3. Configurar un servidor web como Nginx para servir los archivos estáticos y hacer proxy al backend

## Licencia

Este proyecto está licenciado bajo MIT License.
