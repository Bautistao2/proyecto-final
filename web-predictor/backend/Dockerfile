FROM python:3.12-slim

WORKDIR /app

COPY requirements_app1.txt .
RUN pip install --no-cache-dir -r requirements_app1.txt

COPY src ./src
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Configurar variables de entorno para indicar que estamos en Docker
ENV DOCKER=true
ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 8000

# Usar el script de entrada para verificar modelos antes de iniciar
ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
