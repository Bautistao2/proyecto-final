FROM python:3.12-slim

WORKDIR /app

COPY requirements_app1.txt .
RUN pip install --no-cache-dir -r requirements_app1.txt

COPY src ./src
COPY docker-entrypoint.sh /usr/local/bin/

# Asegurar que el script tiene finales de línea Unix y permisos de ejecución
RUN sed -i 's/\r$//' /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# Configurar variables de entorno para indicar que estamos en Docker
ENV DOCKER=true
ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 8000

# Script para verificar modelos al inicio (en línea para evitar problemas de permisos)
RUN echo '#!/bin/bash\necho "📂 Verificando modelos disponibles..."\nresults_models=$(find /app/results -name "*.joblib" 2>/dev/null | wc -l)\nresults_alt_models=$(find /app/results-alt -name "*.joblib" 2>/dev/null | wc -l)\nmodel_results_models=$(find /app/model/results -name "*.joblib" 2>/dev/null | wc -l)\ntotal_models=$((results_models + results_alt_models + model_results_models))\necho "🔍 Modelos encontrados:"\nif [ $results_models -gt 0 ]; then\n    echo "  - /app/results: $results_models modelo(s)"\n    ls -l /app/results/*.joblib\nfi\nif [ $results_alt_models -gt 0 ]; then\n    echo "  - /app/results-alt: $results_alt_models modelo(s)"\n    ls -l /app/results-alt/*.joblib\nfi\nif [ $model_results_models -gt 0 ]; then\n    echo "  - /app/model/results: $model_results_models modelo(s)"\n    ls -l /app/model/results/*.joblib\nfi\nif [ $total_models -eq 0 ]; then\n    echo "⚠️ ADVERTENCIA: No se encontraron modelos .joblib en ninguna ubicación"\n    echo "  La aplicación podría no funcionar correctamente"\nelse\n    echo "✅ Se encontraron $total_models modelo(s) en total"\nfi\necho "🚀 Iniciando aplicación..."\nexec "$@"' > /usr/local/bin/entrypoint.sh && \
    chmod +x /usr/local/bin/entrypoint.sh

# Usar el script de entrada para verificar modelos antes de iniciar
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
