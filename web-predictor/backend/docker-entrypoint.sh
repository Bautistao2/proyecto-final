#!/bin/bash
# Script para verificar modelos disponibles al inicio del contenedor

echo "📂 Verificando modelos disponibles..."

# Búsqueda de modelos en todas las posibles ubicaciones
results_models=$(find /app/results -name "*.joblib" 2>/dev/null | wc -l)
results_alt_models=$(find /app/results-alt -name "*.joblib" 2>/dev/null | wc -l)
model_results_models=$(find /app/model/results -name "*.joblib" 2>/dev/null | wc -l)

total_models=$((results_models + results_alt_models + model_results_models))

# Mostrar los modelos encontrados
echo "🔍 Modelos encontrados:"
if [ $results_models -gt 0 ]; then
    echo "  - /app/results: $results_models modelo(s)"
    ls -l /app/results/*.joblib
fi

if [ $results_alt_models -gt 0 ]; then
    echo "  - /app/results-alt: $results_alt_models modelo(s)"
    ls -l /app/results-alt/*.joblib
fi

if [ $model_results_models -gt 0 ]; then
    echo "  - /app/model/results: $model_results_models modelo(s)"
    ls -l /app/model/results/*.joblib
fi

if [ $total_models -eq 0 ]; then
    echo "⚠️ ADVERTENCIA: No se encontraron modelos .joblib en ninguna ubicación"
    echo "  La aplicación podría no funcionar correctamente"
else
    echo "✅ Se encontraron $total_models modelo(s) en total"
fi

echo "🚀 Iniciando aplicación..."

# Ejecutar el comando original
exec "$@"
