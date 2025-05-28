# main.py
import argparse
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from eneagrama_predictor.optimized_enneagram_system import OptimizedEnneagramSystem, run_optimized_enneagram_system
from eneagrama_predictor.explainability import AdvancedEnneagramExplainer


# Importar portable_joblib si está disponible
try:
    from eneagrama_predictor.portable_joblib import create_portable_model
    PORTABLE_MODEL_AVAILABLE = True
except ImportError:
    PORTABLE_MODEL_AVAILABLE = False
    print("Aviso: portable_joblib no encontrado. No se crearán modelos portables.")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Enneagram Type Prediction System')
    parser.add_argument('--mode', choices=['train', 'predict', 'evaluate', 'optimize'], required=True,
                      help='Mode of operation: train, predict, evaluate, or optimize')    
    parser.add_argument('--data', required=False, default=None,
                      help='Path to the data file (optional, uses Supabase by default)')
    parser.add_argument('--model', default='enneagram_model.joblib',
                      help='Path to save/load the model')
    parser.add_argument('--output', default='results',
                      help='Directory to save results')
    parser.add_argument('--questions', type=int, default=135,
                      help='Number of questions to use (for train/predict mode) or maximum questions (for optimize mode)')
    parser.add_argument('--samples', type=int, default=None,
                      help='Number of samples to predict (for predict mode). None for all samples')
    parser.add_argument('--portable', action='store_true',
                      help='Create portable model versions')
    args = parser.parse_args()
      # Crear directorio de salida si no existe
    os.makedirs(args.output, exist_ok=True)
    
    # Añadir importación de la función de carga
    from eneagrama_predictor.optimized_enneagram_system import load_training_data

    # Cargar datos desde Supabase o CSV
    if args.data:
        print(f"Cargando datos desde archivo CSV: {args.data}...")
        try:
            # Cargar desde CSV específico
            data = pd.read_csv(args.data)
            print(f"✅ Datos cargados desde CSV: {data.shape}")
        except Exception as e:
            print(f"❌ Error cargando CSV: {e}")
            print("Intentando cargar desde Supabase como fallback...")
            try:
                X, y = load_training_data(source='supabase')
                # Convertir de vuelta a formato original para compatibilidad
                data = pd.concat([X, y], axis=1)
                print(f"✅ Datos cargados desde Supabase: {data.shape}")
            except Exception as e2:
                print(f"❌ Error cargando desde Supabase: {e2}")
                return
    else:
        print("📡 Cargando datos desde Supabase...")
        try:
            X, y = load_training_data(source='supabase')
            # Convertir a formato original para compatibilidad con el resto del código
            data = pd.concat([X, y], axis=1)
            print(f"✅ Datos cargados desde Supabase: {data.shape}")
        except Exception as e:
            print(f"❌ Error cargando desde Supabase: {e}")
            print("💡 Tip: Usa --data path/to/file.csv para cargar desde CSV")
            return
      # Determinar número de columnas de características (por defecto 135)
    num_features = min(args.questions, data.shape[1] - 2)  # -2 para dejar las columnas de tipo y ala
    
    # Determinar los nombres correctos de las columnas
    if 'true_type' in data.columns and 'true_wing' in data.columns:
        type_col = 'true_type'
        wing_col = 'true_wing'
    else:
        type_col = 'eneatipo'
        wing_col = 'ala'
      # Para el modo optimize, usar todas las características
    if args.mode == 'optimize':
        # Usar todas las columnas disponibles (hasta 135)
        max_features = min(135, data.shape[1] - 2)
        X = data.iloc[:, :max_features]
    else:
        # Usar el número especificado de características
        X = data.iloc[:, :num_features]
    
    # Crear DataFrame con las columnas necesarias
    y = pd.DataFrame({
        'true_type': data[type_col],
        'true_wing': data[wing_col]
    })

    print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")

    # Realizar operación según el modo seleccionado
    if args.mode == 'train':
        train_model(X, y, args.model, args.output, args.portable)
    elif args.mode == 'predict':
        predict_samples(X, y, args.model, args.output, args.samples)
    elif args.mode == 'evaluate':
        evaluate_model(X, y, args.model, args.output)
    elif args.mode == 'optimize':
        optimize_questions(X, y, args.output, args.questions, args.portable)

def train_model(X, y, model_path, output_dir, portable=False):
    """Entrena un nuevo modelo y lo guarda"""
    print("\nIniciando entrenamiento del sistema...")
    
    # Generar timestamp y nombres de archivo consistentes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"model_{timestamp}"
    
    # Construir rutas completas
    model_save_path = os.path.join(output_dir, f"{base_filename}.joblib")
    metrics_file = os.path.join(output_dir, f"{base_filename}_metrics.json")
    
    # Asegurar que el directorio existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Entrenar sistema
    metrics, system = run_optimized_enneagram_system(
        data_path=None,
        output_path=model_save_path,
        custom_data=(X, y)
    )
    
    # Crear modelo portable si se solicitó
    if portable and PORTABLE_MODEL_AVAILABLE:
        portable_path = os.path.join(output_dir, f"{base_filename}_portable.joblib")
        try:
            create_portable_model(system, portable_path)
            print(f"Modelo portable guardado en: {portable_path}")
        except Exception as e:
            print(f"Error al crear modelo portable: {e}")
    
    # Guardar métricas
    with open(metrics_file, 'w') as f:
        json.dump({
            'type_accuracy': float(metrics['type_accuracy']),
            'wing_accuracy': float(metrics['wing_accuracy']),
            'features_used': X.shape[1],
            'samples_count': X.shape[0],
            'timestamp': timestamp,
            'model_file': os.path.basename(model_save_path)
        }, f, indent=4)
    
    print("\nResultados del entrenamiento:")
    print(f"Accuracy de Eneatipo: {metrics['type_accuracy']:.4f}")
    print(f"Accuracy de Ala: {metrics['wing_accuracy']:.4f}")
    print(f"\nModelo guardado en: {model_save_path}")
    print(f"Métricas guardadas en: {metrics_file}")

def load_model(model_path):
    """Carga un modelo previamente entrenado"""
    import joblib
    
    try:
        print(f"Cargando modelo desde {model_path}...")
        system = joblib.load(model_path)
        print("Modelo cargado correctamente")
        return system
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        print("Creando un nuevo sistema...")
        return OptimizedEnneagramSystem()

def predict_samples(X, y, model_path, output_dir, num_samples=5):
    """Genera predicciones y reportes explicativos para muestras"""
    # Cargar modelo
    system = load_model(model_path)
    
    # Crear explicador
    explainer = AdvancedEnneagramExplainer(system, feature_names=X.columns)
    
    # Limitar número de muestras
    samples_to_predict = min(num_samples, X.shape[0])
    
    print(f"\nGenerando predicciones para {samples_to_predict} muestras...")
    
    # Generar predicciones y reportes
    for i in range(samples_to_predict):
        sample = X.iloc[i:i+1]
        sample_y = y.iloc[i:i+1]
        
        # Generar reporte
        report = explainer.generate_report(sample)
        
        # Guardar reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"prediction_report_{i+1}_{timestamp}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            # Añadir info de tipo real si disponible
            if not sample_y.empty:
                true_type = sample_y['true_type'].values[0]
                true_wing = sample_y['true_wing'].values[0]
                f.write(f"# REPORTE DE PREDICCIÓN\n\n")
                f.write(f"## Información Real (Ground Truth)\n")
                f.write(f"Tipo Real: {true_type}\n")
                f.write(f"Ala Real: {true_wing}\n\n")
            
            # Escribir reporte generado
            f.write(report)
        
        print(f"Reporte generado para muestra {i+1} y guardado en {report_file}")
        
        # Para la primera muestra, mostrar reporte en consola
        if i == 0:
            print("\nEjemplo de reporte generado:")
            print("=" * 50)
            print(report[:500] + "...\n[Reporte truncado para la visualización]")
            print("=" * 50)
    
    # Generar camino de crecimiento para la primera muestra
    if samples_to_predict > 0:
        sample = X.iloc[0:1]
        predictions = system.predict(sample)
        eneatipo = int(predictions['eneatipo'][0])
        ala = int(predictions['ala'][0])
        
        growth_path = explainer.suggest_growth_path(eneatipo, ala)
        growth_file = os.path.join(output_dir, f"growth_path_{eneatipo}w{ala}_{timestamp}.md")
        
        with open(growth_file, 'w', encoding='utf-8') as f:
            f.write(growth_path)
        
        print(f"\nCamino de crecimiento generado para Tipo {eneatipo}w{ala} y guardado en {growth_file}")

def evaluate_by_type(system, X, y):
    """Evalúa el rendimiento del modelo por tipo de eneagrama"""
    # Hacer predicciones
    predictions = system.predict(X)
    
    # Determinar los nombres correctos de las columnas
    type_col = 'true_type'
    
    # Inicializar resultados
    type_accuracies = {}
    
    # Para cada tipo posible (1-9)
    for tipo in range(1, 10):
        # Seleccionar muestras de este tipo
        indices = y[y[type_col] == tipo].index
        
        if len(indices) > 0:
            # Contar predicciones correctas
            correct_type = (predictions['eneatipo'][indices] == tipo).sum()
            
            # Calcular accuracy para este tipo
            type_accuracies[str(tipo)] = {
                'accuracy': float(correct_type / len(indices)),
                'sample_count': int(len(indices))
            }
    
    return type_accuracies

def evaluate_model(X, y, model_path, output_dir):
    """Evalúa el modelo en el conjunto de datos completo"""
    # Cargar modelo
    system = load_model(model_path)
    
    print("\nEvaluando modelo en el conjunto de datos...")
    
    # Evaluar en datos completos
    metrics = system.evaluate(X, y)
    
    # Evaluar por tipo
    type_accuracies = evaluate_by_type(system, X, y)
    
    # Crear matriz de confusión
    confusion_matrix = create_confusion_matrix(system, X, y)
    
    # Guardar métricas
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_file = os.path.join(output_dir, f"evaluation_metrics_{timestamp}.json")
    
    # Métricas completas
    full_metrics = {
        'overall': {
            'type_accuracy': float(metrics['type_accuracy']),
            'wing_accuracy': float(metrics['wing_accuracy']),
        },
        'by_type': type_accuracies,
        'dataset_info': {
            'samples': X.shape[0],
            'features': X.shape[1],
            'timestamp': timestamp
        }
    }
    
    # Guardar en json
    with open(metrics_file, 'w') as f:
        json.dump(full_metrics, f, indent=4)
    
    # Visualizar resultados
    plt.figure(figsize=(10, 6))
    tipos = [i for i in range(1, 10) if str(i) in type_accuracies]
    accuracies = [type_accuracies[str(i)]['accuracy'] for i in tipos]
    
    plt.bar(tipos, accuracies)
    plt.axhline(y=metrics['type_accuracy'], color='r', linestyle='--', label=f'Media: {metrics["type_accuracy"]:.2f}')
    plt.xlabel('Tipo de Eneagrama')
    plt.ylabel('Accuracy')
    plt.title('Rendimiento por Tipo de Eneagrama')
    plt.ylim(0, 1.0)
    plt.xticks(tipos)
    plt.legend()
    
    # Guardar gráfico
    plot_file = os.path.join(output_dir, f"accuracy_by_type_{timestamp}.png")
    plt.savefig(plot_file)
    
    # Mostrar resultados
    print("\nResultados de la evaluación:")
    print(f"Accuracy de Eneatipo: {metrics['type_accuracy']:.4f}")
    print(f"Accuracy de Ala: {metrics['wing_accuracy']:.4f}")
    
    print("\nAccuracy por tipo:")
    for tipo in range(1, 10):
        if str(tipo) in type_accuracies:
            accuracy = type_accuracies[str(tipo)]['accuracy']
            samples = type_accuracies[str(tipo)]['sample_count']
            print(f"  Tipo {tipo}: {accuracy:.4f} ({samples} muestras)")
    
    print(f"\nMétricas guardadas en: {metrics_file}")
    print(f"Gráfico guardado en: {plot_file}")
    
    # Guardar matriz de confusión
    save_confusion_matrix(confusion_matrix, output_dir, timestamp)

def create_confusion_matrix(system, X, y):
    """Crea una matriz de confusión para evaluar el rendimiento del modelo"""
    predictions = system.predict(X)
    
    # Crear matriz 9x9 (para los 9 tipos)
    confusion = np.zeros((9, 9), dtype=int)
    
    # Llenar matriz
    for true_type in range(1, 10):
        for pred_type in range(1, 10):
            # Contar muestras con true_type que fueron predichas como pred_type
            count = ((y['true_type'] == true_type) & (predictions['eneatipo'] == pred_type)).sum()
            confusion[true_type-1, pred_type-1] = count
    
    return confusion

def save_confusion_matrix(confusion, output_dir, timestamp):
    """Guarda una visualización de la matriz de confusión"""
    plt.figure(figsize=(10, 8))
    
    # Crear mapa de calor
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusión')
    plt.colorbar()
    
    # Etiquetas
    tipos = [f"Tipo {i}" for i in range(1, 10)]
    plt.xticks(range(9), tipos, rotation=45)
    plt.yticks(range(9), tipos)
    
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    
    # Añadir valores a las celdas
    for i in range(9):
        for j in range(9):
            plt.text(j, i, str(confusion[i, j]),
                     horizontalalignment="center",
                     color="white" if confusion[i, j] > confusion.max() / 2 else "black")
    
    plt.tight_layout()
    
    # Guardar gráfico
    matrix_file = os.path.join(output_dir, f"confusion_matrix_{timestamp}.png")
    plt.savefig(matrix_file)
    print(f"Matriz de confusión guardada en: {matrix_file}")

def optimize_questions(X, y, output_dir, max_questions=135, portable=False):
    """Optimiza el número de preguntas evaluando diferentes cantidades"""
    print("\n===== OPTIMIZACIÓN DEL NÚMERO DE PREGUNTAS =====")
    
    # 1. Entrenar modelo completo como referencia
    print("\nEntrenando modelo completo...")
    system = OptimizedEnneagramSystem()
    metrics = system.train(X, y)
    
    # Guardar modelo completo
    full_model_path = os.path.join(output_dir, 'model_full.joblib')
    joblib.dump(system, full_model_path)
    print(f"Modelo completo guardado en: {full_model_path}")
    
    # Crear modelo portable si se solicitó
    if portable and PORTABLE_MODEL_AVAILABLE:
        portable_path = os.path.join(output_dir, 'model_full_portable.joblib')
        try:
            create_portable_model(system, portable_path)
            print(f"Modelo portable guardado en: {portable_path}")
        except Exception as e:
            print(f"Error al crear modelo portable: {e}")
    
    # 2. Definir cantidades de preguntas a evaluar
    # Ajustar según el máximo especificado
    if max_questions > 100:
        question_counts = [60, 70, 80, 90, 100, max_questions]
    else:
        # Si es un número menor, crear una secuencia razonable
        step = max(5, max_questions // 8)  # al menos 8 puntos de evaluación
        question_counts = list(range(step, max_questions, step)) + [max_questions]
    
    # 3. Inicializar resultados con el modelo completo
    results = [{
        'n_questions': X.shape[1],  # número de características usadas
        'type_accuracy': metrics['type_accuracy'],
        'wing_accuracy': metrics['wing_accuracy']
    }]
    
    # 4. Evaluar cada cantidad de preguntas (excepto las completas que ya evaluamos)
    for n_questions in [q for q in question_counts if q != X.shape[1]]:
        print(f"\n===== EVALUANDO MODELO CON {n_questions} PREGUNTAS =====")
        
        # Identificar las preguntas más importantes
        print(f"Identificando las {n_questions} preguntas más importantes...")
        important = system.identify_important_features(X, y, n_questions)
        selected_indices = important['selected_indices']
        
        # Crear dataset reducido
        X_reduced = X.iloc[:, selected_indices]
        
        # Guardar lista de preguntas seleccionadas para referencia
        questions_path = os.path.join(output_dir, f'preguntas_top_{n_questions}.txt')
        with open(questions_path, 'w', encoding='utf-8') as f:
            f.write(f"Lista de las {n_questions} preguntas más importantes para la predicción de eneagrama:\n\n")
            for i, feature in enumerate(important['feature_names']):
                f.write(f"{i+1}. {feature}\n")
        
        # Entrenar nuevo modelo con el conjunto reducido
        print(f"Entrenando modelo con {n_questions} preguntas...")
        reduced_system = OptimizedEnneagramSystem()
        reduced_metrics = reduced_system.train(X_reduced, y)
        
        # Guardar el modelo reducido
        reduced_model_path = os.path.join(output_dir, f'model_{n_questions}_preguntas.joblib')
        joblib.dump(reduced_system, reduced_model_path)
        print(f"Modelo con {n_questions} preguntas guardado en: {reduced_model_path}")
        
        # Crear modelo portable si se solicitó
        if portable and PORTABLE_MODEL_AVAILABLE:
            portable_path = os.path.join(output_dir, f'model_{n_questions}_preguntas_portable.joblib')
            try:
                create_portable_model(reduced_system, portable_path)
                print(f"Modelo portable guardado en: {portable_path}")
            except Exception as e:
                print(f"Error al crear modelo portable: {e}")
        
        # Guardar resultados
        results.append({
            'n_questions': n_questions,
            'type_accuracy': reduced_metrics['type_accuracy'],
            'wing_accuracy': reduced_metrics['wing_accuracy']
        })
        
        print(f"Resultados con {n_questions} preguntas:")
        print(f"  Accuracy Tipo: {reduced_metrics['type_accuracy']:.4f}")
        print(f"  Accuracy Ala: {reduced_metrics['wing_accuracy']:.4f}")
    
    # 5. Ordenar resultados por número de preguntas
    results.sort(key=lambda x: x['n_questions'])
    
    # 6. Mostrar tabla comparativa
    print("\n===== RESULTADOS COMPARATIVOS =====")
    print("\nRendimiento por número de preguntas:")
    print("Num Preguntas | Accuracy Tipo | Accuracy Ala")
    print("-"*50)
    for result in results:
        print(f"{result['n_questions']:12d} | {result['type_accuracy']:.4f} | {result['wing_accuracy']:.4f}")
    
    # 7. Guardar resultados en CSV
    csv_path = os.path.join(output_dir, 'optimization_results.csv')
    results_df = pd.DataFrame(results)
    results_df.to_csv(csv_path, index=False)
    print(f"\nResultados guardados en: {csv_path}")
    
    # 8. Crear visualización
    questions = [r['n_questions'] for r in results]
    type_acc = [r['type_accuracy'] for r in results]
    wing_acc = [r['wing_accuracy'] for r in results]

    plt.figure(figsize=(12, 8))
    
    # Gráfico de líneas
    plt.subplot(2, 1, 1)
    plt.plot(questions, type_acc, 'o-', linewidth=2, markersize=8, label='Accuracy Tipo')
    plt.plot(questions, wing_acc, 's-', linewidth=2, markersize=8, label='Accuracy Ala')
    plt.xlabel('Número de Preguntas')
    plt.ylabel('Accuracy')
    plt.title('Rendimiento vs. Número de Preguntas')
    plt.grid(True)
    plt.legend()
    
    # Gráfico de barras
    plt.subplot(2, 1, 2)
    bar_width = 0.35
    index = np.arange(len(questions))
    plt.bar(index, type_acc, bar_width, label='Accuracy Tipo')
    plt.bar(index + bar_width, wing_acc, bar_width, label='Accuracy Ala')
    plt.xlabel('Número de Preguntas')
    plt.ylabel('Accuracy')
    plt.title('Comparación de Accuracy')
    plt.xticks(index + bar_width/2, questions)
    plt.legend()
    
    # Guardar la visualización
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'optimization_plot.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Visualización guardada en: {plot_path}")
    
    # 9. Determinar el número óptimo de preguntas
    # Buscar el punto donde añadir más preguntas ofrece rendimiento marginal
    best_n = find_optimal_question_count(results)
    
    print(f"\nNúmero de preguntas recomendado: {best_n}")
    print("Este número ofrece un buen equilibrio entre longitud del cuestionario y precisión.")
    
    # 10. Guardar el cuestionario recomendado
    recommended = system.identify_important_features(X, y, best_n)
    recommended_path = os.path.join(output_dir, 'cuestionario_recomendado.txt')
    with open(recommended_path, 'w', encoding='utf-8') as f:
        f.write(f"CUESTIONARIO RECOMENDADO DE ENEAGRAMA ({best_n} PREGUNTAS)\n")
        f.write("Este cuestionario reducido ofrece un buen equilibrio entre longitud y precisión.\n\n")
        for i, feature in enumerate(recommended['feature_names']):
            f.write(f"{i+1}. {feature}\n")
    print(f"Cuestionario recomendado guardado en: {recommended_path}")

def find_optimal_question_count(results):
    """Encuentra el número óptimo de preguntas usando análisis de rendimiento marginal"""
    # Ordenar por número de preguntas
    sorted_results = sorted(results, key=lambda x: x['n_questions'])
    
    # Calcular mejoras marginales
    best_n = sorted_results[0]['n_questions']  # valor predeterminado
    best_improvement_ratio = 0
    
    for i in range(len(sorted_results) - 1):
        current = sorted_results[i]
        next_one = sorted_results[i + 1]
        
        # Diferencia en número de preguntas
        question_diff = next_one['n_questions'] - current['n_questions']
        
        # Mejora en accuracy
        accuracy_improvement = next_one['type_accuracy'] - current['type_accuracy']
        
        # Tasa de mejora por pregunta adicional
        if question_diff > 0:
            improvement_ratio = accuracy_improvement / question_diff
            
            # Buscar la mejor relación costo-beneficio (mayor mejora por pregunta)
            if improvement_ratio > best_improvement_ratio:
                best_improvement_ratio = improvement_ratio
                best_n = current['n_questions']
            
            # Si la mejora es menos del 0.2% por pregunta adicional, considerar este punto como óptimo
            if improvement_ratio < 0.002 and current['n_questions'] >= 59:
                return current['n_questions']
    
    # Si no encontramos un punto claro, usar heurística
    # Retornar el primer punto con al menos 85% de la accuracy máxima
    max_accuracy = max([r['type_accuracy'] for r in sorted_results])
    threshold = 0.85 * max_accuracy
    
    for result in sorted_results:
        if result['type_accuracy'] >= threshold:
            return result['n_questions']
    
    # Si todo falla, retornar el número más bajo evaluado
    return sorted_results[0]['n_questions']

if __name__ == "__main__":
    main()