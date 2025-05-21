# evaluate_question_counts.py
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Añadir el directorio actual al path para encontrar portable_joblib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from eneagrama_predictor.portable_joblib import create_portable_model
from eneagrama_predictor.optimized_enneagram_system import OptimizedEnneagramSystem, run_optimized_enneagram_system

def main():
    parser = argparse.ArgumentParser(description='Evaluar diferentes cantidades de preguntas')
    parser.add_argument('--data', required=True, help='Path to the data file')
    parser.add_argument('--output', default='question_metrics.csv', help='Output metrics file')
    parser.add_argument('--save-models', action='store_true', help='Save models for each question count')
    args = parser.parse_args()

    print(f"Cargando datos desde {args.data}...")
    try:
        # Cargar datos con el formato correcto para true_type y true_wing
        data = pd.read_csv(args.data)
        X = data.iloc[:, :135]  # First 135 columns are features
        
        # Crear DataFrame con las columnas necesarias (manteniendo true_type y true_wing)
        y = pd.DataFrame({
            'true_type': data['true_type'],
            'true_wing': data['true_wing']
        })

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Crear directorio para outputs si no existe
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)

    # 1. Entrenar modelo completo como referencia
    print("\n===== ENTRENANDO MODELO COMPLETO (135 PREGUNTAS) =====")
    full_metrics, system = run_optimized_enneagram_system(args.data)
    
    # Guardar el modelo completo como portable si se ha solicitado
    if args.save_models:
        joblib_path = os.path.join(output_dir, 'model_135_preguntas.joblib')
        portable_path = os.path.join(output_dir, 'portable_model_135_preguntas.joblib')
        # Guardar modelo normal
        joblib.dump(system, joblib_path)
        # Guardar modelo portable
        create_portable_model(system, portable_path)
        print(f"Modelo completo (135 preguntas) guardado como portable en: {portable_path}")
    
    # 2. Definir cantidades de preguntas a evaluar
    question_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 135]
    
    # 3. Inicializar resultados con el modelo completo
    results = [{
        'n_questions': 135,
        'type_accuracy': full_metrics['type_accuracy'],
        'wing_accuracy': full_metrics['wing_accuracy']
    }]
    
    # 4. Evaluar cada cantidad de preguntas (excepto 135 que ya evaluamos)
    for n_questions in [q for q in question_counts if q != 135]:
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
        temp_joblib_path = 'temp_model.joblib'
        print(f"Entrenando modelo con {n_questions} preguntas...")
        reduced_metrics, reduced_system = run_optimized_enneagram_system(
            args.data, 
            output_path=temp_joblib_path,
            custom_data=(X_reduced, y)
        )
        
        # Guardar el modelo reducido si se solicitó
        if args.save_models:
            joblib_path = os.path.join(output_dir, f'model_{n_questions}_preguntas.joblib')
            
            # Guardar modelo normal
            joblib.dump(reduced_system, joblib_path)
            print(f"Modelo con {n_questions} preguntas guardado en: {joblib_path}")
            
            try:
                # Intentar guardar como modelo portable
                portable_path = os.path.join(output_dir, f'portable_model_{n_questions}_preguntas.joblib')
                create_portable_model(reduced_system, portable_path)
                print(f"Modelo portable con {n_questions} preguntas guardado en: {portable_path}")
            except Exception as e:
                print(f"Warning: Could not create portable model: {e}")
        
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
    
    # 7. Guardar resultados en CSV en la carpeta outputs
    csv_path = os.path.join(output_dir, args.output)
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
    
    # Guardar la visualización en outputs
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparison_metrics.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Visualización guardada en: {plot_path}")
    
    # 9. Guardar cuestionario reducido
    # Determinar el "mejor equilibrio" entre cantidad de preguntas y rendimiento
    
    # Buscar el punto donde añadir más preguntas ofrece rendimiento marginal
    # Como heurística, buscaremos el punto donde añadir 10 preguntas mejora menos de 2%
    best_n = 30  # Valor predeterminado si no encontramos un punto claro
    
    for i in range(1, len(results)-1):
        current = results[i]
        next_one = results[i+1]
        n_diff = next_one['n_questions'] - current['n_questions']
        acc_improvement = next_one['type_accuracy'] - current['type_accuracy']
        
        # Si añadir n_diff preguntas mejora menos de 2%, consideramos que este es un buen equilibrio
        if n_diff > 0 and (acc_improvement / n_diff) < 0.002:  # menos de 0.2% por pregunta
            best_n = current['n_questions']
            break
    
    print(f"\nNúmero de preguntas recomendado: {best_n}")
    print("Este número ofrece un buen equilibrio entre longitud del cuestionario y precisión.")
    
    # Guardar el cuestionario recomendado en outputs
    if best_n != 135:
        recommended = system.identify_important_features(X, y, best_n)
        recommended_path = os.path.join(output_dir, 'cuestionario_recomendado.txt')
        with open(recommended_path, 'w', encoding='utf-8') as f:
            f.write(f"CUESTIONARIO RECOMENDADO DE ENEAGRAMA ({best_n} PREGUNTAS)\n")
            f.write("Este cuestionario reducido ofrece un buen equilibrio entre longitud y precisión.\n\n")
            for i, feature in enumerate(recommended['feature_names']):
                f.write(f"{i+1}. {feature}\n")
        print(f"Cuestionario recomendado guardado en: {recommended_path}")

if __name__ == "__main__":
    main()