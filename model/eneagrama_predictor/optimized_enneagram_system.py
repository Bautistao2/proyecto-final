#optimized_enneagram_system.py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from eneagrama_predictor.data_preprocessing import DataPreprocessor
from eneagrama_predictor.data_augmentation import EnneagramDataAugmentor

# Importaciones para Supabase
import os
from pathlib import Path
try:
    from supabase import create_client, Client
    from dotenv import load_dotenv
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False    
    # Solo mostrar advertencia si se está ejecutando directamente el módulo
    import sys
    if __name__ == "__main__" or "main" in sys.argv[0]:
        print("⚠️ Supabase no disponible. Usando CSV como fallback.")

def load_training_data(source='auto', csv_fallback_path='data.csv'):
    """
    Carga datos de entrenamiento desde Supabase o CSV como fallback
    
    Args:
        source: 'auto', 'supabase', 'csv'
        csv_fallback_path: Ruta al archivo CSV de fallback
        
    Returns:
        tuple: (X, y) donde X son las características y y las etiquetas
    """
    print(f"🔄 Intentando cargar datos con fuente: {source}")
    
    if source == 'auto' or source == 'supabase':
        if SUPABASE_AVAILABLE:
            try:
                print("📡 Conectando a Supabase...")
                
                # Cargar variables de entorno
                env_path = Path(__file__).parent.parent.parent / '.env'
                if env_path.exists():
                    load_dotenv(dotenv_path=env_path)
                else:
                    # Buscar .env en otros lugares
                    for possible_path in [Path.cwd() / '.env', Path(__file__).parent / '.env']:
                        if possible_path.exists():
                            load_dotenv(dotenv_path=possible_path)
                            break
                
                # Configuración de Supabase
                supabase_url = os.getenv("SUPABASE_URL")
                supabase_key = os.getenv("SUPABASE_KEY")
                table_name = os.getenv("SUPABASE_TABLE", "respuestas_para_ia")
                
                if not (supabase_url and supabase_key):
                    raise Exception("Credenciales de Supabase no encontradas en variables de entorno")
                
                # Crear cliente Supabase
                supabase: Client = create_client(supabase_url, supabase_key)
                
                print(f"📊 Cargando datos desde tabla: {table_name}")
                
                # Cargar todos los datos en lotes para evitar límites
                all_data = []
                batch_size = 1000
                offset = 0
                
                while True:
                    response = supabase.table(table_name).select("*").range(offset, offset + batch_size - 1).execute()
                    
                    if not response.data:
                        break
                    
                    all_data.extend(response.data)
                    offset += batch_size
                    
                    if len(response.data) < batch_size:
                        break
                
                if not all_data:
                    raise Exception(f"No se encontraron datos en la tabla {table_name}")
                
                # Convertir a DataFrame
                df = pd.DataFrame(all_data)
                print(f"✅ Datos cargados desde Supabase: {df.shape}")
                
                # Verificar que tenemos las columnas necesarias
                question_cols = [col for col in df.columns if col.startswith('Q') and col[1:].isdigit()]
                if len(question_cols) < 100:
                    raise Exception(f"Columnas de preguntas insuficientes: {len(question_cols)}")
                
                # Separar características (X) y etiquetas (y)
                X = df[question_cols].copy()
                
                # Buscar columnas de etiquetas
                if 'true_type' in df.columns and 'true_wing' in df.columns:
                    y = df[['true_type', 'true_wing']].copy()
                elif 'eneatipo' in df.columns and 'ala' in df.columns:
                    y = df[['eneatipo', 'ala']].rename(columns={'eneatipo': 'true_type', 'ala': 'true_wing'})
                else:
                    raise Exception("No se encontraron columnas de etiquetas válidas")
                
                print(f"📋 Características: {X.shape}")
                print(f"🎯 Etiquetas: {y.shape}")
                print(f"🔢 Tipos únicos: {sorted(y['true_type'].unique())}")
                print(f"🪶 Alas únicas: {sorted(y['true_wing'].unique())}")
                
                return X, y
                
            except Exception as e:
                print(f"❌ Error cargando desde Supabase: {e}")
                if source == 'supabase':
                    raise
                print("🔄 Fallback a CSV...")
    
    # Fallback a CSV
    print(f"📄 Cargando datos desde CSV: {csv_fallback_path}")
    try:
        df = pd.read_csv(csv_fallback_path)
        X = df.iloc[:, :135]  # Preguntas Q1-Q135
        y = df[['true_type', 'true_wing']].copy()
        
        print(f"✅ Datos cargados desde CSV: {df.shape}")
        print(f"📋 Características: {X.shape}")
        print(f"🎯 Etiquetas: {y.shape}")
        
        return X, y
        
    except Exception as e:
        print(f"❌ Error cargando CSV: {e}")
        raise Exception(f"No se pudieron cargar datos ni desde Supabase ni desde CSV: {e}")

class OptimizedEnneagramSystem:
    """Sistema optimizado para predicción de eneatipos"""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor(feature_threshold=0.2)
        self.ensemble = {
            'models': [],
            'weights': []
        }
        self.metrics = {
            'type_accuracy': 0.0,
            'wing_accuracy': 0.0
        }
    
    def train(self, X, y):
        """Entrena el sistema con datos de entrada"""
        print("Aumentando datos...")

        # Crear objeto de aumento de datos
        augmentor = EnneagramDataAugmentor(min_samples=100)
        
        # Determinar los nombres correctos de las columnas
        type_col = 'true_type' if 'true_type' in y.columns else 'eneatipo'
        wing_col = 'true_wing' if 'true_wing' in y.columns else 'ala'
        
        # Aumentar datos balanceando tipos
        X_aug, y_type_aug, y_wing_aug = augmentor.augment_data(
            X, y[type_col], y[wing_col]
        )
        
        # Opcionalmente aumentar combinaciones tipo-ala poco frecuentes
        # Esto garantiza un mínimo de muestras para cada combinación válida
        X_aug, y_type_aug, y_wing_aug = augmentor.augment_rare_wings(
            X_aug, y_type_aug, y_wing_aug, wing_min_samples=20
        )

        # Convertir de vuelta al formato de DataFrame esperado
        y_aug = pd.DataFrame({
            'eneatipo': y_type_aug,
            'ala': y_wing_aug
        })
        
        print(f"Tamaño del dataset original: {len(X)}")
        print(f"Tamaño del dataset aumentado: {len(X_aug)}")
        
        # Verificar y ajustar etiquetas antes del split
        y_aug['eneatipo'] = y_aug['eneatipo'].clip(1, 9) - 1  # Convertir 1-9 a 0-8
        y_aug['ala'] = y_aug['ala'].clip(1, 9) - 1  # Convertir 1-9 a 0-8
        
        # Split con datos aumentados
        X_train, X_val, y_train, y_val = train_test_split(
            X_aug, y_aug, test_size=0.2, random_state=42, stratify=y_aug['eneatipo']
        )
        
        # Preprocesar datos
        print("Preprocesando datos...")
        X_train_scaled = self.preprocessor.fit_transform(X_train, y_train['eneatipo'])
        X_val_scaled = self.preprocessor.transform(X_val)
        
        # Crear y entrenar modelos del ensemble
        models = self._create_ensemble_models(X_train_scaled.shape[1])
        
        for model in models:
            # Compilar modelo con métricas para cada salida
            model.compile(
                optimizer='adam',
                loss=['sparse_categorical_crossentropy', 'sparse_categorical_crossentropy'],
                metrics={
                    'eneatipo': ['accuracy'],
                    'ala': ['accuracy']
                }
            )
            
            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            # Entrenar modelo
            history = model.fit(
                X_train_scaled,
                [y_train['eneatipo'], y_train['ala']],
                validation_data=(X_val_scaled, [y_val['eneatipo'], y_val['ala']]),
                epochs=100,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=1
                        )
            
            self.ensemble['models'].append(model)
            self.ensemble['weights'].append(1.0/len(models))
        
        # Evaluar con datos originales para métricas finales
        metrics = self.evaluate(X, y)
        self.metrics.update(metrics)
        
        return self.metrics
    
    def predict(self, X):
        """Realiza predicciones para nuevos datos"""
        # Preprocesar datos
        X_scaled = self.preprocessor.transform(X)
        
        # Predicciones del ensemble
        predictions = {
            'eneatipo': [],
            'ala': [],
            'eneatipo_probabilidades': [],
            'ala_probabilidades': []
        }
        
        # Obtener predicciones de cada modelo
        for model, weight in zip(self.ensemble['models'], self.ensemble['weights']):
            eneatipo_pred, ala_pred = model.predict(X_scaled)
            predictions['eneatipo_probabilidades'].append(eneatipo_pred * weight)
            predictions['ala_probabilidades'].append(ala_pred * weight)
        
        # Combinar predicciones
        eneatipo_probs = sum(predictions['eneatipo_probabilidades'])
        ala_probs = sum(predictions['ala_probabilidades'])
        
        # Obtener clases predichas
        predictions['eneatipo'] = np.argmax(eneatipo_probs, axis=1) + 1
        predictions['ala'] = np.argmax(ala_probs, axis=1) + 1
        predictions['eneatipo_probabilidades'] = eneatipo_probs
        predictions['ala_probabilidades'] = ala_probs
        
        return predictions
    
    def evaluate(self, X, y):
        """Evalúa el rendimiento del sistema"""
        predictions = self.predict(X)
        
        # Determinar los nombres correctos de las columnas
        type_col = 'true_type' if 'true_type' in y.columns else 'eneatipo'
        wing_col = 'true_wing' if 'true_wing' in y.columns else 'ala'
        
        # Calcular métricas
        type_correct = (predictions['eneatipo'] == y[type_col]).mean()
        wing_correct = (predictions['ala'] == y[wing_col]).mean()
        
        return {
            'type_accuracy': type_correct,
            'wing_accuracy': wing_correct
        }
    
    def _create_ensemble_models(self, input_shape):
        """Crea modelos más profundos para el ensemble"""
        models = []
        
        # Arquitecturas más complejas
        architectures = [
            [512, 256, 128, 64],  # Modelo más profundo
            [256, 256, 128, 128, 64],  # Modelo con más capas
            [384, 256, 128, 64, 32],  # Modelo alternativo
            [512, 384, 256, 128, 64]  # Modelo más ancho
        ]
        
        for units in architectures:
            inputs = tf.keras.Input(shape=(input_shape,))
            x = inputs
            
            # Capa de normalización inicial
            x = tf.keras.layers.BatchNormalization()(x)
            
            # Capas ocultas con residual connections
            for unit in units:
                residual = x
                x = tf.keras.layers.Dense(unit)(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.Activation('relu')(x)
                x = tf.keras.layers.Dropout(0.4)(x)  # Aumentamos dropout
                
                # Residual connection si es posible
                if residual.shape[-1] == unit:
                    x = tf.keras.layers.Add()([x, residual])
            
            # Capas de salida separadas con atención
            attention = tf.keras.layers.Dense(units[-1], activation='tanh')(x)
            attention = tf.keras.layers.Dense(1, activation='sigmoid')(attention)
            x = tf.keras.layers.Multiply()([x, attention])
            
            eneatipo_output = tf.keras.layers.Dense(9, activation='softmax', name='eneatipo')(x)
            ala_output = tf.keras.layers.Dense(9, activation='softmax', name='ala')(x)
            
            model = tf.keras.Model(inputs=inputs, outputs=[eneatipo_output, ala_output])
            models.append(model)
        
        return models
    
    def identify_important_features(self, X, y, n_features=30):
        """Identifica las características (preguntas) más importantes.
        
        Args:
            X: DataFrame con todas las características
            y: DataFrame con etiquetas (true_type, true_wing)
            n_features: Número de características a seleccionar
            
        Returns:
            Dictionary con información sobre características importantes
        """
        feature_names = X.columns.tolist()
        
        # Determinar los nombres correctos de las columnas
        type_col = 'true_type' if 'true_type' in y.columns else 'eneatipo'
        
        # Usar DataPreprocessor para selección basada en información mutua
        temp_preprocessor = DataPreprocessor(feature_threshold=0.0)  # No filtrar, solo rankear
        
        # Preprocesar datos para obtener información mutua
        temp_preprocessor.fit_transform(X, y[type_col])
        
        # Obtener los scores de información mutua directamente
        # Aplicar escalado para calcular información mutua
        X_array = X.values if isinstance(X, pd.DataFrame) else X
        X_scaled = temp_preprocessor.scaler.fit_transform(X_array)
        
        # Calcular información mutua directamente
        mi_scores = mutual_info_classif(X_scaled, y[type_col], random_state=42)
        
        # Ordenar características por importancia
        sorted_indices = np.argsort(-mi_scores)
        top_indices = sorted_indices[:n_features]
        
        # Obtener nombres e importancias
        top_features = [(feature_names[i], mi_scores[i]) for i in top_indices]
        
        return {
            'top_features': top_features,
            'selected_indices': top_indices,
            'feature_names': [feature_names[i] for i in top_indices],
            'mi_scores': mi_scores
        }

    def create_reduced_model(self, X, y, n_features=30):
        """Crea un modelo optimizado con un número reducido de preguntas.
        
        Args:
            X: DataFrame con todas las características
            y: DataFrame con etiquetas
            n_features: Número de características a utilizar
            
        Returns:
            Dictionary con modelo reducido e información asociada
        """
        # Identificar características importantes
        important_features = self.identify_important_features(X, y, n_features)
        selected_indices = important_features['selected_indices']
        
        # Crear dataset reducido
        X_reduced = X.iloc[:, selected_indices]
        
        # Crear nuevo sistema y entrenar
        reduced_system = OptimizedEnneagramSystem()
        reduced_system.train(X_reduced, y)
        
        return {
            'system': reduced_system,
            'selected_indices': selected_indices,
            'feature_names': important_features['feature_names'],
            'importance_info': important_features
        }

    def evaluate_question_reduction(self, X, y, question_counts=[20, 30, 50, 75, 100]):
        """Evalúa el rendimiento con diferentes cantidades de preguntas.
        
        Args:
            X: DataFrame con todas las características
            y: DataFrame con etiquetas
            question_counts: Lista de números de preguntas a evaluar
            
        Returns:
            Lista de resultados con métricas para cada cantidad de preguntas
        """
        results = []
        
        # Determinar los nombres correctos de las columnas
        type_col = 'true_type' if 'true_type' in y.columns else 'eneatipo'
        
        # Usar validación cruzada para evaluación robusta
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for n_questions in question_counts:
            print(f"\nEvaluando modelo con {n_questions} preguntas...")
            fold_scores = []
            
            for fold, (train_idx, test_idx) in enumerate(cv.split(X, y[type_col])):
                print(f"  Fold {fold+1}/3...")
                
                # División de datos
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Crear sistema temporal para identificar características
                temp_system = OptimizedEnneagramSystem()
                temp_system.train(X_train, y_train)
                
                # Identificar mejores preguntas
                important = self.identify_important_features(X_train, y_train, n_questions)
                selected_indices = important['selected_indices']
                
                # Entrenar con subset
                X_train_reduced = X_train.iloc[:, selected_indices]
                X_test_reduced = X_test.iloc[:, selected_indices]
                
                reduced_system = OptimizedEnneagramSystem()
                reduced_system.train(X_train_reduced, y_train)
                
                # Evaluar
                metrics = reduced_system.evaluate(X_test_reduced, y_test)
                fold_scores.append(metrics)
            
            # Promediar resultados
            avg_metrics = {
                'n_questions': n_questions,
                'type_accuracy': np.mean([score['type_accuracy'] for score in fold_scores]),
                'wing_accuracy': np.mean([score['wing_accuracy'] for score in fold_scores])
            }
            
            print(f"  Resultados con {n_questions} preguntas:")
            print(f"    Accuracy Tipo: {avg_metrics['type_accuracy']:.4f}")
            print(f"    Accuracy Ala: {avg_metrics['wing_accuracy']:.4f}")
            
            results.append(avg_metrics)
        
        return results

# Función principal para ejecutar el sistema completo
## def run_optimized_enneagram_system(data_path, output_path='optimized_enneagram_system.joblib'):
    """Ejecuta el sistema completo con datos proporcionados"""
    #print("=== SISTEMA OPTIMIZADO DE CLASIFICACIÓN DE ENEAGRAMA ===")
    

    # 1. Cargar datos
    #print("\nCargando datos...")
    #df = pd.read_csv(data_path)
    #X = df.iloc[:, :135]  # Preguntas Q1-Q135
    #y = df[['true_type', 'true_wing']].rename(columns={'true_type': 'eneatipo', 'true_wing': 'ala'})

    
    # 2. Construir y entrenar sistema
    #system = OptimizedEnneagramSystem()
    #metrics = system.train(X, y)
    
    # 3. Guardar sistema
    #import joblib
    #joblib.dump(system, output_path)
    
    # 4. Devolver métricas finales
    #return metrics, system

def run_optimized_enneagram_system(data_path=None, output_path='optimized_enneagram_system.joblib', custom_data=None, data_source='auto'):
    """Ejecuta el sistema completo con datos proporcionados"""
    print("=== SISTEMA OPTIMIZADO DE CLASIFICACIÓN DE ENEAGRAMA ===")

    system = OptimizedEnneagramSystem()
    
    if custom_data is None:
        # 1. Cargar datos usando la nueva función
        print("\nCargando datos...")
        
        # Usar la función de carga inteligente
        X, y = load_training_data(source=data_source, csv_fallback_path=data_path or 'data.csv')
        
        # Renombrar columnas para compatibilidad
        y = y.rename(columns={'true_type': 'eneatipo', 'true_wing': 'ala'})
        
        # === LÍNEAS COMENTADAS: CARGA CSV ORIGINAL ===
        # df = pd.read_csv(data_path)
        # X = df.iloc[:, :135]  # Preguntas Q1-Q135
        # y = df[['true_type', 'true_wing']].rename(columns={'true_type': 'eneatipo', 'true_wing': 'ala'})
    else:
        # Usar datos personalizados proporcionados
        X, y = custom_data
    
    # 2. Construir y entrenar sistema
    system = OptimizedEnneagramSystem()
    metrics = system.train(X, y)
    
    # 3. Guardar sistema
    import joblib
    joblib.dump(system, output_path)
    
    # 4. Devolver métricas finales
    return metrics, system

# Ejecución si se ejecuta como script
if __name__ == "__main__":
    print("=== SISTEMA OPTIMIZADO DE CLASIFICACIÓN DE ENEAGRAMA ===")
    
    # 1. Cargar datos con Supabase como prioridad
    print("\nCargando datos...")
    
    # === LÍNEAS COMENTADAS: CARGA CSV ORIGINAL ===
    # data_path = 'data.csv'
    # metrics, system = run_optimized_enneagram_system(data_path)
    
    # Usar sistema híbrido: Supabase primero, CSV como fallback
    try:
        metrics, system = run_optimized_enneagram_system(
            data_path='data.csv',  # Fallback CSV
            data_source='auto'     # Intentar Supabase primero
        )
    except Exception as e:
        print(f"❌ Error en el entrenamiento: {e}")
        print("🔄 Intentando solo con CSV...")
        metrics, system = run_optimized_enneagram_system(
            data_path='data.csv',
            data_source='csv'      # Solo CSV
        )
    
    # 2. Imprimir resultados finales
    print("\n=== RESULTADOS FINALES ===")
    print(f"Accuracy de Eneatipo: {metrics['type_accuracy']:.4f}")
    print(f"Accuracy de Ala: {metrics['wing_accuracy']:.4f}")
    print("\n¡Proceso completado con éxito!")