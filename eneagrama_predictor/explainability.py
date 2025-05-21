#explainability.py
import numpy as np
import pandas as pd
import shap
import tensorflow as tf

class AdvancedEnneagramExplainer:
    """Sistema avanzado de explicabilidad para modelos de eneagrama"""
    
    def __init__(self, system):
        self.system = system
        self.feature_importance = None
        self.type_descriptions = self._load_type_descriptions()
        self.wing_combinations = self._load_wing_combinations()
        
    def _load_type_descriptions(self):
        """Carga descripciones de los tipos de eneagrama"""
        # Descripciones simplificadas de los tipos
        return {
            1: "Perfeccionista - Motivados por hacer lo correcto, ser buenos y mejorar las cosas.",
            2: "Ayudador - Motivados por el deseo de ser amados y apreciados, y expresar sentimientos positivos hacia los demás.",
            3: "Triunfador - Motivados por el deseo de sentirse valiosos, aceptados y distinguidos.",
            4: "Individualista - Motivados por el deseo de expresar su individualidad y ser auténticos.",
            5: "Investigador - Motivados por el deseo de entender el mundo, ser capaces y competentes.",
            6: "Leal - Motivados por la búsqueda de seguridad y apoyo.",
            7: "Entusiasta - Motivados por el deseo de sentirse felices, planificar experiencias gratificantes y mantener opciones abiertas.",
            8: "Desafiador - Motivados por el deseo de ser fuertes y evitar la vulnerabilidad.",
            9: "Pacificador - Motivados por el deseo de mantener la paz interior y exterior."
        }
    
    def _load_wing_combinations(self):
        """Carga descripciones de combinaciones tipo-ala"""
        return {
            (1, 9): "1w9: Perfeccionista sereno - Idealistas, controlados, objetivos y analíticos.",
            (1, 2): "1w2: Perfeccionista servicial - Perfeccionistas más cálidos y orientados a ayudar.",
            (2, 1): "2w1: Ayudador servidor - Deseo de ayudar guiado por principios éticos.",
            (2, 3): "2w3: Ayudador anfitrión - Más asertivos y centrados en la imagen.",
            (3, 2): "3w2: Triunfador encantador - Triunfadores más orientados a personas.",
            (3, 4): "3w4: Triunfador profesional - Más introspectivos y creativos.",
            (4, 3): "4w3: Individualista ambicioso - Más orientados al éxito exterior.",
            (4, 5): "4w5: Individualista bohemio - Más reservados, intelectuales y observadores.",
            (5, 4): "5w4: Investigador innovador - Más creativos y conectados con emociones.",
            (5, 6): "5w6: Investigador problemático - Más precavidos y sistemáticos.",
            (6, 5): "6w5: Leal fóbico - Más orientados a la intelectualidad y la seguridad.",
            (6, 7): "6w7: Leal contrafóbico - Más excitables y orientados a la distracción.",
            (7, 6): "7w6: Entusiasta planificador - Más conectados con su ansiedad y necesidades de seguridad.",
            (7, 8): "7w8: Entusiasta autoritario - Más asertivos y orientados al poder.",
            (8, 7): "8w7: Desafiador entusiasta - Líderes más optimistas y versátiles.",
            (8, 9): "8w9: Desafiador diplomático - Más tranquilos y resistentes.",
            (9, 8): "9w8: Pacificador defensor - Más asertivos cuando es necesario.",
            (9, 1): "9w1: Pacificador idealista - Mayor sentido de corrección y orden."
        }
    
    def explain_prediction(self, X, feature_names=None):
        """Explica la predicción usando SHAP values"""
        try:
            # Función de predicción para SHAP
            def model_predict(x):
                if isinstance(x, np.ndarray):
                    x = pd.DataFrame(x, columns=X.columns)
                return self.ensemble_predict(x)['eneatipo_probs']
            
            # Ajustar número de clusters al tamaño de datos
            n_clusters = min(10, len(X))
            background = shap.kmeans(X, n_clusters) if len(X) > 1 else X
            
            # Crear explainer SHAP
            explainer = shap.KernelExplainer(model_predict, background)
            
            # Calcular SHAP values con menos muestras para eficiencia
            shap_values = explainer.shap_values(X, nsamples=100)
            
            # Procesar feature importance
            if isinstance(shap_values, list):
                feature_importance = np.abs(np.array(shap_values)).mean(axis=1).mean(axis=0)
            else:
                feature_importance = np.abs(shap_values).mean(axis=0)
                
            # Obtener features más importantes
            important_features = []
            feature_ranks = np.argsort(-feature_importance)
            
            for idx in feature_ranks[:10]:
                fname = feature_names[idx] if feature_names is not None else f"Feature {idx}"
                importance = feature_importance[idx]
                important_features.append((fname, importance))
            
            return important_features
            
        except Exception as e:
            print(f"Error en explicación SHAP: {str(e)}")
            # Fallback a método más simple de feature importance
            return self._fallback_feature_importance(X, feature_names)

    def _fallback_feature_importance(self, X, feature_names=None):
        """Método alternativo de feature importance cuando SHAP falla"""
        try:
            # Obtener predicción base
            base_pred = self.ensemble_predict(X)
            base_type = base_pred['eneatipo']
            base_probs = base_pred['eneatipo_probs']
            
            # Inicializar array de importancias
            importance = np.zeros(X.shape[1])
            X_perturbed = X.copy()
            
            # Obtener estadísticas del dataset
            feature_means = X.mean()
            feature_stds = X.std()
            
            for i in range(X.shape[1]):
                impact = 0
                original_value = X_perturbed.iloc[0, i]
                
                # Probar diferentes perturbaciones
                perturbations = [
                    feature_means[i],  # Valor medio
                    feature_means[i] + feature_stds[i],  # Media + 1 std
                    feature_means[i] - feature_stds[i]   # Media - 1 std
                ]
                
                for new_value in perturbations:
                    # Aplicar perturbación
                    X_perturbed.iloc[0, i] = new_value
                    
                    # Obtener nueva predicción
                    new_pred = self.ensemble_predict(X_perturbed)
                    new_probs = new_pred['eneatipo_probs']
                    
                    # Calcular impacto como cambio en probabilidades
                    prob_diff = np.sum(np.abs(new_probs - base_probs))
                    impact = max(impact, prob_diff)
                
                # Restaurar valor original
                X_perturbed.iloc[0, i] = original_value
                importance[i] = impact
            
            # Normalizar importancias
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-10)
            
            # Obtener features más importantes
            feature_ranks = np.argsort(-importance)
            important_features = []
            
            for idx in feature_ranks[:10]:
                fname = feature_names[idx] if feature_names is not None else f"Q{idx+1}"
                imp_value = importance[idx]
                important_features.append((fname, imp_value))
            
            return important_features
            
        except Exception as e:
            print(f"Error en cálculo de importancia: {str(e)}")
            return [(f"Q{i+1}", 0.0) for i in range(10)]

    def ensemble_predict(self, X):
        """Realiza predicción usando el ensemble completo"""
        try:
            # Preprocesar datos usando el preprocessor ya entrenado
            X_scaled = self.system.preprocessor.transform(X)
            
            # Inicializar arrays para predicciones
            eneatipo_probs = np.zeros((len(X), 9))
            ala_probs = np.zeros((len(X), 9))
            
            # Obtener predicciones de cada modelo
            for model, weight in zip(self.system.ensemble['models'], self.system.ensemble['weights']):
                eneatipo_pred, ala_pred = model.predict(X_scaled, verbose=0)
                eneatipo_probs += eneatipo_pred * weight
                ala_probs += ala_pred * weight
            
            # Convertir a predicciones finales (ajustar a rango 1-9)
            eneatipo = np.argmax(eneatipo_probs, axis=1) + 1
            ala = np.argmax(ala_probs, axis=1) + 1
            
            return {
                'eneatipo': eneatipo[0],
                'ala': ala[0],
                'eneatipo_probs': eneatipo_probs[0],
                'ala_probs': ala_probs[0]
            }
        except Exception as e:
            print(f"Error en predicción: {str(e)}")
            raise
    
    def generate_report(self, X):
        """Genera un reporte completo de predicción"""
        # Obtener predicción
        pred = self.ensemble_predict(X)
        eneatipo = pred['eneatipo']
        ala = pred['ala']
        
        # Generar explicación
        feature_names = [f"Q{i+1}" for i in range(X.shape[1])]
        important_features = self.explain_prediction(X, feature_names)
        
        # Construir reporte
        report = [
            f"\nPredicción de Eneatipo: {eneatipo}",
            f"Predicción de Ala: {ala}",
            f"\nDescripción del tipo {eneatipo}:",
            self.type_descriptions[eneatipo],
            f"\nDescripción de la combinación {eneatipo}w{ala}:",
            self.wing_combinations.get((eneatipo, ala), "Combinación no disponible"),
            "\nCaracterísticas más influyentes:",
        ]
        
        for feat, imp in important_features:
            report.append(f"- {feat}: {imp:.4f}")
        
        return "\n".join(report)

