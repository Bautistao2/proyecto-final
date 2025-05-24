# data_preprocessing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

class DataPreprocessor:
    """Clase para el preprocesamiento avanzado de datos"""
    
    def __init__(self, n_components: float = 0.95, feature_threshold: float = 0.25):
        self.scaler = RobustScaler(quantile_range=(5, 95))
        self.pca = PCA(n_components=n_components, random_state=42)
        self.feature_threshold = feature_threshold
        self.selected_features = None
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Procesamiento completo de los datos"""
        
        # Verificar si X es DataFrame o array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Escalado robusto
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Selección de características basada en información mutua
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
        self.selected_features = mi_scores > np.percentile(mi_scores, self.feature_threshold * 100)
        
        # Verificar que haya al menos una característica seleccionada
        if not any(self.selected_features):
            # Si ninguna característica supera el umbral, seleccionar las mejores 10
            top_indices = np.argsort(mi_scores)[-10:]
            self.selected_features = np.zeros_like(mi_scores, dtype=bool)
            self.selected_features[top_indices] = True
            
        X_selected = X_scaled[:, self.selected_features]
        
        # Reducción de dimensionalidad con PCA
        X_pca = self.pca.fit_transform(X_selected)
        
        return X_pca
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Aplicar transformación a nuevos datos"""
        # Verificar si X es DataFrame o array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
            
        # Aplicar transformaciones
        X_scaled = self.scaler.transform(X_array)
        
        # Verificar que selected_features esté inicializado
        if self.selected_features is None:
            raise ValueError("El método fit_transform debe ser llamado antes que transform")
            
        X_selected = X_scaled[:, self.selected_features]
        return self.pca.transform(X_selected)
    
    def get_feature_importance(self, feature_names=None):
        """Devuelve la importancia de cada característica original"""
        if self.selected_features is None:
            raise ValueError("El método fit_transform debe ser llamado primero")
            
        # Si no se proporcionaron nombres, usar índices
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.selected_features))]
            
        # Crear un diccionario {nombre_característica: importancia}
        feature_importance = {}
        for i, (name, selected) in enumerate(zip(feature_names, self.selected_features)):
            # Solo incluir características seleccionadas
            if selected:
                # Calcular importancia como contribución a componentes principales
                importance = np.sum(np.abs(self.pca.components_[:, np.where(self.selected_features)[0] == i]))
                feature_importance[name] = importance
                
        # Normalizar valores
        max_imp = max(feature_importance.values()) if feature_importance else 1.0
        for key in feature_importance:
            feature_importance[key] /= max_imp
            
        return feature_importance
    
    def get_variance_explained(self):
        """Devuelve la varianza explicada por los componentes principales"""
        if not hasattr(self.pca, 'explained_variance_ratio_'):
            raise ValueError("El método fit_transform debe ser llamado primero")
            
        return {
            'variance_ratio': self.pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca.explained_variance_ratio_),
            'n_components': self.pca.n_components_
        }