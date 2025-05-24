# data_augmentation.py
import pandas as pd
import numpy as np  
import warnings
warnings.filterwarnings('ignore')

class EnneagramDataAugmentor:
    """Clase para aumentar datos de tipos poco representados"""
    
    def __init__(self, min_samples: int = 100):
        self.min_samples = min_samples
        
    def augment_data(self, X: pd.DataFrame, y_type: pd.Series, y_wing: pd.Series = None):
        """Genera datos sintéticos para balancear clases"""
        # Cuenta de muestras por tipo
        type_counts = y_type.value_counts()
        
        # Datos originales
        X_augmented = X.copy()
        y_type_augmented = y_type.copy()
        y_wing_augmented = y_wing.copy() if y_wing is not None else None
        
        # Para cada tipo con pocas muestras
        for eneatype, count in type_counts.items():
            if count < self.min_samples:
                # Número de muestras a generar
                n_samples_to_generate = self.min_samples - count
                
                # Índices de muestras de este tipo
                indices = y_type[y_type == eneatype].index
                
                # Técnica 1: SMOTE-like (promedios con ruido)
                for _ in range(n_samples_to_generate):
                    # Seleccionar dos muestras aleatorias del mismo tipo
                    idx1, idx2 = np.random.choice(indices, 2, replace=True)
                    
                    # Crear nueva muestra como promedio + ruido
                    new_sample = (X.loc[idx1] + X.loc[idx2]) / 2
                    
                    # Añadir ruido gaussiano proporcional
                    noise = np.random.normal(0, 0.05, size=len(new_sample))
                    new_sample += noise * new_sample
                    
                    # Limitar valores a rango válido (suponiendo test 1-5)
                    new_sample = np.clip(new_sample, 1, 5)
                    
                    # Añadir nueva muestra
                    X_augmented = pd.concat([X_augmented, pd.DataFrame([new_sample], columns=X.columns)])
                    y_type_augmented = pd.concat([y_type_augmented, pd.Series([eneatype])])
                    
                    # Si hay datos de ala
                    if y_wing is not None:
                        # Usar el ala de una de las muestras originales
                        wing = y_wing.loc[idx1]
                        y_wing_augmented = pd.concat([y_wing_augmented, pd.Series([wing])])
        
        # Resetear índices
        X_augmented.reset_index(drop=True, inplace=True)
        y_type_augmented.reset_index(drop=True, inplace=True)
        if y_wing is not None:
            y_wing_augmented.reset_index(drop=True, inplace=True)
        
        print(f"Datos originales: {len(X)} muestras")
        print(f"Datos aumentados: {len(X_augmented)} muestras")
        
        return X_augmented, y_type_augmented, y_wing_augmented
    
    def augment_rare_wings(self, X: pd.DataFrame, y_type: pd.Series, y_wing: pd.Series, 
                          wing_min_samples: int = 20):
        """Aumenta datos para combinaciones tipo-ala poco frecuentes"""
        # Contar combinaciones tipo-ala
        type_wing_counts = pd.crosstab(y_type, y_wing)
        
        # Datos aumentados
        X_augmented = X.copy()
        y_type_augmented = y_type.copy() 
        y_wing_augmented = y_wing.copy()
        
        # Para cada tipo
        for eneatype in range(1, 10):
            # Para cada posible ala de este tipo
            for wing in self._get_possible_wings(eneatype):
                # Si hay pocas muestras de esta combinación
                try:
                    if type_wing_counts.loc[eneatype, wing] < wing_min_samples:
                        # Muestras a generar
                        n_to_generate = wing_min_samples - type_wing_counts.loc[eneatype, wing]
                        
                        # Índices de esta combinación
                        indices = y_type[(y_type == eneatype) & (y_wing == wing)].index
                        
                        # Si no hay muestras de esta combinación, usar muestras del tipo
                        if len(indices) == 0:
                            indices = y_type[y_type == eneatype].index
                        
                        # Generar muestras
                        for _ in range(n_to_generate):
                            if len(indices) > 0:
                                # Seleccionar muestra al azar
                                idx = np.random.choice(indices)
                                
                                # Copiar con modificaciones leves
                                new_sample = X.loc[idx].copy()
                                
                                # Modificar aleatoriamente 10% de las respuestas
                                indices_to_modify = np.random.choice(
                                    len(new_sample), 
                                    size=int(0.1 * len(new_sample)), 
                                    replace=False
                                )
                                
                                for i in indices_to_modify:
                                    # Añadir perturbación leve
                                    new_sample[i] += np.random.choice([-0.5, 0, 0.5])
                                    
                                # Limitar valores
                                new_sample = np.clip(new_sample, 1, 5)
                                
                                # Añadir nueva muestra
                                X_augmented = pd.concat([X_augmented, pd.DataFrame([new_sample], columns=X.columns)])
                                y_type_augmented = pd.concat([y_type_augmented, pd.Series([eneatype])])
                                y_wing_augmented = pd.concat([y_wing_augmented, pd.Series([wing])])
                except KeyError:
                    # Si no existe la combinación en el crosstab
                    continue
                    
        # Resetear índices
        X_augmented.reset_index(drop=True, inplace=True)
        y_type_augmented.reset_index(drop=True, inplace=True)
        y_wing_augmented.reset_index(drop=True, inplace=True)
        
        print(f"Datos originales: {len(X)} muestras")
        print(f"Datos aumentados con balance de alas: {len(X_augmented)} muestras")
        
        return X_augmented, y_type_augmented, y_wing_augmented
    
    def _get_possible_wings(self, eneatype: int) -> list:
        """Devuelve las alas posibles para un eneatipo"""
        if eneatype == 9:
            return [8, 1]
        elif eneatype == 1:
            return [9, 2]
        else:
            return [eneatype - 1, eneatype + 1]