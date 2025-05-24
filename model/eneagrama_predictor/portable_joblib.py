# portable_joblib.py
import inspect
import sys
import types
import joblib
import importlib
import warnings
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator

def create_portable_model(model: Any, output_path: str) -> None:
    """
    Creates a portable version of the model by extracting essential attributes
    from OptimizedEnneagramSystem.
    
    Args:
        model: The trained OptimizedEnneagramSystem model
        output_path: Where to save the portable model
    """
    # Extract core model attributes
    portable_dict = {
        'model': model.model,  # The trained keras model
        'feature_names': model.feature_names if hasattr(model, 'feature_names') else None,
        'feature_indices': model.feature_indices if hasattr(model, 'feature_indices') else None,
        'scaler': model.scaler if hasattr(model, 'scaler') else None,
        'selected_features': model.selected_features if hasattr(model, 'selected_features') else None
    }
    
    # Save the portable version
    try:
        joblib.dump(portable_dict, output_path)
        print(f"Portable model saved to: {output_path}")
    except Exception as e:
        print(f"Error saving portable model: {str(e)}")
        raise

def create_portable_model(system, filename):
    """Crea un archivo joblib que contiene el modelo y su código fuente."""
    
    # Obtener el código fuente de los módulos relevantes
    module_code = {}
    
    required_modules = [
        'optimized_enneagram_system', 
        'data_preprocessing', 
        'data_augmentation',
        'explainability'  # Added explainability module
    ]
    
    for module_name in required_modules:
        try:
            # Intentar importar el módulo si no está ya importado
            if module_name not in sys.modules:
                importlib.import_module(module_name)
                
            module = sys.modules.get(module_name)
            if module:
                # Obtener el código fuente completo del módulo
                module_code[module_name] = inspect.getsource(module)
                # También guardar la ruta del archivo para referencia
                module_code[f"{module_name}_file"] = inspect.getfile(module)
            else:
                warnings.warn(f"No se pudo encontrar el módulo {module_name}")
        except Exception as e:
            warnings.warn(f"Error al procesar el módulo {module_name}: {e}")
    
    # Crear un paquete portátil con el sistema y el código fuente
    portable_package = {
        'system': system,
        'module_code': module_code,
        'version': '1.0'
    }
    
    # Guardar todo en un solo archivo
    joblib.dump(portable_package, filename, compress=3)
    
    print(f"Modelo portable guardado en {filename}")
    print(f"Módulos encapsulados: {', '.join(module_name for module_name in module_code if not module_name.endswith('_file'))}")
    return filename

def load_portable_model(filename):
    """Carga un modelo portable incluyendo sus dependencias."""
    
    # Cargar el paquete
    print(f"Cargando modelo portable desde {filename}...")
    package = joblib.load(filename)
    
    if 'version' not in package or 'module_code' not in package or 'system' not in package:
        raise ValueError("El archivo no parece contener un modelo portable válido")
    
    # Extraer el código fuente y crear módulos dinámicamente
    for module_name, source_code in package['module_code'].items():
        # Saltar las entradas de rutas de archivo
        if module_name.endswith('_file'):
            continue
            
        print(f"Cargando módulo: {module_name}")
        
        # Crear un módulo dinámico en memoria
        mod = types.ModuleType(module_name)
        
        # Añadir algunas propiedades estándar al módulo
        mod.__file__ = package['module_code'].get(f"{module_name}_file", f"<{module_name}>")
        mod.__name__ = module_name
        mod.__package__ = None
        
        # Compilar y ejecutar el código fuente
        try:
            compiled_code = compile(source_code, mod.__file__, 'exec')
            exec(compiled_code, mod.__dict__)
            
            # Registrar el módulo en sys.modules
            sys.modules[module_name] = mod
            
            print(f"Módulo {module_name} cargado correctamente")
        except Exception as e:
            print(f"Error al cargar el módulo {module_name}: {e}")
            raise
    
    # Ahora que los módulos están disponibles, devolver el sistema
    print("Modelo cargado correctamente")
    return package['system']