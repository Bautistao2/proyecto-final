import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self):
        self.supabase: Optional[Client] = None
        self.table_name = os.getenv("TEST_RESULTS_TABLE", "test_results")  # Configurable desde .env
        self._initialize_supabase()
    
    def _initialize_supabase(self):
        """Inicializa conexión con Supabase"""
        if not SUPABASE_AVAILABLE:
            logger.warning("⚠️ Supabase no disponible")
            return
        
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_KEY")
            
            if supabase_url and supabase_key:
                self.supabase = create_client(supabase_url, supabase_key)
                logger.info("✅ Conexión a Supabase establecida")
            else:
                logger.warning("⚠️ Credenciales de Supabase no encontradas")
        except Exception as e:
            logger.error(f"❌ Error conectando a Supabase: {e}")
    
    async def save_test_result(self, 
                              answers: List[int],  # Lista de 80 respuestas
                              eneatipo: int, 
                              ala: int) -> Optional[str]:
        """
        Guarda resultado del test en la base de datos
        
        Args:
            answers: Lista de 80 respuestas (valores 1-5)
            eneatipo: Eneatipo predicho (1-9)
            ala: Ala predicha (1-9)
        
        Returns:
            session_id del registro guardado o None si falló
        """
        if not self.supabase:
            logger.warning("⚠️ Supabase no disponible, no se guardará el resultado")
            return None
        
        try:
            # Generar ID único de sesión
            session_id = str(uuid.uuid4())
            
            # Preparar datos - convertir lista de respuestas a columnas individuales
            result_data = {
                "session_id": session_id,
                "eneatipo_predicho": eneatipo,
                "ala_predicha": ala
            }
            
            # Añadir cada respuesta como columna individual (q1, q2, ..., q80)
            for i, answer in enumerate(answers[:80], 1):  # Asegurar máximo 80 preguntas
                result_data[f"q{i}"] = int(answer)
            
            # Insertar en Supabase
            response = self.supabase.table(self.table_name).insert(result_data).execute()
            
            if response.data:
                logger.info(f"✅ Resultado guardado con session_id: {session_id}")
                return session_id
            else:
                logger.error("❌ No se pudo guardar el resultado")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error guardando resultado: {e}")
            return None
    
    async def get_test_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas básicas de tests realizados"""
        if not self.supabase:
            return {"error": "Base de datos no disponible"}
        
        try:
            # Contar total de tests
            response = self.supabase.table(self.table_name).select("eneatipo_predicho").execute()
            
            if response.data:
                total_tests = len(response.data)
                
                # Contar distribución por eneatipos
                eneatipo_distribution = {}
                for row in response.data:
                    tipo = row["eneatipo_predicho"]
                    eneatipo_distribution[tipo] = eneatipo_distribution.get(tipo, 0) + 1
                
                return {
                    "total_tests": total_tests,
                    "eneatipo_distribution": eneatipo_distribution,
                    "last_updated": datetime.now().isoformat()
                }
            else:
                return {"total_tests": 0, "eneatipo_distribution": {}}
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo estadísticas: {e}")
            return {"error": str(e)}

# Instancia global del servicio
db_service = DatabaseService()
