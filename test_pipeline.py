import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

print("=== VERIFICACIÓN DEL ENTORNO DE TRABAJO ===")
print(f"[OK] Pandas versión: {pd.__version__}")
print(f"[OK] NumPy versión: {np.__version__}")

# Creación de DataFrame sintético de prueba
data = pd.DataFrame({
    'edad': [25, 45, 35, 50, 23],
    'ingresos': [50000, 80000, 62000, 45000, 31000],
    'riesgo': [0, 0, 1, 0, 1]
})

X = data[['edad', 'ingresos']]
y = data['riesgo']

clf = RandomForestClassifier(n_estimators=10)
clf.fit(X, y)

print("[OK] Modelo Scikit-Learn entrenado exitosamente.")
print("=== ENTORNO LISTO PARA MINERÍA DE DATOS ===")

# --- Prueba de conexión a MySQL (descomenta y ajusta credenciales en .env) ---
# from src.database.connection import MySQLConnector
#
# db = MySQLConnector()
# df = db.extract_data("SELECT * FROM tu_tabla LIMIT 10;")
# print(df.head())
