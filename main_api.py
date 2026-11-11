import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Reutiliza la misma función de preprocesamiento e inferencia
# ya construida y probada en inferencia_nuevos_jugadores.py
from inferencia_nuevos_jugadores import preprocesar_y_predecir

# ------------------------------------------------------------------
# 1. Crear la aplicación FastAPI
# ------------------------------------------------------------------
app = FastAPI(title="API de Scouting - Predicción de Jugadores Élite")

# Permite que index.html (abierto como archivo local o en otro puerto)
# pueda hacer peticiones fetch() a esta API sin ser bloqueado por el navegador
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# 2. Definir el "molde" de los datos que la API va a recibir
#    (modelo Pydantic: valida automáticamente el JSON entrante)
# ------------------------------------------------------------------
class Jugador(BaseModel):
    jugador: str
    posicion: str
    estatura_cm: Optional[float] = None  # puede llegar sin estatura
    edad: int


# ------------------------------------------------------------------
# 3. Endpoint raíz, solo para verificar que el servidor está vivo
# ------------------------------------------------------------------
@app.get("/")
def home():
    return {"mensaje": "API de Scouting activa. Usa POST /predict para hacer una predicción."}


# ------------------------------------------------------------------
# 4. Endpoint POST /predict
# ------------------------------------------------------------------
@app.post("/predict")
def predict(jugador: Jugador):
    # 1. Convertir el objeto Pydantic recibido en un DataFrame de una fila
    df_nuevo = pd.DataFrame([jugador.dict()])

    # 2. Aplicar exactamente el mismo preprocesamiento e inferencia
    #    que se usó en el script de simulación
    resultado = preprocesar_y_predecir(df_nuevo)

    # 3. Extraer la predicción y la probabilidad de la primera (única) fila
    fila = resultado.iloc[0]
    prediccion = fila["Prediccion_Clase"]
    probabilidad = float(fila["Probabilidad_Elite"])

    # 4. Responder en JSON
    return {
        "jugador": jugador.jugador,
        "prediccion": prediccion,
        "probabilidad_elite": probabilidad
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_api:app", host="127.0.0.1", port=8000, reload=True)
