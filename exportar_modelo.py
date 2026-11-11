import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Conector personalizado a la base de datos (corregido: connection, no connection2)
from src.database.connection import MySQLConnector


def main():
    print("=================================================================")
    print("  FASE 1: ENTRENAMIENTO Y PERSISTENCIA DEL MODELO (DEPLOYMENT)")
    print("=================================================================\n")

    db = MySQLConnector()

    query = """
        SELECT
            p.player_id,
            p.name AS jugador,
            p.position AS posicion,
            p.height_in_cm AS estatura_cm,
            TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
            p.market_value_in_eur AS valor_mercado
        FROM players p
        WHERE p.market_value_in_eur IS NOT NULL
          AND p.date_of_birth IS NOT NULL;
    """

    df = db.extract_data(query)

    # Preprocesamiento
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')
    df['posicion'] = df['posicion'].fillna('Sin Posicion')

    # Guardar las medianas de imputación para reutilizar en producción
    medianas_estatura = df.groupby('posicion')['estatura_cm'].median().to_dict()
    mediana_global_estatura = df['estatura_cm'].median()
    mediana_global_edad = df['edad'].median()

    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(lambda x: x.fillna(x.median()))
    df['estatura_cm'] = df['estatura_cm'].fillna(mediana_global_estatura)
    df['edad'] = df['edad'].fillna(mediana_global_edad)

    umbral_elite = df['valor_mercado'].quantile(0.75)
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)
    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']

    # Entrenar modelo final
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=4, class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf_model.fit(X, y)

    # Crear paquete de despliegue
    paquete_despliegue = {
        'modelo': rf_model,
        'columnas_X': X.columns.tolist(),
        'medianas_estatura': medianas_estatura,
        'mediana_global_estatura': mediana_global_estatura,
        'mediana_global_edad': mediana_global_edad,
        'umbral_elite_eur': umbral_elite
    }

    os.makedirs('models', exist_ok=True)
    ruta_modelo = 'models/modelo_scouting_rf.joblib'
    joblib.dump(paquete_despliegue, ruta_modelo)

    print(f"Paquete de modelo exportado exitosamente en: '{ruta_modelo}'")


if __name__ == "__main__":
    main()
