"""
Script para cargar las tablas clubs y competitions (necesarias para la
Practica de Clase Nº 2) en la base de datos MySQL local (XAMPP).

Uso:
    python src/database/load_data_clase2.py
"""

import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

user = os.getenv("DB_USER", "root")
password = os.getenv("DB_PASSWORD", "")
host = os.getenv("DB_HOST", "localhost")
port = os.getenv("DB_PORT", "3306")
db_name = os.getenv("DB_NAME", "")

connection_string = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}"
engine = create_engine(connection_string)

CLUBS_PATH = "data/raw/clubs.csv"
COMPETITIONS_PATH = "data/raw/competitions.csv"


def cargar_clubs():
    print("Leyendo clubs.csv...")
    df = pd.read_csv(CLUBS_PATH)
    print(f"  {len(df)} filas leídas.")
    print("Cargando tabla 'clubs' en MySQL...")
    df.to_sql("clubs", con=engine, if_exists="replace", index=False, chunksize=1000)
    print("[OK] Tabla 'clubs' cargada exitosamente.")


def cargar_competitions():
    print("Leyendo competitions.csv...")
    df = pd.read_csv(COMPETITIONS_PATH)
    print(f"  {len(df)} filas leídas.")
    print("Cargando tabla 'competitions' en MySQL...")
    df.to_sql("competitions", con=engine, if_exists="replace", index=False, chunksize=1000)
    print("[OK] Tabla 'competitions' cargada exitosamente.")


if __name__ == "__main__":
    print("=== CARGA DE TABLAS PARA LA PRÁCTICA 2 ===")
    cargar_clubs()
    cargar_competitions()
    print("=== CARGA COMPLETA ===")
