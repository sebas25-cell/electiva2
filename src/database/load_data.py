"""
Script para cargar los datasets de Transfermarkt (players y player_valuations)
en la base de datos MySQL local (XAMPP), usando las credenciales del .env.

Uso:
    python src/database/load_data.py
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

# Rutas de los CSV (ajusta si los guardaste en otro lugar)
PLAYERS_PATH = "data/raw/players.csv"
VALUATIONS_PATH = "data/raw/player_valuations.csv"

def cargar_players():
    print("Leyendo players.csv...")
    df = pd.read_csv(PLAYERS_PATH)
    print(f"  {len(df)} filas leídas.")
    print("Cargando tabla 'players' en MySQL...")
    df.to_sql("players", con=engine, if_exists="replace", index=False, chunksize=1000)
    print("[OK] Tabla 'players' cargada exitosamente.")

def cargar_player_valuations():
    print("Leyendo player_valuations.csv...")
    df = pd.read_csv(VALUATIONS_PATH)
    print(f"  {len(df)} filas leídas.")
    print("Cargando tabla 'player_valuations' en MySQL (puede tardar un poco, son ~656k filas)...")
    df.to_sql("player_valuations", con=engine, if_exists="replace", index=False, chunksize=5000)
    print("[OK] Tabla 'player_valuations' cargada exitosamente.")

if __name__ == "__main__":
    print("=== CARGA DE DATOS A MYSQL ===")
    cargar_players()
    cargar_player_valuations()
    print("=== CARGA COMPLETA ===")
