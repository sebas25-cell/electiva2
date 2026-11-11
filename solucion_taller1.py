import os
import pandas as pd
import matplotlib.pyplot as plt
from src.database.connection import MySQLConnector


def main():
    print("=" * 70)
    print("TALLER EVALUATIVO N° 1 - MINERÍA DE DATOS")
    print("INTEGRACIÓN Y PREPROCESAMIENTO DE DATOS DE FÚTBOL")
    print("=" * 70)

    # ---------------------------------------------------------------
    # PUNTO 2: EXTRACCIÓN E INTEGRACIÓN MULTITABLA
    # ---------------------------------------------------------------
    db = MySQLConnector()

    query = """
    SELECT
        p.name AS jugador,
        p.position AS posicion,
        p.market_value_in_eur AS valor_mercado,
        TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
        c.name AS club,
        comp.name AS liga
    FROM players p
    INNER JOIN clubs c
        ON p.current_club_id = c.club_id
    INNER JOIN competitions comp
        ON c.domestic_competition_id = comp.competition_id;
    """

    print("\n[1] Ejecutando consulta SQL...")
    df_main = db.extract_data(query)

    if df_main.empty:
        print("[AVISO] La consulta no devolvió datos.")
        return

    print("\n--- PRIMERAS 5 FILAS ---")
    print(df_main.head(5).to_string(index=False))

    print("\n--- SHAPE DEL DATAFRAME ---")
    print(f"Filas: {df_main.shape[0]}")
    print(f"Columnas: {df_main.shape[1]}")
    print(f"Shape total: {df_main.shape}")

    # ---------------------------------------------------------------
    # PUNTO 3: PREPROCESAMIENTO Y LIMPIEZA
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PUNTO 3: PREPROCESAMIENTO Y LIMPIEZA")
    print("=" * 70)

    # 3.1 Conversión de tipos
    df_main["valor_mercado"] = pd.to_numeric(
        df_main["valor_mercado"], errors="coerce"
    )
    df_main["edad"] = pd.to_numeric(
        df_main["edad"], errors="coerce"
    )

    print("\n--- TIPOS DE DATOS ---")
    print(df_main[["valor_mercado", "edad"]].dtypes)

    # 3.2 Identificación de nulos
    print("\n--- VALORES NULOS POR COLUMNA ---")
    print(df_main.isnull().sum())

    # Imputación de edad usando la mediana
    mediana_edad = df_main["edad"].median()
    df_main["edad"] = df_main["edad"].fillna(mediana_edad)

    print(f"\nMediana utilizada para imputar edad: {mediana_edad}")

    # 3.3 Eliminación de filas con valor_mercado nulo
    filas_antes = len(df_main)

    df_main = df_main.dropna(subset=["valor_mercado"]).copy()

    filas_despues = len(df_main)
    filas_eliminadas = filas_antes - filas_despues

    print(f"\nFilas antes del filtrado: {filas_antes}")
    print(f"Filas eliminadas por valor_mercado nulo: {filas_eliminadas}")
    print(f"Filas después del filtrado: {filas_despues}")

    # ---------------------------------------------------------------
    # PUNTO 4: FEATURE ENGINEERING
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PUNTO 4: FEATURE ENGINEERING Y VISUALIZACIÓN")
    print("=" * 70)

    def clasificar_segmento(valor):
        if valor < 1_000_000:
            return "Bajo"
        elif valor < 10_000_000:
            return "Medio"
        else:
            return "Alto"

    df_main["segmento_valor"] = df_main["valor_mercado"].apply(
        clasificar_segmento
    )

    print("\n--- CANTIDAD DE JUGADORES POR SEGMENTO ---")
    conteo_segmentos = (
        df_main["segmento_valor"]
        .value_counts()
        .reindex(["Bajo", "Medio", "Alto"], fill_value=0)
    )
    print(conteo_segmentos)

    # ---------------------------------------------------------------
    # GRÁFICO DE BARRAS
    # ---------------------------------------------------------------
    os.makedirs("data/processed", exist_ok=True)

    plt.figure(figsize=(8, 5))
    conteo_segmentos.plot(kind="bar")
    plt.title("Cantidad de jugadores por segmento de valor")
    plt.xlabel("Segmento de valor")
    plt.ylabel("Cantidad de jugadores")
    plt.xticks(rotation=0)
    plt.tight_layout()

    ruta_grafico = "data/processed/segmentos_valor.png"
    plt.savefig(ruta_grafico, dpi=300)
    plt.close()

    print(f"\n[OK] Gráfico guardado en: {ruta_grafico}")


if __name__ == "__main__":
    main()
