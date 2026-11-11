import os
import pandas as pd
import matplotlib.pyplot as plt
from src.database.connection import MySQLConnector


def main():
    print("=================================================================")
    print(" MINERÍA DE DATOS - PRÁCTICA 2: INTEGRACIÓN Y ANÁLISIS MULTITABLA")
    print("=================================================================\n")

    # 1. Instanciar conector
    db = MySQLConnector()

    # 2. Consulta SQL con Múltiples JOINs y Cálculos en SGBD
    query = """
    SELECT
        p.player_id,
        p.name AS jugador,
        p.position AS posicion,
        p.height_in_cm AS estatura_cm,
        TIMESTAMPDIFF(YEAR, p.date_of_birth, CURDATE()) AS edad,
        p.market_value_in_eur AS valor_mercado,
        c.name AS club,
        comp.name AS liga,
        comp.type AS tipo_liga
    FROM players p
    INNER JOIN clubs c ON p.current_club_id = c.club_id
    INNER JOIN competitions comp ON c.domestic_competition_id = comp.competition_id
    WHERE p.market_value_in_eur IS NOT NULL
      AND p.date_of_birth IS NOT NULL;
    """

    print("[1] Ejecutando consulta relacional compleja...")
    df = db.extract_data(query)

    if df.empty:
        print("[AVISO] La consulta no devolvió datos. Revisa la base de datos.")
        return

    print(f"[OK] Datos cargados correctamente. Filas extraídas: {len(df):,}\n")

    # 3. LIMPIEZA Y PREPROCESAMIENTO EN PANDAS
    print("[2] Aplicando técnicas de preprocesamiento de datos...")

    # Imputación de valores faltantes en la estatura con la media
    estatura_media = df['estatura_cm'].mean()
    df['estatura_cm'] = df['estatura_cm'].fillna(estatura_media)

    # Conversión explícita a tipos numéricos
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')

    # 4. FEATURE ENGINEERING AVANZADO: Clasificación por Nivel Comercial
    def clasificar_perfil_comercial(valor):
        if valor >= 50_000_000:
            return '1. Elite Mundial'
        elif valor >= 15_000_000:
            return '2. Competitivo Top'
        elif valor >= 3_000_000:
            return '3. Promedio Liga'
        else:
            return '4. Desarrollo / Bajo Costo'

    df['perfil_comercial'] = df['valor_mercado'].apply(clasificar_perfil_comercial)

    # 5. AGRUPAMIENTOS MULTIDIMENSIONALES (PIVOT TABLE)
    print("\n--- VALOR PROMEDIO POR LIGA Y TIPO DE JUGADOR (€) ---")
    tabla_pivote = df.pivot_table(
        index='liga',
        columns='perfil_comercial',
        values='player_id',
        aggfunc='count',
        fill_value=0
    )
    print(tabla_pivote.head(10))

    # 6. IDENTIFICAR TOP 5 CLUBES CON MAYOR VALOR DE PLANTILLA
    print("\n--- TOP 5 CLUBES MÁS VALIOSOS DEL DATASET ---")
    top_clubes = (
        df.groupby('club')
        .agg(
            valor_total=('valor_mercado', 'sum'),
            promedio_jugador=('valor_mercado', 'mean'),
            plantilla=('player_id', 'count')
        )
        .reset_index()
        .sort_values(by='valor_total', ascending=False)
        .head(5)
    )
    top_clubes['valor_total_formato'] = top_clubes['valor_total'].apply(lambda x: f"€{x:,.2f}")
    top_clubes['promedio_formato'] = top_clubes['promedio_jugador'].apply(lambda x: f"€{x:,.2f}")

    print(top_clubes[['club', 'plantilla', 'valor_total_formato', 'promedio_formato']].to_string(index=False))

    # 7. GENERAR Y GUARDAR GRÁFICO AUTOMÁTICO
    print("\n[3] Generando gráfico de distribución de valores por posición...")
    plt.figure(figsize=(10, 5))
    df_pos = df.groupby('posicion')['valor_mercado'].mean().sort_values(ascending=False).head(8)
    df_pos.plot(kind='barh', color='darkblue')
    plt.title('Top 8 Posiciones más Valiosas en el Mercado (€)')
    plt.xlabel('Valor Promedio (Euros)')
    plt.ylabel('Posición')
    plt.tight_layout()

    # Guardar gráfico en carpeta de resultados
    os.makedirs("data/processed", exist_ok=True)
    ruta_grafico = "data/processed/valor_por_posicion.png"
    plt.savefig(ruta_grafico)
    print(f"[OK] Gráfico guardado en: {ruta_grafico}")

    # -----------------------------------------------------------------
    # RETO 1: Filtro Avanzado (> 185 cm y < 23 años)
    # -----------------------------------------------------------------
    df_filtrado = df[(df['estatura_cm'] > 185) & (df['edad'] < 23)]
    print(f"\n--- RETO 1: JUGADORES PROMESA Y ALTOS (> 185 cm, < 23 AÑOS): {len(df_filtrado)} ENCONTRADOS ---")
    print(df_filtrado[['jugador', 'posicion', 'estatura_cm', 'edad', 'club']].head(10).to_string(index=False))

    # -----------------------------------------------------------------
    # RETO 2: Liga con más jugadores '1. Elite Mundial'
    # -----------------------------------------------------------------
    df_elite = df[df['perfil_comercial'] == '1. Elite Mundial']
    top_ligas_elite = (
        df_elite.groupby('liga')['player_id']
        .count()
        .reset_index()
        .rename(columns={'player_id': 'total_jugadores_elite'})
        .sort_values(by='total_jugadores_elite', ascending=False)
    )
    print("\n--- RETO 2: LIGAS CON MAYOR CANTIDAD DE JUGADORES 'ÉLITE MUNDIAL' ---")
    print(top_ligas_elite.to_string(index=False))

    # -----------------------------------------------------------------
    # RETO 3: Exportar la tabla pivote a Excel
    # -----------------------------------------------------------------
    ruta_excel = "data/processed/reporte_ligas.xlsx"
    tabla_pivote.to_excel(ruta_excel)
    print(f"\n--- RETO 3: [OK] Tabla pivote exportada exitosamente a: {ruta_excel} ---")


if __name__ == "__main__":
    main()
