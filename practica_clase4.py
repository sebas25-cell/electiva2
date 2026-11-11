import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Módulos de Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Conector personalizado (corregido: connection, no connection2)
from src.database.connection import MySQLConnector


def main():
    print("=================================================================")
    print("  MINERÍA DE DATOS - PRÁCTICA 4: CLASIFICACIÓN Y MATRIZ DE CONFUSIÓN")
    print("=================================================================\n")

    # -----------------------------------------------------------------
    # 1. INGESTA Y EXTRACCIÓN DE DATOS DESDE MYSQL
    # -----------------------------------------------------------------
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

    print("[1] Extrayendo datos desde la base de datos relacional...")
    df = db.extract_data(query)

    if df.empty:
        print("[ERROR] La consulta no devolvió datos.")
        return

    print(f"[OK] Registros extraídos: {len(df):,}\n")

    # -----------------------------------------------------------------
    # 2. PREPROCESAMIENTO Y DEFINICIÓN DE LA VARIABLE TARGET (CLASE)
    # -----------------------------------------------------------------
    print("[2] Ejecutando preprocesamiento y creando la variable binaria 'Es_Elite'...")

    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')

    # Imputación de estatura por mediana de posición
    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(lambda x: x.fillna(x.median()))
    df['estatura_cm'] = df['estatura_cm'].fillna(df['estatura_cm'].median())

    # DEFINICIÓN DE LA CLASE OBJETIVO (TARGET):
    # Definimos como 'Élite' (1) a los jugadores en el top 25% de valor de mercado (> Q3)
    umbral_elite = df['valor_mercado'].quantile(0.75)
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    conteo_clases = df['es_elite'].value_counts()
    print(f"  • Umbral para ser considerado Élite: €{umbral_elite:,.2f}")
    print(f"  • Distribución de clases -> Estándar (0): {conteo_clases[0]:,} | Élite (1): {conteo_clases[1]:,}\n")

    # -----------------------------------------------------------------
    # 3. FEATURE ENGINEERING (MATRIZ X Y VECTOR Y)
    # -----------------------------------------------------------------
    print("[3] Generando variables explicativas (One-Hot Encoding)...")

    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)

    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']

    # División del dataset (80% Entrenamiento, 20% Prueba con Estratificación)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------------------------------------------
    # 4. MODELADO Y COMPARACIÓN DE ALGORITMOS
    # -----------------------------------------------------------------
    print("[4] Entrenando Regresión Logística y Árbol de Decisión...\n")

    # A. Modelo 1: Regresión Logística
    modelo_log = LogisticRegression(max_iter=1000, random_state=42)
    modelo_log.fit(X_train, y_train)
    y_pred_log = modelo_log.predict(X_test)

    # B. Modelo 2: Árbol de Decisión (Limitamos la profundidad a 3 para evitar overfitting)
    modelo_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    modelo_tree.fit(X_train, y_train)
    y_pred_tree = modelo_tree.predict(X_test)

    # -----------------------------------------------------------------
    # 5. EVALUACIÓN Y TABLA COMPARATIVA
    # -----------------------------------------------------------------
    def calcular_metricas(y_real, y_pred):
        return {
            'Accuracy': accuracy_score(y_real, y_pred),
            'Precision': precision_score(y_real, y_pred, zero_division=0),
            'Recall': recall_score(y_real, y_pred, zero_division=0),
            'F1-Score': f1_score(y_real, y_pred, zero_division=0)
        }

    met_log = calcular_metricas(y_test, y_pred_log)
    met_tree = calcular_metricas(y_test, y_pred_tree)

    df_comparativa = pd.DataFrame([met_log, met_tree], index=['Regresión Logística', 'Árbol de Decisión'])

    print("=================================================================")
    print("           TABLA COMPARATIVA DE RENDIMIENTO DE MODELOS           ")
    print("=================================================================")
    print(df_comparativa.round(4).to_string())
    print("=================================================================\n")

    # -----------------------------------------------------------------
    # 6. VISUALIZACIÓN: MATRICES DE CONFUSIÓN Y ÁRBOL DE DECISIÓN
    # -----------------------------------------------------------------
    print("[5] Generando panel gráfico de evaluación...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('EVALUACIÓN DE MODELOS DE CLASIFICACIÓN - JUGADORES ÉLITE', fontsize=14, fontweight='bold')

    # Gráfico 1: Matriz de Confusión - Regresión Logística
    cm_log = confusion_matrix(y_test, y_pred_log)
    sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Estándar (0)', 'Élite (1)'],
                yticklabels=['Estándar (0)', 'Élite (1)'])
    axes[0].set_title('Matriz de Confusión: Regresión Logística')
    axes[0].set_xlabel('Predicción del Modelo')
    axes[0].set_ylabel('Clase Real')

    # Gráfico 2: Matriz de Confusión - Árbol de Decisión
    cm_tree = confusion_matrix(y_test, y_pred_tree)
    sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=['Estándar (0)', 'Élite (1)'],
                yticklabels=['Estándar (0)', 'Élite (1)'])
    axes[1].set_title('Matriz de Confusión: Árbol de Decisión')
    axes[1].set_xlabel('Predicción del Modelo')
    axes[1].set_ylabel('Clase Real')

    # Gráfico 3: Estructura del Árbol de Decisión
    plot_tree(modelo_tree, feature_names=list(X.columns), class_names=['Estándar', 'Élite'],
              filled=True, ax=axes[2], fontsize=8)
    axes[2].set_title('Estructura Reglas del Árbol de Decisión')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Exportar imagen
    directorio = "data/processed"
    os.makedirs(directorio, exist_ok=True)
    ruta_grafico = os.path.join(directorio, "evaluacion_clasificacion_clase4.png")
    plt.savefig(ruta_grafico, dpi=300)
    plt.show()

    print(f"[OK] Panel gráfico guardado exitosamente en: {ruta_grafico}")


if __name__ == "__main__":
    main()
