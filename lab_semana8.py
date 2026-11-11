import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Módulos de Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)

# Conector personalizado a la base de datos (corregido: connection, no connection2)
from src.database.connection import MySQLConnector


def main():
    print("=================================================================")
    print("  MINERÍA DE DATOS - LAB 6: EVALUACIÓN ROC-AUC Y RANDOM FOREST")
    print("=================================================================\n")

    # -----------------------------------------------------------------
    # FASE 1: INGESTA Y PREPARACIÓN DE DATOS (CRISP-DM)
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

    print("[1] Extrayendo datos desde MySQL...")
    df = db.extract_data(query)

    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')
    df['posicion'] = df['posicion'].fillna('Sin Posicion')

    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(lambda x: x.fillna(x.median()))
    df['estatura_cm'] = df['estatura_cm'].fillna(df['estatura_cm'].median())
    df['edad'] = df['edad'].fillna(df['edad'].median())

    # Target: Élite (Top 25% de mayor valor comercial)
    umbral_elite = df['valor_mercado'].quantile(0.75)
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    # Matriz de variables (X) y vector objetivo (y)
    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)
    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']

    # Depuración Anti-NaN
    filas_con_nan = X.isna().any(axis=1)
    if filas_con_nan.sum() > 0:
        X = X[~filas_con_nan]
        y = y[~filas_con_nan]

    # División 80/20 Estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[OK] Preprocesamiento completado. Registros procesados: {X.shape[0]}.")
    print(f"     Clase 0 (Estándar): {np.sum(y==0)} | Clase 1 (Élite): {np.sum(y==1)}\n")

    # -----------------------------------------------------------------
    # FASE 2: ENTRENAMIENTO DE MODELOS
    # -----------------------------------------------------------------
    print("[2] Entrenando Árbol de Decisión Individual...")

    # Se aplica poda previa y balanceo de costo
    tree_model = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
    tree_model.fit(X_train, y_train)

    print("[3] Entrenando Random Forest (Ensamble de 100 Árboles)...")

    # Se crea el bosque con 100 árboles diversos e independientes
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # -----------------------------------------------------------------
    # FASE 3: EVALUACIÓN DE PROBABILIDADES Y CURVA ROC-AUC
    # -----------------------------------------------------------------
    # Se extraen las probabilidades para la clase Élite (columna 1)
    y_prob_tree = tree_model.predict_proba(X_test)[:, 1]
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

    # Cálculo del Área Bajo la Curva (AUC)
    auc_tree = roc_auc_score(y_test, y_prob_tree)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    print("\n=================================================")
    print("  COMPARATIVA ROC-AUC (CAPACIDAD DE DISCRIMINACIÓN)")
    print("=================================================")
    print(f"  • Árbol de Decisión Único ROC-AUC: {auc_tree:.4f}")
    print(f"  • Random Forest Classifier ROC-AUC: {auc_rf:.4f}")
    print("=================================================\n")

    # Obtención de importancia de variables del Random Forest
    importancias = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)

    print("Top Variables Más Relevantes según Random Forest:")
    print(importancias.head(5))
    print("\n")

    # -----------------------------------------------------------------
    # FASE 4: VISUALIZACIÓN DE RESULTADOS (GRÁFICA COMPLETA)
    # -----------------------------------------------------------------
    print("[4] Generando panel visual de resultados...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Gráfico 1: Curvas ROC Comparativas (Árbol vs Bosque)
    fpr_tree, tpr_tree, _ = roc_curve(y_test, y_prob_tree)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

    axes[0].plot(fpr_tree, tpr_tree, color='#e67e22', linestyle='--', linewidth=2,
                 label=f'Árbol Único (AUC = {auc_tree:.3f})')
    axes[0].plot(fpr_rf, tpr_rf, color='#27ae60', linewidth=2.5,
                 label=f'Random Forest (AUC = {auc_rf:.3f})')
    axes[0].plot([0, 1], [0, 1], color='gray', linestyle=':', label='Clasificador Aleatorio (AUC = 0.500)')
    axes[0].set_title('Comparativa de Curvas ROC', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Tasa de Falsos Positivos (FPR - Falsas Alarmas)', fontsize=10)
    axes[0].set_ylabel('Tasa de Verdaderos Positivos (TPR - Recall)', fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(loc='lower right', frameon=True)

    # Gráfico 2: Importancia de Variables en el Bosque
    top_importancias = importancias.head(6).sort_values(ascending=True)
    axes[1].barh(top_importancias.index, top_importancias.values, color='#2980b9', edgecolor='black', alpha=0.85)
    axes[1].set_title('Top Importancia de Variables (Random Forest)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Importancia Relativa (Gini Impurity)', fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.6, axis='x')

    plt.tight_layout()
    plt.savefig('evaluacion_rf_vs_tree.png', dpi=300)
    print("[OK] Gráficas guardadas como 'evaluacion_rf_vs_tree.png'.")
    plt.show()


if __name__ == "__main__":
    main()
