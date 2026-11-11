import os
import pandas as pd
import numpy as np

# Módulos de Scikit-Learn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Conector a la base de datos MySQL del proyecto (corregido: connection, no connection2)
from src.database.connection import MySQLConnector


def main():
    print("=================================================================")
    print("  MINERÍA DE DATOS - TALLER PRACTICO: OPTIMIZACIÓN Y DIAGNÓSTICO")
    print("=================================================================\n")

    # -----------------------------------------------------------------
    # FASE 1: INGESTA Y PREPARACIÓN DE DATOS
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

    if df.empty:
        print("[ERROR] No se obtuvieron datos.")
        return

    # Limpieza de nulos e imputación por mediana
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')

    df['posicion'] = df['posicion'].fillna('Sin Posicion')

    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(lambda x: x.fillna(x.median()))
    df['estatura_cm'] = df['estatura_cm'].fillna(df['estatura_cm'].median())
    df['edad'] = df['edad'].fillna(df['edad'].median())

    # Definición de la Clase Target (Élite = Top 25% de valor comercial)
    umbral_elite = df['valor_mercado'].quantile(0.75)
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    # Construcción de la matriz X y vector y
    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)
    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']

    # Filtro anti-NaN
    filas_con_nan = X.isna().any(axis=1)
    if filas_con_nan.sum() > 0:
        X = X[~filas_con_nan]
        y = y[~filas_con_nan]

    # División 80% Entrenamiento / 20% Prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[OK] Preprocesamiento completado. Datos cargados: {X.shape[0]} registros.\n")

    # -----------------------------------------------------------------
    # FASE 2: DETECCIÓN DE OVERFITTING (ÁRBOL SIN CONTROL)
    # -----------------------------------------------------------------
    print("[2] Evaluando Árbol de Decisión Sin Límites (max_depth=None)...")

    tree_libre = DecisionTreeClassifier(max_depth=None, random_state=42)
    tree_libre.fit(X_train, y_train)

    acc_train_libre = accuracy_score(y_train, tree_libre.predict(X_train))
    acc_test_libre = accuracy_score(y_test, tree_libre.predict(X_test))
    brecha_libre = acc_train_libre - acc_test_libre

    print(f"  • Accuracy en Train: {acc_train_libre:.4f}")
    print(f"  • Accuracy en Test:  {acc_test_libre:.4f}")
    print(f"  • Brecha (Train - Test): {brecha_libre:.4f} ({brecha_libre*100:.2f}%)\n")

    # -----------------------------------------------------------------
    # FASE 3: APLICACIÓN DE PODA Y REGULARIZACIÓN
    # -----------------------------------------------------------------
    print("[3] Aplicando técnicas de optimización...")

    # A. Árbol Podado (Pre-pruning)
    tree_opt = DecisionTreeClassifier(max_depth=7, min_samples_leaf=10, random_state=42)
    tree_opt.fit(X_train, y_train)

    acc_train_podado = accuracy_score(y_train, tree_opt.predict(X_train))
    acc_test_podado = accuracy_score(y_test, tree_opt.predict(X_test))
    brecha_podado = acc_train_podado - acc_test_podado

    print("\n--- A. Árbol Podado (max_depth=3, min_samples_leaf=10) ---")
    print(f"  • Accuracy en Train: {acc_train_podado:.4f}")
    print(f"  • Accuracy en Test:  {acc_test_podado:.4f}")
    print(f"  • Nueva Brecha:      {brecha_podado:.4f} ({brecha_podado*100:.2f}%)")

    # B. Regresión Logística Regularizada
    log_freno = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    log_freno.fit(X_train, y_train)

    acc_train_log = accuracy_score(y_train, log_freno.predict(X_train))
    acc_test_log = accuracy_score(y_test, log_freno.predict(X_test))

    print("\n--- B. Regresión Logística Regularizada (C=0.1) ---")
    print(f"  • Accuracy en Train: {acc_train_log:.4f}")
    print(f"  • Accuracy en Test:  {acc_test_log:.4f}\n")

    # -----------------------------------------------------------------
    # FASE 4: VALIDACIÓN CRUZADA (K-FOLD CV, K=5)
    # -----------------------------------------------------------------
    print("[4] Evaluando estabilidad del Árbol Podado mediante K-Fold CV (K=5)...")

    scores_f1 = cross_val_score(tree_opt, X, y, cv=5, scoring='f1')
    scores_acc = cross_val_score(tree_opt, X, y, cv=5, scoring='accuracy')

    print("\n=================================================")
    print("  RESULTADOS DE VALIDACIÓN CRUZADA (K=5)")
    print("=================================================")
    print(f"  • F1-Score por Pliegue:    {np.round(scores_f1, 4)}")
    print(f"  • Promedio F1-Score:       {scores_f1.mean():.4f}")
    print(f"  • Desviación Estándar F1:  {scores_f1.std():.4f}")
    print(f"  • Accuracy Promedio:       {scores_acc.mean():.4f}")
    print("=================================================\n")


if __name__ == "__main__":
    main()
