import pandas as pd
import numpy as np

# Módulos de Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Conector personalizado a la base de datos (corregido: connection, no connection2)
from src.database.connection import MySQLConnector


def main():
    print("=================================================================")
    print("  MINERÍA DE DATOS - LAB 5: CLASES DESBALANCEADAS Y GRIDSEARCHCV")
    print("=================================================================\n")

    # -----------------------------------------------------------------
    # FASE 1: EXTRAER Y PREPARAR EL DATASET DESBALANCEADO
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

    print("[1] Cargando datos desde MySQL...")
    df = db.extract_data(query)

    # Limpieza e imputación
    df['estatura_cm'] = pd.to_numeric(df['estatura_cm'], errors='coerce')
    df['edad'] = pd.to_numeric(df['edad'], errors='coerce')
    df['valor_mercado'] = pd.to_numeric(df['valor_mercado'], errors='coerce')
    df['posicion'] = df['posicion'].fillna('Sin Posicion')

    df['estatura_cm'] = df.groupby('posicion')['estatura_cm'].transform(lambda x: x.fillna(x.median()))
    df['estatura_cm'] = df['estatura_cm'].fillna(df['estatura_cm'].median())
    df['edad'] = df['edad'].fillna(df['edad'].median())

    # Target Desbalanceado: Élite (Top 25% = 1, Resto 75% = 0)
    umbral_elite = df['valor_mercado'].quantile(0.75)
    df['es_elite'] = (df['valor_mercado'] >= umbral_elite).astype(int)

    # Variables predictoras (X) y Target (y)
    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)
    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']

    # Depuración Anti-NaN
    filas_con_nan = X.isna().any(axis=1)
    if filas_con_nan.sum() > 0:
        X = X[~filas_con_nan]
        y = y[~filas_con_nan]

    # División 80/20 manteniendo la proporción de clases (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[OK] Dataset listo: {X.shape[0]} registros.")
    print(f"     Proporción de Target (y): {np.bincount(y)} (0: Estándar, 1: Élite)\n")

    # -----------------------------------------------------------------
    # FASE 2: DEMOSTRACIÓN DE LA PARADOJA DE LA EXACTITUD
    # -----------------------------------------------------------------
    print("[2] Entrenando Árbol Estándar (Sin tratamiento de desbalance)...")

    tree_base = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_base.fit(X_train, y_train)
    y_pred_base = tree_base.predict(X_test)

    print("\n--- MODELO BASE (DESBALANCEADO) ---")
    print(f"  • Accuracy Global: {accuracy_score(y_test, y_pred_base):.4f}")
    print(f"  • F1-Score (Clase Élite): {f1_score(y_test, y_pred_base, pos_label=1):.4f}")
    print("\nReporte Detallado:")
    print(classification_report(y_test, y_pred_base, target_names=['Estándar (0)', 'Élite (1)']))

    # -----------------------------------------------------------------
    # FASE 3: REBALANCEO ALGORÍTMICO (class_weight='balanced')
    # -----------------------------------------------------------------
    print("\n[3] Aplicando penalización de costo (class_weight='balanced')...")

    tree_bal = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    tree_bal.fit(X_train, y_train)
    y_pred_bal = tree_bal.predict(X_test)

    print("\n--- MODELO CON PESOS BALANCEADOS ---")
    print(f"  • Accuracy Global: {accuracy_score(y_test, y_pred_bal):.4f}")
    print(f"  • F1-Score (Clase Élite): {f1_score(y_test, y_pred_bal, pos_label=1):.4f}")
    print("\nReporte Detallado:")
    print(classification_report(y_test, y_pred_bal, target_names=['Estándar (0)', 'Élite (1)']))

    # -----------------------------------------------------------------
    # FASE 4: OPTIMIZACIÓN AUTOMÁTICA CON GRIDSEARCHCV
    # -----------------------------------------------------------------
    print("\n[4] Ejecutando GridSearchCV (Búsqueda sistemática + K-Fold CV)...")

    # Definir la Malla (Grid) de combinaciones
    param_grid = {
        'max_depth': [2, 3, 4, 5, 6],
        'min_samples_leaf': [3, 5, 10, 15],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None]
    }

    # Configurar el Robot de Búsqueda
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='f1',  # Optimizar buscando el mejor F1-Score
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nMejores Hiperparámetros Encontrados: {grid_search.best_params_}")
    print(f"Mejor F1-Score Promedio en CV: {grid_search.best_score_:.4f}")

    # Evaluación en datos de prueba no vistos (Test Set)
    modelo_optimo = grid_search.best_estimator_
    y_pred_opt = modelo_optimo.predict(X_test)

    print("\n=================================================")
    print("  EVALUACIÓN FINAL DEL MODELO ÓPTIMO (TEST SET)")
    print("=================================================")
    print(classification_report(y_test, y_pred_opt, target_names=['Estándar (0)', 'Élite (1)']))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred_opt))
    print("=================================================\n")


if __name__ == "__main__":
    main()
