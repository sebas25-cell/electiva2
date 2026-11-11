import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Módulos de Scikit-Learn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)

# Conector personalizado a MySQL (corregido: connection, no connection2)
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

    print("[1] Extrayendo datos desde MySQL...")
    df = db.extract_data(query)

    if df.empty:
        print("[ERROR] No se obtuvieron datos.")
        return

    # Limpieza e imputación por mediana
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

    # Construcción de variables (X, y)
    df_dummies = pd.get_dummies(df[['posicion']], drop_first=True, dtype=int)
    X = pd.concat([df[['edad', 'estatura_cm']], df_dummies], axis=1)
    y = df['es_elite']

    # Depuración Anti-NaN
    filas_con_nan = X.isna().any(axis=1)
    if filas_con_nan.sum() > 0:
        X = X[~filas_con_nan]
        y = y[~filas_con_nan]

    # División 80/20 estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[OK] Preprocesamiento completado: {X.shape[0]} registros totales.")
    print(f"     Distribución del Target -> Estándar (0): {np.sum(y==0)} | Élite (1): {np.sum(y==1)}\n")

    # -----------------------------------------------------------------
    # FASE 2: DEMOSTRACIÓN DE LA PARADOJA DE LA EXACTITUD
    # -----------------------------------------------------------------
    print("[2] Evaluando Modelo Base Sin Tratamiento de Desbalance...")

    tree_base = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree_base.fit(X_train, y_train)
    y_pred_base = tree_base.predict(X_test)

    acc_base = accuracy_score(y_test, y_pred_base)
    f1_base = f1_score(y_test, y_pred_base, pos_label=1)

    print("\n--- MODELO BASE (SIN TRATAMIENTO) ---")
    print(f"  • Accuracy Global: {acc_base:.4f}")
    print(f"  • F1-Score (Clase Élite): {f1_base:.4f}")
    print("\nReporte Detallado:")
    print(classification_report(y_test, y_pred_base, target_names=['Estándar (0)', 'Élite (1)']))

    # -----------------------------------------------------------------
    # FASE 3: REBALANCEO ALGORÍTMICO (class_weight='balanced')
    # -----------------------------------------------------------------
    print("\n[3] Aplicando penalización de costo (class_weight='balanced')...")

    tree_bal = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    tree_bal.fit(X_train, y_train)
    y_pred_bal = tree_bal.predict(X_test)

    acc_bal = accuracy_score(y_test, y_pred_bal)
    f1_bal = f1_score(y_test, y_pred_bal, pos_label=1)

    print("\n--- MODELO CON PESOS BALANCEADOS ---")
    print(f"  • Accuracy Global: {acc_bal:.4f}")
    print(f"  • F1-Score (Clase Élite): {f1_bal:.4f}")
    print("\nReporte Detallado:")
    print(classification_report(y_test, y_pred_bal, target_names=['Estándar (0)', 'Élite (1)']))

    # -----------------------------------------------------------------
    # FASE 4: OPTIMIZACIÓN SISTEMÁTICA CON GRIDSEARCHCV
    # -----------------------------------------------------------------
    print("\n[4] Ejecutando GridSearchCV (Búsqueda combinatoria en Malla + CV)...")

    param_grid = {
        'max_depth': [2, 3, 4, 5, 6],
        'min_samples_leaf': [3, 5, 10, 15],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None]
    }

    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"\nMejores Hiperparámetros Encontrados: {grid_search.best_params_}")
    print(f"Mejor F1-Score Promedio en CV: {grid_search.best_score_:.4f}")

    # Evaluación en Test Set
    modelo_optimo = grid_search.best_estimator_
    y_pred_opt = modelo_optimo.predict(X_test)

    acc_opt = accuracy_score(y_test, y_pred_opt)
    f1_opt = f1_score(y_test, y_pred_opt, pos_label=1)

    print("\n=================================================")
    print("  EVALUACIÓN FINAL DEL MODELO ÓPTIMO (TEST SET)")
    print("=================================================")
    print(classification_report(y_test, y_pred_opt, target_names=['Estándar (0)', 'Élite (1)']))
    print("Matriz de Confusión:")
    print(confusion_matrix(y_test, y_pred_opt))
    print("=================================================\n")

    # -----------------------------------------------------------------
    # FASE 5: GENERACIÓN DE GRÁFICAS EXPLICATIVAS DEL PROCESO
    # -----------------------------------------------------------------
    print("[5] Generando panel visual del laboratorio...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Gráfico: Distribución de Clases (Desbalance)
    conteo_clases = y.value_counts()
    axes[0, 0].bar(['Estándar (0)', 'Élite (1)'], conteo_clases.values,
                   color=['#2980b9', '#e74c3c'], edgecolor='black')
    axes[0, 0].set_title('1. Distribución del Target (Desbalance de Clases)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Cantidad de Jugadores', fontsize=10)
    for i, v in enumerate(conteo_clases.values):
        axes[0, 0].text(i, v + 2, str(v), ha='center', fontweight='bold')

    # 2. Gráfico: Evolución de Métricas por Fase
    modelos = ['Base', 'Pesos Bal.', 'GridSearch Opt.']
    acc_scores = [acc_base, acc_bal, acc_opt]
    f1_scores = [f1_base, f1_bal, f1_opt]

    x = np.arange(len(modelos))
    width = 0.35

    axes[0, 1].bar(x - width/2, acc_scores, width, label='Accuracy', color='#3498db', edgecolor='black')
    axes[0, 1].bar(x + width/2, f1_scores, width, label='F1-Score (Élite)', color='#2ecc71', edgecolor='black')
    axes[0, 1].set_title('2. Comparativa de Métricas por Etapa', fontsize=11, fontweight='bold')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(modelos)
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].legend(loc='upper right')
    axes[0, 1].grid(True, linestyle='--', alpha=0.5, axis='y')

    # 3. Gráfico: Matriz de Confusión Modelo Base
    cm_base = confusion_matrix(y_test, y_pred_base)
    disp_base = ConfusionMatrixDisplay(confusion_matrix=cm_base, display_labels=['Estándar', 'Élite'])
    disp_base.plot(ax=axes[1, 0], cmap='Blues', colorbar=False)
    axes[1, 0].set_title('3. Matriz Confusión: Modelo Base', fontsize=11, fontweight='bold')

    # 4. Gráfico: Matriz de Confusión Modelo Optimizado
    cm_opt = confusion_matrix(y_test, y_pred_opt)
    disp_opt = ConfusionMatrixDisplay(confusion_matrix=cm_opt, display_labels=['Estándar', 'Élite'])
    disp_opt.plot(ax=axes[1, 1], cmap='Greens', colorbar=False)
    axes[1, 1].set_title('4. Matriz Confusión: Modelo Optimizado', fontsize=11, fontweight='bold')

    plt.tight_layout()

    # Exportación de la gráfica para la clase
    os.makedirs("data/processed", exist_ok=True)
    ruta_grafico = "data/processed/panel_desbalance_gridsearch.png"
    plt.savefig(ruta_grafico, dpi=300)
    print(f"[OK] Gráfica guardada exitosamente en: {ruta_grafico}")
    plt.show()


if __name__ == "__main__":
    main()
