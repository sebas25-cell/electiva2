# ================================================================
# EVALUACIÓN INTEGRAL DE MINERÍA DE DATOS
# MÓDULOS 1 AL 6
# ================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# MÓDULOS DE SCIKIT-LEARN
# ================================================================

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# ================================================================
# CONEXIÓN A MYSQL
# ================================================================

from src.database.connection import MySQLConnector


# ================================================================
# FUNCIÓN PRINCIPAL
# ================================================================

def main():

    print("=================================================================")
    print("       EVALUACIÓN INTEGRAL DE MINERÍA DE DATOS")
    print("                    MÓDULOS 1 AL 6")
    print("=================================================================\n")

    # ============================================================
    # MÓDULO 1: INGESTA Y PREPROCESAMIENTO
    # ============================================================

    print("[FASE 1] Extrayendo datos desde la base MySQL...")

    # Crear conexión
    db = MySQLConnector()

    # ------------------------------------------------------------
    # CONSULTA SQL
    # ------------------------------------------------------------

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

    # Extraer datos
    df = db.extract_data(query)

    # Verificar si llegaron datos
    if df is None or df.empty:

        print("[ERROR] No se obtuvieron datos de la base de datos.")

        return

    print(f"[OK] Datos extraídos: {len(df)} registros.\n")

    # ============================================================
    # LIMPIEZA DE DATOS
    # ============================================================

    print("[FASE 1.1] Realizando limpieza y preprocesamiento...")

    # Convertir variables numéricas
    df["estatura_cm"] = pd.to_numeric(
        df["estatura_cm"],
        errors="coerce"
    )

    df["edad"] = pd.to_numeric(
        df["edad"],
        errors="coerce"
    )

    df["valor_mercado"] = pd.to_numeric(
        df["valor_mercado"],
        errors="coerce"
    )

    # Posiciones nulas
    df["posicion"] = df["posicion"].fillna(
        "Sin Posicion"
    )

    # ------------------------------------------------------------
    # IMPUTACIÓN POR MEDIANA AGRUPADA POR POSICIÓN
    # ------------------------------------------------------------

    df["estatura_cm"] = (
        df.groupby("posicion")["estatura_cm"]
        .transform(
            lambda x: x.fillna(x.median())
        )
    )

    # Si todavía existen valores nulos,
    # utilizar la mediana general
    df["estatura_cm"] = df["estatura_cm"].fillna(
        df["estatura_cm"].median()
    )

    # Imputación de edad
    df["edad"] = df["edad"].fillna(
        df["edad"].median()
    )

    # Eliminar registros donde todavía falte
    # el valor de mercado
    df = df.dropna(
        subset=["valor_mercado"]
    )

    # ============================================================
    # TARGET: ES_ELITE
    # ============================================================

    # Umbral correspondiente al percentil 75
    umbral_elite = df["valor_mercado"].quantile(
        0.75
    )

    # Jugadores cuyo valor está en el 25% superior
    # serán considerados Élite
    df["es_elite"] = (
        df["valor_mercado"] >= umbral_elite
    ).astype(int)

    print(
        f"[OK] Umbral para considerar un jugador Élite: "
        f"{umbral_elite:,.2f}"
    )

    # ============================================================
    # ONE-HOT ENCODING
    # ============================================================

    df_dummies = pd.get_dummies(
        df[["posicion"]],
        drop_first=True,
        dtype=int
    )

    # Variables predictoras
    X = pd.concat(
        [
            df[["edad", "estatura_cm"]],
            df_dummies
        ],
        axis=1
    )

    # Variable objetivo
    y = df["es_elite"]

    # ============================================================
    # DEPURACIÓN FINAL DE NaN
    # ============================================================

    filas_con_nan = X.isna().any(axis=1)

    if filas_con_nan.sum() > 0:

        print(
            f"[INFO] Eliminando "
            f"{filas_con_nan.sum()} filas con NaN."
        )

        X = X[~filas_con_nan]
        y = y[~filas_con_nan]

    print(
        f"[OK] Preprocesamiento completado."
    )

    print(
        f"[OK] Registros procesados: {X.shape[0]}"
    )

    print(
        f"[OK] Variables predictoras: {X.shape[1]}\n"
    )

    # ============================================================
    # DIVISIÓN TRAIN / TEST
    # 80% ENTRENAMIENTO
    # 20% PRUEBA
    # ESTRATIFICADA
    # ============================================================

    print(
        "[FASE 1.2] Dividiendo dataset "
        "en Train y Test..."
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"[OK] Train: {X_train.shape[0]} registros"
    )

    print(
        f"[OK] Test : {X_test.shape[0]} registros\n"
    )

    # ============================================================
    # MÓDULO 2:
    # DIAGNÓSTICO DE OVERFITTING
    # ÁRBOL SIN CONTROL
    # ============================================================

    print("=================================================================")
    print("[FASE 2] DIAGNÓSTICO DE OVERFITTING")
    print("=================================================================")

    # Árbol sin límite de profundidad
    tree_libre = DecisionTreeClassifier(
        max_depth=None,
        random_state=42
    )

    # Entrenamiento
    tree_libre.fit(
        X_train,
        y_train
    )

    # Predicciones
    y_train_pred_libre = tree_libre.predict(
        X_train
    )

    y_test_pred_libre = tree_libre.predict(
        X_test
    )

    # Accuracy Train
    acc_train_libre = accuracy_score(
        y_train,
        y_train_pred_libre
    )

    # Accuracy Test
    acc_test_libre = accuracy_score(
        y_test,
        y_test_pred_libre
    )

    # Brecha
    brecha_libre = (
        acc_train_libre -
        acc_test_libre
    )

    print(
        f" • Accuracy Train: "
        f"{acc_train_libre:.4f}"
    )

    print(
        f" • Accuracy Test:  "
        f"{acc_test_libre:.4f}"
    )

    print(
        f" • Brecha Train - Test: "
        f"{brecha_libre:.4f}"
    )

    print(
        f" • Brecha porcentual: "
        f"{brecha_libre * 100:.2f}%\n"
    )

    # ============================================================
    # MÓDULO 3:
    # REGULARIZACIÓN Y PODA
    # ============================================================

    print("=================================================================")
    print("[FASE 3] REGULARIZACIÓN Y PODA DEL ÁRBOL")
    print("=================================================================")

    # Árbol podado
    tree_podado = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=10,
        random_state=42
    )

    # Entrenamiento
    tree_podado.fit(
        X_train,
        y_train
    )

    # Predicciones Train
    y_train_pred_podado = tree_podado.predict(
        X_train
    )

    # Predicciones Test
    y_test_pred_podado = tree_podado.predict(
        X_test
    )

    # Accuracy Train
    acc_train_podado = accuracy_score(
        y_train,
        y_train_pred_podado
    )

    # Accuracy Test
    acc_test_podado = accuracy_score(
        y_test,
        y_test_pred_podado
    )

    # F1 clase Élite
    f1_podado = f1_score(
        y_test,
        y_test_pred_podado,
        pos_label=1,
        zero_division=0
    )

    # Brecha
    brecha_podado = (
        acc_train_podado -
        acc_test_podado
    )

    print(
        f" • Accuracy Train: "
        f"{acc_train_podado:.4f}"
    )

    print(
        f" • Accuracy Test:  "
        f"{acc_test_podado:.4f}"
    )

    print(
        f" • Brecha Train - Test: "
        f"{brecha_podado:.4f}"
    )

    print(
        f" • F1-Score Clase Élite: "
        f"{f1_podado:.4f}\n"
    )

    # Reducción de la brecha
    reduccion_brecha = (
        brecha_libre -
        brecha_podado
    )

    print(
        f" • Reducción de la brecha: "
        f"{reduccion_brecha:.4f}"
    )

    print(
        f" • Reducción porcentual: "
        f"{reduccion_brecha * 100:.2f}%\n"
    )

    # ============================================================
    # MÓDULO 4:
    # DESBALANCE DE CLASES
    # ============================================================

    print("=================================================================")
    print("[FASE 4] ANÁLISIS DE DESBALANCE DE CLASES")
    print("=================================================================")

    # ------------------------------------------------------------
    # DISTRIBUCIÓN DE CLASES
    # ------------------------------------------------------------

    conteo_clases = y.value_counts().sort_index()

    print("\nDistribución de clases:")

    for clase, cantidad in conteo_clases.items():

        nombre = (
            "Estándar"
            if clase == 0
            else "Élite"
        )

        porcentaje = (
            cantidad / len(y)
        ) * 100

        print(
            f" • {nombre} ({clase}): "
            f"{cantidad} "
            f"({porcentaje:.2f}%)"
        )

    # ------------------------------------------------------------
    # MODELO SIN TRATAMIENTO DEL DESBALANCE
    # ------------------------------------------------------------

    modelo_sin_balance = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=10,
        random_state=42
    )

    modelo_sin_balance.fit(
        X_train,
        y_train
    )

    y_pred_sin_balance = (
        modelo_sin_balance.predict(X_test)
    )

    f1_sin_balance = f1_score(
        y_test,
        y_pred_sin_balance,
        pos_label=1,
        zero_division=0
    )

    precision_sin_balance = precision_score(
        y_test,
        y_pred_sin_balance,
        pos_label=1,
        zero_division=0
    )

    recall_sin_balance = recall_score(
        y_test,
        y_pred_sin_balance,
        pos_label=1,
        zero_division=0
    )

    acc_sin_balance = accuracy_score(
        y_test,
        y_pred_sin_balance
    )

    print("\nModelo SIN balanceo:")

    print(
        f" • Accuracy: {acc_sin_balance:.4f}"
    )

    print(
        f" • Precision Élite: "
        f"{precision_sin_balance:.4f}"
    )

    print(
        f" • Recall Élite: "
        f"{recall_sin_balance:.4f}"
    )

    print(
        f" • F1 Élite: "
        f"{f1_sin_balance:.4f}"
    )

    # ------------------------------------------------------------
    # MODELO CON BALANCEO
    # ------------------------------------------------------------

    modelo_balanceado = DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    )

    modelo_balanceado.fit(
        X_train,
        y_train
    )

    y_pred_balanceado = (
        modelo_balanceado.predict(X_test)
    )

    f1_balanceado = f1_score(
        y_test,
        y_pred_balanceado,
        pos_label=1,
        zero_division=0
    )

    precision_balanceado = precision_score(
        y_test,
        y_pred_balanceado,
        pos_label=1,
        zero_division=0
    )

    recall_balanceado = recall_score(
        y_test,
        y_pred_balanceado,
        pos_label=1,
        zero_division=0
    )

    acc_balanceado = accuracy_score(
        y_test,
        y_pred_balanceado
    )

    print("\nModelo CON balanceo:")

    print(
        f" • Accuracy: {acc_balanceado:.4f}"
    )

    print(
        f" • Precision Élite: "
        f"{precision_balanceado:.4f}"
    )

    print(
        f" • Recall Élite: "
        f"{recall_balanceado:.4f}"
    )

    print(
        f" • F1 Élite: "
        f"{f1_balanceado:.4f}\n"
    )

    # ============================================================
    # MÓDULO 5:
    # GRIDSEARCHCV
    # ============================================================

    print("=================================================================")
    print("[FASE 5] OPTIMIZACIÓN CON GRIDSEARCHCV")
    print("=================================================================")

    # Malla de hiperparámetros
    param_grid = {

        "max_depth": [
            2,
            3,
            4,
            5,
            6
        ],

        "min_samples_leaf": [
            3,
            5,
            10,
            15
        ],

        "criterion": [
            "gini",
            "entropy"
        ],

        "class_weight": [
            "balanced",
            None
        ]
    }

    # Modelo base
    modelo_base = DecisionTreeClassifier(
        random_state=42
    )

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
        verbose=1
    )

    # Entrenamiento
    print(
        "\nEjecutando búsqueda de hiperparámetros..."
    )

    grid_search.fit(
        X_train,
        y_train
    )

    # ============================================================
    # MEJOR MODELO
    # ============================================================

    modelo_optimo = (
        grid_search.best_estimator_
    )

    print(
        "\n[OK] Búsqueda completada."
    )

    print(
        "\nMejores hiperparámetros:"
    )

    print(
        grid_search.best_params_
    )

    print(
        f"\nMejor F1 obtenido durante Cross Validation: "
        f"{grid_search.best_score_:.4f}"
    )

    # ============================================================
    # EVALUACIÓN SOBRE TEST
    # ============================================================

    y_pred_opt = (
        modelo_optimo.predict(X_test)
    )

    acc_opt = accuracy_score(
        y_test,
        y_pred_opt
    )

    f1_opt = f1_score(
        y_test,
        y_pred_opt,
        pos_label=1,
        zero_division=0
    )

    precision_opt = precision_score(
        y_test,
        y_pred_opt,
        pos_label=1,
        zero_division=0
    )

    recall_opt = recall_score(
        y_test,
        y_pred_opt,
        pos_label=1,
        zero_division=0
    )

    print("\nResultados del modelo optimizado:")

    print(
        f" • Accuracy Test: "
        f"{acc_opt:.4f}"
    )

    print(
        f" • Precision Élite: "
        f"{precision_opt:.4f}"
    )

    print(
        f" • Recall Élite: "
        f"{recall_opt:.4f}"
    )

    print(
        f" • F1-Score Élite: "
        f"{f1_opt:.4f}\n"
    )

    # ============================================================
    # MÓDULO 6:
    # MATRIZ DE CONFUSIÓN
    # ============================================================

    print("=================================================================")
    print("[FASE 6] MATRIZ DE CONFUSIÓN Y ANÁLISIS")
    print("=================================================================")

    cm_libre = confusion_matrix(
        y_test,
        y_test_pred_libre
    )

    cm_opt = confusion_matrix(
        y_test,
        y_pred_opt
    )

    print("\nMatriz de Confusión - Árbol Sin Control:")

    print(cm_libre)

    print("\nMatriz de Confusión - Modelo Optimizado:")

    print(cm_opt)

    # ------------------------------------------------------------
    # EXTRAER COMPONENTES
    # ------------------------------------------------------------

    tn_libre, fp_libre, fn_libre, tp_libre = (
        cm_libre.ravel()
    )

    tn_opt, fp_opt, fn_opt, tp_opt = (
        cm_opt.ravel()
    )

    print("\nÁrbol Sin Control:")

    print(
        f" • Verdaderos Negativos: {tn_libre}"
    )

    print(
        f" • Falsos Positivos: {fp_libre}"
    )

    print(
        f" • Falsos Negativos: {fn_libre}"
    )

    print(
        f" • Verdaderos Positivos: {tp_libre}"
    )

    print("\nModelo Optimizado:")

    print(
        f" • Verdaderos Negativos: {tn_opt}"
    )

    print(
        f" • Falsos Positivos: {fp_opt}"
    )

    print(
        f" • Falsos Negativos: {fn_opt}"
    )

    print(
        f" • Verdaderos Positivos: {tp_opt}"
    )

    # ============================================================
    # REPORTE DE CLASIFICACIÓN
    # ============================================================

    print("\n=================================================================")
    print("REPORTE DE CLASIFICACIÓN - MODELO FINAL")
    print("=================================================================")

    print(
        classification_report(
            y_test,
            y_pred_opt,
            target_names=[
                "Estándar",
                "Élite"
            ],
            zero_division=0
        )
    )

    # ============================================================
    # PANEL VISUAL 2x2
    # ============================================================

    print(
        "[FASE 7] Generando panel de gráficas..."
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 10)
    )

    # ============================================================
    # GRÁFICO 1:
    # OVERFITTING
    # ============================================================

    etapas_over = [
        "Árbol Sin Límite",
        "Árbol Podado"
    ]

    train_accs = [
        acc_train_libre,
        acc_train_podado
    ]

    test_accs = [
        acc_test_libre,
        acc_test_podado
    ]

    x = np.arange(
        len(etapas_over)
    )

    width = 0.35

    axes[0, 0].bar(
        x - width / 2,
        train_accs,
        width,
        label="Train Accuracy",
        edgecolor="black"
    )

    axes[0, 0].bar(
        x + width / 2,
        test_accs,
        width,
        label="Test Accuracy",
        edgecolor="black"
    )

    axes[0, 0].set_title(
        "1. Diagnóstico de Overfitting y Efecto de Poda",
        fontsize=11,
        fontweight="bold"
    )

    axes[0, 0].set_xticks(x)

    axes[0, 0].set_xticklabels(
        etapas_over
    )

    axes[0, 0].set_ylim(
        0,
        1.1
    )

    axes[0, 0].set_ylabel(
        "Accuracy"
    )

    axes[0, 0].legend(
        loc="lower right"
    )

    axes[0, 0].grid(
        True,
        linestyle="--",
        alpha=0.5,
        axis="y"
    )

    # ============================================================
    # GRÁFICO 2:
    # DESBALANCE
    # ============================================================

    clases_nombres = [
        "Estándar (0)",
        "Élite (1)"
    ]

    valores_clases = [
        conteo_clases.get(0, 0),
        conteo_clases.get(1, 0)
    ]

    axes[0, 1].bar(
        clases_nombres,
        valores_clases,
        edgecolor="black"
    )

    axes[0, 1].set_title(
        "2. Inspección del Desbalance de Clases",
        fontsize=11,
        fontweight="bold"
    )

    axes[0, 1].set_ylabel(
        "Cantidad de Jugadores"
    )

    for i, valor in enumerate(
        valores_clases
    ):

        axes[0, 1].text(
            i,
            valor + max(valores_clases) * 0.01,
            str(valor),
            ha="center",
            fontweight="bold"
        )

    # ============================================================
    # GRÁFICO 3:
    # MATRIZ DE CONFUSIÓN SIN CONTROL
    # ============================================================

    disp_libre = ConfusionMatrixDisplay(
        confusion_matrix=cm_libre,
        display_labels=[
            "Estándar",
            "Élite"
        ]
    )

    disp_libre.plot(
        ax=axes[1, 0],
        cmap="Reds",
        colorbar=False
    )

    axes[1, 0].set_title(
        "3. Matriz Confusión: Árbol Sin Control",
        fontsize=11,
        fontweight="bold"
    )

    # ============================================================
    # GRÁFICO 4:
    # MATRIZ DE CONFUSIÓN MODELO FINAL
    # ============================================================

    disp_opt = ConfusionMatrixDisplay(
        confusion_matrix=cm_opt,
        display_labels=[
            "Estándar",
            "Élite"
        ]
    )

    disp_opt.plot(
        ax=axes[1, 1],
        cmap="Greens",
        colorbar=False
    )

    axes[1, 1].set_title(
        "4. Matriz Confusión: Modelo Final Optimizado",
        fontsize=11,
        fontweight="bold"
    )

    # ============================================================
    # AJUSTAR Y GUARDAR
    # ============================================================

    plt.tight_layout()

    nombre_grafica = (
        "resultado_evaluacion_integral.png"
    )

    plt.savefig(
        nombre_grafica,
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"\n[OK] Panel gráfico guardado como:"
    )

    print(
        f"     {nombre_grafica}"
    )

    # Mostrar gráfica
    plt.show()

    # ============================================================
    # RESUMEN FINAL
    # ============================================================

    print("\n=================================================================")
    print("                     RESUMEN FINAL")
    print("=================================================================")

    print(
        f"Registros analizados: {len(X)}"
    )

    print(
        f"Umbral Élite: {umbral_elite:,.2f}"
    )

    print(
        f"Accuracy árbol libre: "
        f"{acc_test_libre:.4f}"
    )

    print(
        f"Brecha árbol libre: "
        f"{brecha_libre:.4f}"
    )

    print(
        f"Accuracy árbol podado: "
        f"{acc_test_podado:.4f}"
    )

    print(
        f"Brecha árbol podado: "
        f"{brecha_podado:.4f}"
    )

    print(
        f"Reducción de brecha: "
        f"{reduccion_brecha:.4f}"
    )

    print(
        f"F1 árbol podado: "
        f"{f1_podado:.4f}"
    )

    print(
        f"F1 modelo optimizado: "
        f"{f1_opt:.4f}"
    )

    print(
        f"Accuracy modelo optimizado: "
        f"{acc_opt:.4f}"
    )

    print(
        f"Precision Élite optimizado: "
        f"{precision_opt:.4f}"
    )

    print(
        f"Recall Élite optimizado: "
        f"{recall_opt:.4f}"
    )

    print(
        f"Verdaderos Positivos finales: "
        f"{tp_opt}"
    )

    print(
        f"Falsos Positivos finales: "
        f"{fp_opt}"
    )

    print(
        f"Falsos Negativos finales: "
        f"{fn_opt}"
    )

    print(
        "\n================================================================="
    )

    print(
        "EVALUACIÓN FINALIZADA CORRECTAMENTE"
    )

    print(
        "=================================================================")


# ================================================================
# EJECUCIÓN DEL PROGRAMA
# ================================================================

if __name__ == "__main__":
    main()