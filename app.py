# ==========================================================
# app.py
# ==========================================================
# Ejecutar:
#
# streamlit run app.py
#
# Requiere:
#
# pip install streamlit pandas matplotlib scikit-learn
# ==========================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="Predicción de Ventas",
    layout="wide"
)

st.title("Predicción de Ventas")
st.divider()


# ==========================================================
# CARGA DE DATOS
# ==========================================================

@st.cache_data
def cargar_datos():

    df = pd.read_csv("train.csv")

    df = df.drop_duplicates()

    df["Sales"] = pd.to_numeric(
        df["Sales"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["Sales"]
    )

    return df


df = cargar_datos()


# ==========================================================
# ENTRENAR REGRESIÓN LINEAL
# ==========================================================

@st.cache_resource
def entrenar_modelo_lineal(df):

    features = [
        "Category",
        "Sub-Category",
        "Segment",
        "Region",
        "Ship Mode"
    ]

    X = df[features]
    y = df["Sales"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorias",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                features
            )
        ]
    )

    modelo = make_pipeline(
        preprocessor,
        LinearRegression()
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=5
    )

    modelo.fit(
        X_train,
        y_train
    )

    y_pred = modelo.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    return (
        modelo,
        X_test,
        y_test,
        y_pred,
        r2
    )


(
    modelo_lineal,
    X_test_lineal,
    y_test_lineal,
    y_pred_lineal,
    r2_lineal
) = entrenar_modelo_lineal(df)


# ==========================================================
# ENTRENAR RED NEURONAL
# ==========================================================

@st.cache_resource
def entrenar_red_neuronal(df):

    features = [
        "Category",
        "Sub-Category",
        "Segment",
        "Region",
        "Ship Mode"
    ]

    X = df[features]
    y = df["Sales"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorias",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                features
            )
        ]
    )

    modelo = TransformedTargetRegressor(
        regressor=make_pipeline(
            preprocessor,
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                alpha=0.001,
                max_iter=500,
                early_stopping=True,
                random_state=5
            )
        ),
        transformer=StandardScaler()
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=5
    )

    modelo.fit(
        X_train,
        y_train
    )

    y_pred = modelo.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        y_pred
    )
    mae = mean_absolute_error(
        y_test,
        y_pred
    )
    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    return (
        modelo,
        X_test,
        y_test,
        y_pred,
        r2,
        mae,
        rmse
    )


# ==========================================================
# ENTRENAR RED NEURONAL DE CLASIFICACIÓN
# ==========================================================

@st.cache_resource
def entrenar_red_clasificacion(df):

    features = [
        "Category",
        "Sub-Category",
        "Segment",
        "Region",
        "Ship Mode"
    ]
    clases = [
        "Ventas bajas",
        "Ventas medias",
        "Ventas altas"
    ]

    X = df[features]
    y = df["Sales"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        random_state=5
    )

    limite_bajo = y_train.quantile(1 / 3)
    limite_alto = y_train.quantile(2 / 3)
    limites = [
        -float("inf"),
        limite_bajo,
        limite_alto,
        float("inf")
    ]

    y_train_clases = pd.cut(
        y_train,
        bins=limites,
        labels=False
    ).astype("int64")
    y_test_clases = pd.cut(
        y_test,
        bins=limites,
        labels=False
    ).astype("int64")

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorias",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first"
                ),
                features
            )
        ]
    )

    modelo = make_pipeline(
        preprocessor,
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.001,
            max_iter=500,
            early_stopping=True,
            random_state=5
        )
    )

    modelo.fit(
        X_train,
        y_train_clases
    )

    y_pred = modelo.predict(
        X_test
    )
    accuracy = accuracy_score(
        y_test_clases,
        y_pred
    )
    matriz_confusion = confusion_matrix(
        y_test_clases,
        y_pred,
        labels=range(len(clases))
    )

    return (
        modelo,
        y_test_clases,
        y_pred,
        accuracy,
        matriz_confusion,
        limite_bajo,
        limite_alto,
        clases
    )


# ==========================================================
# LAYOUT PRINCIPAL
# ==========================================================

col_filtros, col_resultado = st.columns(
    [1, 3],
    gap="large"
)

# ==========================================================
# PANEL IZQUIERDO
# ==========================================================

with col_filtros:

    st.subheader("Filtros")

    modelo_seleccionado = st.selectbox(
        "Modelo",
        [
            "Regresión Lineal",
            "Redes Neuronales",
            "Redes Neuronales de Clasificación",
            "Árbol de Decisión"
        ]
    )

    categorias = sorted(
        df["Category"]
        .dropna()
        .unique()
    )

    categoria = st.selectbox(
        "Categoría",
        categorias
    )

    df_sub = df[
        df["Category"] == categoria
    ]

    subcategorias = sorted(
        df_sub["Sub-Category"]
        .dropna()
        .unique()
    )

    subcategoria = st.selectbox(
        "Sub-Categoría",
        subcategorias
    )

    df_segment = df[
        (df["Category"] == categoria)
        &
        (df["Sub-Category"] == subcategoria)
    ]

    segmentos = sorted(
        df_segment["Segment"]
        .dropna()
        .unique()
    )

    segmento = st.selectbox(
        "Segmento",
        segmentos
    )

    df_region = df[
        (df["Category"] == categoria)
        &
        (df["Sub-Category"] == subcategoria)
        &
        (df["Segment"] == segmento)
    ]

    regiones = sorted(
        df_region["Region"]
        .dropna()
        .unique()
    )

    region = st.selectbox(
        "Región",
        regiones
    )

    df_ship = df[
        (df["Category"] == categoria)
        &
        (df["Sub-Category"] == subcategoria)
        &
        (df["Segment"] == segmento)
        &
        (df["Region"] == region)
    ]

    ship_modes = sorted(
        df_ship["Ship Mode"]
        .dropna()
        .unique()
    )

    ship_mode = st.selectbox(
        "Modo de Envío",
        ship_modes
    )

# ==========================================================
# PEDIDO
# ==========================================================

nuevo_pedido = pd.DataFrame([{
    "Category": categoria,
    "Sub-Category": subcategoria,
    "Segment": segmento,
    "Region": region,
    "Ship Mode": ship_mode
}])

# ==========================================================
# PANEL DERECHO
# ==========================================================

with col_resultado:

    if modelo_seleccionado == "Regresión Lineal":

        prediccion = modelo_lineal.predict(
            nuevo_pedido
        )[0]

        st.success(
            f"""
            **Resultado de la predicción**

            Para la Categoría **{categoria}**,
            Sub-Categoría **{subcategoria}**,
            Segmento **{segmento}**,
            Región **{region}**
            y modo de envío **{ship_mode}**,
            el modelo de Regresión Lineal predice
            una venta de **{prediccion:.2f} USD**
            con un nivel de explicación (R²)
            de **{r2_lineal * 100:.2f}%**.
            """
        )

        fig, ax = plt.subplots()

        fig.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.95,
            bottom=0.08
        )

        ax.scatter(
            y_test_lineal,
            y_pred_lineal,
            alpha=0.4,
            color="blue",
            s=15
        )

        limite_min = min(
            y_test_lineal.min(),
            y_pred_lineal.min()
        )

        limite_max = max(
            y_test_lineal.max(),
            y_pred_lineal.max()
        )

        ax.plot(
            [limite_min, limite_max],
            [limite_min, limite_max],
            color="red",
            linewidth=2,
            label="Predicción ideal"
        )

        # Punto resaltado de la predicción
        ax.scatter(
            prediccion,
            prediccion,
            color="green",
            marker="*",
            s=300,
            label="Predicción seleccionada"
        )

        ax.annotate(
            f"${prediccion:.2f}",
            (prediccion, prediccion),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8
        )

        ax.set_xlabel(
            "Ventas reales",
            fontsize=10
        )

        ax.set_ylabel(
            "Ventas predichas",
            fontsize=10
        )

        ax.set_title(
            "Regresión Lineal Múltiple",
            fontsize=12,
            fontweight="bold"
        )

        ax.tick_params(
            labelsize=8
        )

        ax.legend(
            fontsize=8
        )

        ax.grid(
            alpha=0.3
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

    elif modelo_seleccionado == "Redes Neuronales":

        (
            modelo_red_neuronal,
            X_test_red,
            y_test_red,
            y_pred_red,
            r2_red,
            mae_red,
            rmse_red
        ) = entrenar_red_neuronal(df)

        prediccion_red = modelo_red_neuronal.predict(
            nuevo_pedido
        )[0]

        st.success(
            f"""
            **Resultado de la predicción**

            Para la Categoría **{categoria}**,
            Sub-Categoría **{subcategoria}**,
            Segmento **{segmento}**,
            Región **{region}**
            y modo de envío **{ship_mode}**,
            la Red Neuronal estima ventas de
            **{prediccion_red:.2f} USD**.
            """
        )

        st.subheader("Evaluación en datos de prueba")
        col_r2, col_mae, col_rmse = st.columns(3)

        col_r2.metric(
            "R²",
            f"{r2_red:.3f}"
        )
        col_mae.metric(
            "MAE (USD)",
            f"{mae_red:.2f}"
        )
        col_rmse.metric(
            "RMSE (USD)",
            f"{rmse_red:.2f}"
        )

        fig, ax = plt.subplots()

        ax.scatter(
            y_test_red,
            y_pred_red,
            alpha=0.4,
            color="blue",
            s=15,
            label="Predicciones de prueba"
        )

        limite_min = min(
            y_test_red.min(),
            y_pred_red.min()
        )
        limite_max = max(
            y_test_red.max(),
            y_pred_red.max()
        )

        ax.plot(
            [limite_min, limite_max],
            [limite_min, limite_max],
            color="red",
            linewidth=2,
            label="Predicción ideal"
        )

        ax.set_xlabel(
            "Ventas reales (USD)",
            fontsize=10
        )
        ax.set_ylabel(
            "Ventas predichas (USD)",
            fontsize=10
        )
        ax.set_title(
            "Red Neuronal: valores reales vs. predichos",
            fontsize=12,
            fontweight="bold"
        )
        ax.tick_params(
            labelsize=8
        )
        ax.legend(
            fontsize=8
        )
        ax.grid(
            alpha=0.3
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

    elif modelo_seleccionado == "Redes Neuronales de Clasificación":

        (
            modelo_clasificacion,
            y_test_clases,
            y_pred_clases,
            accuracy_clasificacion,
            matriz_confusion,
            limite_bajo,
            limite_alto,
            clases
        ) = entrenar_red_clasificacion(df)

        prediccion_clase = modelo_clasificacion.predict(
            nuevo_pedido
        )[0]
        nombre_clase = clases[int(prediccion_clase)]
        probabilidades = modelo_clasificacion.predict_proba(
            nuevo_pedido
        )[0]
        confianza_clase = probabilidades[
            list(modelo_clasificacion.classes_).index(prediccion_clase)
        ]

        st.success(
            f"""
            **Clasificación de ventas**

            Para la Categoría **{categoria}**,
            Sub-Categoría **{subcategoria}**,
            Segmento **{segmento}**,
            Región **{region}**
            y modo de envío **{ship_mode}**,
            el modelo clasifica las ventas como
            **{nombre_clase}**.

            Probabilidad estimada para esta clase:
            **{confianza_clase * 100:.2f}%**.
            """
        )

        st.caption(
            f"Rangos definidos con terciles del conjunto de entrenamiento: "
            f"bajas ≤ ${limite_bajo:.2f}; "
            f"medias > ${limite_bajo:.2f} y ≤ ${limite_alto:.2f}; "
            f"altas > ${limite_alto:.2f}."
        )
        st.metric(
            "Exactitud en datos de prueba",
            f"{accuracy_clasificacion * 100:.2f}%"
        )

        fig, ax = plt.subplots()
        imagen = ax.imshow(
            matriz_confusion,
            cmap="Blues"
        )
        fig.colorbar(
            imagen,
            ax=ax,
            label="Cantidad de pedidos"
        )

        posiciones = range(len(clases))
        ax.set_xticks(posiciones)
        ax.set_xticklabels(
            clases,
            rotation=20,
            ha="right"
        )
        ax.set_yticks(posiciones)
        ax.set_yticklabels(clases)
        ax.set_xlabel("Clase predicha")
        ax.set_ylabel("Clase real")
        ax.set_title(
            "Red Neuronal de Clasificación: matriz de confusión",
            fontsize=12,
            fontweight="bold"
        )

        for fila in range(len(clases)):
            for columna in range(len(clases)):
                ax.text(
                    columna,
                    fila,
                    matriz_confusion[fila, columna],
                    ha="center",
                    va="center"
                )

        fig.tight_layout()
        st.pyplot(
            fig,
            use_container_width=True
        )

    else:

        df_arbol = df[
            df["Category"] == categoria
        ].copy()

        features_arbol = [
            "Sub-Category",
            "Region"
        ]

        X_arbol = df_arbol[
            features_arbol
        ]

        y_arbol = df_arbol[
            "Sales"
        ]

        preprocessor_arbol = ColumnTransformer(
            transformers=[
                (
                    "categorias",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                    features_arbol
                )
            ]
        )

        modelo_arbol = make_pipeline(
            preprocessor_arbol,
            DecisionTreeRegressor(
                max_depth=3,
                random_state=5
            )
        )

        modelo_arbol.fit(
            X_arbol,
            y_arbol
        )

        prediccion_arbol = modelo_arbol.predict(
            pd.DataFrame([{
                "Sub-Category": subcategoria,
                "Region": region
            }])
        )[0]

        st.success(
            f"""
            **Resultado de la predicción**

            Para la Categoría **{categoria}**,
            Sub-Categoría **{subcategoria}**
            y Región **{region}**,
            el Árbol de Decisión estima una
            venta de **{prediccion_arbol:.2f} USD**.
            """
        )

        encoder = (
            modelo_arbol
            .named_steps["columntransformer"]
            .named_transformers_["categorias"]
        )

        feature_names = encoder.get_feature_names_out(
            features_arbol
        )

        feature_names_limpios = []

        for nombre in feature_names:

            nombre = nombre.replace(
                "Sub-Category_",
                "Sub-Category = "
            )

            nombre = nombre.replace(
                "Region_",
                "Region = "
            )

            feature_names_limpios.append(
                nombre
            )

        arbol = modelo_arbol.named_steps[
            "decisiontreeregressor"
        ]

        fig, ax = plt.subplots()

        fig.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.95,
            bottom=0.05
        )

        plot_tree(
            arbol,
            feature_names=feature_names_limpios,
            filled=True,
            rounded=True,
            fontsize=8,
            precision=2,
            ax=ax
        )

        ax.set_title(
            f"Árbol de Decisión - {categoria}",
            fontsize=12,
            fontweight="bold"
        )

        st.pyplot(
            fig,
            use_container_width=True
        )
