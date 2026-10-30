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
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score


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