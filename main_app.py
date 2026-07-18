"""
Dashboard interactivo - Pobreza en Medellín (datos sintéticos)
================================================================
Universidad EAFIT · Maestría/Pregrado en Ciencia de Datos · 2026

IMPORTANTE: Todos los datos que muestra este dashboard son 100% SINTÉTICOS,
generados aleatoriamente dentro de la propia aplicación con NumPy/Pandas.
NO corresponden a cifras oficiales del DANE, la Alcaldía de Medellín ni de
ninguna otra fuente real. Su único fin es académico/demostrativo.

El dashboard permite:
- Generar 500 registros sintéticos (configurable) con 10 columnas sobre
  condiciones socioeconómicas por comuna/corregimiento de Medellín.
- Analizar cuáles comunas resultan "más afectadas" según distintas variables.
- Ver al menos una serie de tiempo (evolución mensual 2019-2024, incluyendo
  un choque simulado tipo pandemia).
- Filtrar y graficar todo de forma interactiva con Plotly.
- Acceder mediante una clave de acceso (9876).

Ejecutar con:
    streamlit run main_app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pobreza en Medellín (Sintético) | EAFIT",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CLAVE_ACCESO = "9876"

# ---------------------------------------------------------------------------
# Catálogos de referencia
# ---------------------------------------------------------------------------
# Índice de vulnerabilidad BASE por comuna/corregimiento (0 = baja pobreza,
# 1 = alta pobreza). Es una aproximación SINTÉTICA e ilustrativa a partir de
# patrones socioeconómicos generalmente conocidos de Medellín, NO una cifra
# oficial. Se usa para simular correlaciones realistas entre variables.
COMUNAS_VULNERABILIDAD = {
    "Popular": 0.85,
    "Santa Cruz": 0.80,
    "Manrique": 0.75,
    "Aranjuez": 0.65,
    "Castilla": 0.55,
    "Doce de Octubre": 0.78,
    "Robledo": 0.50,
    "Villa Hermosa": 0.72,
    "Buenos Aires": 0.55,
    "La Candelaria": 0.35,
    "Laureles-Estadio": 0.15,
    "La América": 0.30,
    "San Javier": 0.70,
    "El Poblado": 0.05,
    "Guayabal": 0.40,
    "Belén": 0.25,
    "San Sebastián de Palmitas": 0.75,
    "San Cristóbal": 0.60,
    "Altavista": 0.68,
    "San Antonio de Prado": 0.55,
    "Santa Elena": 0.50,
}
COMUNAS = list(COMUNAS_VULNERABILIDAD.keys())

ESTRATOS = [f"Estrato {i}" for i in range(1, 7)]
ACCESO_OPCIONES = ["Completo", "Parcial", "Limitado"]
NIVEL_EDUCATIVO_OPCIONES = [
    "Sin escolaridad", "Primaria", "Secundaria", "Técnico o Tecnológico", "Universitario",
]

CATEGORICAS = ["comuna", "estrato", "acceso_servicios_publicos", "nivel_educativo_predominante"]
NUMERICAS = [
    "poblacion_muestra", "ingreso_promedio_hogar", "tasa_pobreza_monetaria",
    "tasa_pobreza_extrema", "nbi_porcentaje",
]

FECHA_INICIO_SERIE = pd.Timestamp("2019-01-01")
N_MESES_SERIE = 72  # enero 2019 - diciembre 2024
FECHAS_MENSUALES = pd.date_range(FECHA_INICIO_SERIE, periods=N_MESES_SERIE, freq="MS")


# ---------------------------------------------------------------------------
# Utilidad: muestreo categórico vectorizado con pesos distintos por fila
# ---------------------------------------------------------------------------
def _muestreo_categorico_vectorizado(pesos: np.ndarray, categorias: np.ndarray, rng) -> np.ndarray:
    pesos = pesos / pesos.sum(axis=1, keepdims=True)
    acumulado = np.cumsum(pesos, axis=1)
    r = rng.random(pesos.shape[0])
    idx = (r[:, None] < acumulado).argmax(axis=1)
    return categorias[idx]


# ---------------------------------------------------------------------------
# Generación de datos sintéticos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def generar_datos(n_registros: int, semilla: int, simular_imperfecciones: bool = True) -> pd.DataFrame:
    """Genera un dataset sintético de pobreza en Medellín con 10 columnas:

    1. fecha_reporte (datetime)                  - mes del registro (serie de tiempo)
    2. comuna (categórica)                        - comuna/corregimiento
    3. estrato (categórica ordinal)                - Estrato 1 a 6
    4. poblacion_muestra (numérica)                - personas representadas en el registro
    5. ingreso_promedio_hogar (numérica)            - COP mensuales
    6. tasa_pobreza_monetaria (numérica, %)         - variable objetivo principal
    7. tasa_pobreza_extrema (numérica, %)
    8. nbi_porcentaje (numérica, %)                 - necesidades básicas insatisfechas
    9. acceso_servicios_publicos (categórica)       - Completo / Parcial / Limitado
    10. nivel_educativo_predominante (categórica)

    Las variables están correlacionadas entre sí mediante un índice de
    vulnerabilidad por comuna y una tendencia temporal con un choque
    simulado (tipo pandemia) entre 2020 y 2021.
    """
    rng = np.random.default_rng(semilla)

    # --- Comuna y su índice de vulnerabilidad base -------------------------
    comuna = rng.choice(COMUNAS, size=n_registros)
    vulnerabilidad = np.array([COMUNAS_VULNERABILIDAD[c] for c in comuna])
    # pequeño ruido individual para que no todos los registros de una misma
    # comuna sean idénticos
    vulnerabilidad = np.clip(vulnerabilidad + rng.normal(0, 0.05, n_registros), 0, 1)

    # --- Fecha (mensual, 2019-2024) + tendencia temporal --------------------
    idx_mes = rng.integers(0, N_MESES_SERIE, size=n_registros)
    fecha_reporte = FECHAS_MENSUALES[idx_mes]

    tendencia = -0.35 * (idx_mes / 12.0)  # leve mejora estructural con el tiempo
    choque_pandemia = 9.0 * np.exp(-((idx_mes - 16) ** 2) / (2 * 6.0 ** 2))  # pico ~mayo 2020

    ruido = rng.normal(0, 3.0, n_registros)

    # --- Tasa de pobreza monetaria (variable objetivo) ----------------------
    tasa_pobreza_monetaria = 18 + vulnerabilidad * 55 + tendencia + choque_pandemia + ruido
    tasa_pobreza_monetaria = np.clip(tasa_pobreza_monetaria, 1, 98)

    # --- Pobreza extrema: subconjunto de la pobreza monetaria ---------------
    fraccion_extrema = np.clip(0.20 + vulnerabilidad * 0.25 + rng.normal(0, 0.03, n_registros), 0.05, 0.6)
    tasa_pobreza_extrema = np.clip(tasa_pobreza_monetaria * fraccion_extrema, 0.5, tasa_pobreza_monetaria - 0.5)

    # --- NBI (necesidades básicas insatisfechas) -----------------------------
    nbi_porcentaje = np.clip(4 + vulnerabilidad * 42 + rng.normal(0, 4, n_registros), 0, 100)

    # --- Ingreso promedio del hogar (COP) ------------------------------------
    ingreso_base = 5_400_000 * (1 - 0.78 * vulnerabilidad)
    ingreso_promedio_hogar = np.clip(
        ingreso_base + rng.normal(0, 250_000, n_registros), 350_000, 12_000_000
    )

    # --- Estrato socioeconómico (correlacionado con vulnerabilidad) --------
    estrato_num = np.clip(
        np.round(6 - vulnerabilidad * 5 + rng.normal(0, 0.6, n_registros)), 1, 6
    ).astype(int)
    estrato = np.array([f"Estrato {e}" for e in estrato_num])

    # --- Población muestreada (tamaño del registro/encuesta) ----------------
    poblacion_muestra = rng.integers(40, 650, size=n_registros)

    # --- Acceso a servicios públicos (categórica, dependiente) --------------
    pesos_acceso = np.column_stack([
        np.clip(0.92 - vulnerabilidad * 0.85, 0.03, None),   # Completo
        np.full(n_registros, 0.30),                            # Parcial
        vulnerabilidad * 0.75,                                  # Limitado
    ])
    acceso_servicios_publicos = _muestreo_categorico_vectorizado(
        pesos_acceso, np.array(ACCESO_OPCIONES), rng
    )

    # --- Nivel educativo predominante (categórica, dependiente) -------------
    pesos_educ = np.column_stack([
        0.01 + vulnerabilidad * 0.10,                            # Sin escolaridad
        0.08 + vulnerabilidad * 0.28,                            # Primaria
        np.full(n_registros, 0.35),                              # Secundaria
        np.clip(0.28 - vulnerabilidad * 0.18, 0.03, None),       # Técnico/Tecnológico
        np.clip(0.28 - vulnerabilidad * 0.24, 0.02, None),       # Universitario
    ])
    nivel_educativo_predominante = _muestreo_categorico_vectorizado(
        pesos_educ, np.array(NIVEL_EDUCATIVO_OPCIONES), rng
    )

    df = pd.DataFrame({
        "fecha_reporte": fecha_reporte,
        "comuna": comuna,
        "estrato": estrato,
        "poblacion_muestra": poblacion_muestra,
        "ingreso_promedio_hogar": ingreso_promedio_hogar.round(0),
        "tasa_pobreza_monetaria": tasa_pobreza_monetaria.round(2),
        "tasa_pobreza_extrema": tasa_pobreza_extrema.round(2),
        "nbi_porcentaje": nbi_porcentaje.round(2),
        "acceso_servicios_publicos": acceso_servicios_publicos,
        "nivel_educativo_predominante": nivel_educativo_predominante,
    })

    # --- Imperfecciones controladas (nulos y atípicos), como en datos reales
    if simular_imperfecciones:
        idx_null_ingreso = rng.choice(n_registros, size=max(1, int(n_registros * 0.02)), replace=False)
        df.loc[idx_null_ingreso, "ingreso_promedio_hogar"] = np.nan

        idx_null_acceso = rng.choice(n_registros, size=max(1, int(n_registros * 0.015)), replace=False)
        df.loc[idx_null_acceso, "acceso_servicios_publicos"] = None

        idx_outlier = rng.choice(n_registros, size=max(1, int(n_registros * 0.006)), replace=False)
        df.loc[idx_outlier, "tasa_pobreza_monetaria"] = rng.choice([0.5, 99.5, 100.0], size=len(idx_outlier))

    df = df.sort_values("fecha_reporte").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Branding institucional (EAFIT · Ciencia de Datos · 2026)
# ---------------------------------------------------------------------------
def mostrar_marca(container, key_suffix: str = ""):
    logo = container.file_uploader(
        "Logo institucional (opcional)", type=["png", "jpg", "jpeg", "svg"],
        key=f"logo_uploader_{key_suffix}",
        help="Sube el logo oficial de EAFIT en PNG/JPG/SVG para reemplazar el bloque de marca.",
    )
    if logo is not None:
        container.image(logo, use_container_width=True)
    else:
        container.markdown(
            """
            <div style="background:#00205B;padding:20px 12px;border-radius:12px;
                        text-align:center;margin-bottom:4px;">
                <div style="color:#FFFFFF;font-size:28px;font-weight:800;letter-spacing:3px;">
                    EAFIT
                </div>
                <div style="color:#F2C230;font-size:11px;font-weight:700;
                            letter-spacing:3px;margin-top:2px;">
                    UNIVERSIDAD
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    container.markdown(
        """
        <div style="text-align:center;margin-top:-2px;margin-bottom:16px;">
            <div style="font-size:15px;font-weight:700;color:#00205B;">Ciencia de Datos</div>
            <div style="font-size:13px;color:#555;">2026</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Control de acceso (clave: 9876)
# ---------------------------------------------------------------------------
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    mostrar_marca(st.sidebar, key_suffix="login")
    st.sidebar.info("Ingresa la clave de acceso para continuar.")

    col_izq, col_centro, col_der = st.columns([1, 1.3, 1])
    with col_centro:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<h2 style='text-align:center;'>🔒 Dashboard de Pobreza en Medellín</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;color:#666;'>Datos sintéticos · EAFIT · Ciencia de Datos</p>",
            unsafe_allow_html=True,
        )
        clave_ingresada = st.text_input("Clave de acceso", type="password", key="clave_input")
        if st.button("Ingresar", use_container_width=True):
            if clave_ingresada == CLAVE_ACCESO:
                st.session_state.autenticado = True
                st.rerun()
            else:
                st.error("Clave incorrecta. Intenta nuevamente.")
    st.stop()

# ---------------------------------------------------------------------------
# Barra lateral: marca, configuración de datos y filtros
# ---------------------------------------------------------------------------
mostrar_marca(st.sidebar, key_suffix="app")
if st.sidebar.button("🔓 Cerrar sesión"):
    st.session_state.autenticado = False
    st.rerun()
st.sidebar.divider()

st.sidebar.title("⚙️ Configuración")

st.sidebar.subheader("1. Generación de datos sintéticos")
n_registros = st.sidebar.slider(
    "Número de registros", min_value=100, max_value=3000, value=500, step=50
)
semilla = st.sidebar.number_input(
    "Semilla aleatoria (seed)", min_value=0, max_value=99999, value=42, step=1
)
simular_imperfecciones = st.sidebar.checkbox(
    "Simular imperfecciones reales (nulos y atípicos)", value=True,
)
if st.sidebar.button("🔄 Regenerar datos"):
    st.cache_data.clear()

df = generar_datos(n_registros, semilla, simular_imperfecciones)

st.sidebar.subheader("2. Filtros")
rango_fechas = st.sidebar.date_input(
    "Rango de fechas",
    value=(df["fecha_reporte"].min().date(), df["fecha_reporte"].max().date()),
    min_value=df["fecha_reporte"].min().date(),
    max_value=df["fecha_reporte"].max().date(),
)
comunas_sel = st.sidebar.multiselect("Comuna / corregimiento", COMUNAS, default=COMUNAS)
estratos_sel = st.sidebar.multiselect("Estrato", ESTRATOS, default=ESTRATOS)
acceso_sel = st.sidebar.multiselect("Acceso a servicios públicos", ACCESO_OPCIONES, default=ACCESO_OPCIONES)
educ_sel = st.sidebar.multiselect("Nivel educativo predominante", NIVEL_EDUCATIVO_OPCIONES, default=NIVEL_EDUCATIVO_OPCIONES)
rango_pobreza = st.sidebar.slider(
    "Rango de tasa de pobreza monetaria (%)", 0.0, 100.0, (0.0, 100.0)
)
incluir_nulos = st.sidebar.checkbox(
    "Incluir registros con ingreso/acceso nulos en los filtros", value=True
)

# Aplicar filtros
if len(rango_fechas) == 2:
    f_ini, f_fin = pd.Timestamp(rango_fechas[0]), pd.Timestamp(rango_fechas[1])
else:
    f_ini, f_fin = df["fecha_reporte"].min(), df["fecha_reporte"].max()

filtro_acceso = df["acceso_servicios_publicos"].isin(acceso_sel) | (incluir_nulos & df["acceso_servicios_publicos"].isna())

df_f = df[
    (df["fecha_reporte"] >= f_ini)
    & (df["fecha_reporte"] <= f_fin)
    & (df["comuna"].isin(comunas_sel))
    & (df["estrato"].isin(estratos_sel))
    & filtro_acceso
    & (df["nivel_educativo_predominante"].isin(educ_sel))
    & (df["tasa_pobreza_monetaria"].between(rango_pobreza[0], rango_pobreza[1]))
].copy()

# ---------------------------------------------------------------------------
# Encabezado y KPIs
# ---------------------------------------------------------------------------
st.title("🏙️ Dashboard de Pobreza en Medellín (Datos Sintéticos)")
st.caption(
    "⚠️ Datos 100% simulados dentro de la aplicación con fines académicos (EAFIT · Ciencia de Datos). "
    "No representan cifras oficiales del DANE ni de la Alcaldía de Medellín."
)

if df_f.empty:
    st.warning("No hay registros que coincidan con los filtros seleccionados.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros filtrados", f"{len(df_f):,}")
k2.metric("Pobreza monetaria (prom.)", f"{df_f['tasa_pobreza_monetaria'].mean():.1f}%")
k3.metric("Pobreza extrema (prom.)", f"{df_f['tasa_pobreza_extrema'].mean():.1f}%")
k4.metric("Ingreso hogar (prom.)", f"${df_f['ingreso_promedio_hogar'].mean():,.0f} COP")
comuna_mas_afectada = (
    df_f.groupby("comuna")["tasa_pobreza_monetaria"].mean().idxmax()
    if not df_f["comuna"].empty else "N/A"
)
k5.metric("Comuna más afectada", comuna_mas_afectada)

st.divider()

# ---------------------------------------------------------------------------
# Tabs principales
# ---------------------------------------------------------------------------
tab_datos, tab_ranking, tab_cuant, tab_cual, tab_series, tab_graf = st.tabs(
    [
        "📄 Vista general", "🏘️ Ranking por comuna", "🔢 Estadística cuantitativa",
        "🔤 Estadística cualitativa", "⏳ Serie de tiempo", "📊 Gráficos dinámicos",
    ]
)

# --- Tab 1: Vista general ---------------------------------------------------
with tab_datos:
    st.subheader("Vista general de los datos")
    st.dataframe(df_f, use_container_width=True, height=350)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Tipos de datos por columna**")
        tipos = pd.DataFrame({
            "columna": df_f.dtypes.index,
            "tipo_dato": df_f.dtypes.astype(str).values,
        })
        st.dataframe(tipos, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("**Valores nulos por columna**")
        nulos = df_f.isna().sum().rename("nulos").to_frame()
        st.dataframe(nulos, use_container_width=True)

    csv = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar datos filtrados (CSV)", data=csv,
        file_name="pobreza_medellin_sintetico.csv", mime="text/csv"
    )

# --- Tab 2: Ranking por comuna (lugares más afectados) ----------------------
with tab_ranking:
    st.subheader("¿Cuáles comunas resultan más afectadas?")

    col_cfg, col_top = st.columns([2, 1])
    with col_cfg:
        variable_ranking = st.selectbox(
            "Variable para el ranking",
            NUMERICAS,
            index=NUMERICAS.index("tasa_pobreza_monetaria"),
        )
    with col_top:
        top_n = st.slider("Top N comunas", 3, len(COMUNAS), 10)

    ascendente = variable_ranking == "ingreso_promedio_hogar"  # menor ingreso = más afectado
    resumen_comuna = (
        df_f.groupby("comuna")
        .agg(
            tasa_pobreza_monetaria=("tasa_pobreza_monetaria", "mean"),
            tasa_pobreza_extrema=("tasa_pobreza_extrema", "mean"),
            nbi_porcentaje=("nbi_porcentaje", "mean"),
            ingreso_promedio_hogar=("ingreso_promedio_hogar", "mean"),
            poblacion_muestra=("poblacion_muestra", "sum"),
            registros=("comuna", "count"),
        )
        .reset_index()
    )
    resumen_ordenado = resumen_comuna.sort_values(
        variable_ranking, ascending=ascendente
    ).head(top_n)

    fig_rank = px.bar(
        resumen_ordenado.sort_values(variable_ranking, ascending=not ascendente),
        x=variable_ranking, y="comuna", orientation="h", color=variable_ranking,
        color_continuous_scale="Reds" if not ascendente else "Blues_r",
        title=f"Top {top_n} comunas según {variable_ranking}",
    )
    fig_rank.update_layout(height=500)
    st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown("#### Mapa de proporciones (treemap): tamaño = población, color = tasa de pobreza")
    fig_tree = px.treemap(
        resumen_comuna, path=["comuna"], values="poblacion_muestra",
        color="tasa_pobreza_monetaria", color_continuous_scale="Reds",
        title="Comunas por población muestreada y pobreza monetaria promedio",
    )
    fig_tree.update_layout(height=480)
    st.plotly_chart(fig_tree, use_container_width=True)

    with st.expander("Ver tabla resumen por comuna"):
        st.dataframe(
            resumen_comuna.sort_values(variable_ranking, ascending=ascendente),
            use_container_width=True, hide_index=True,
        )

# --- Tab 3: Estadística cuantitativa -----------------------------------------
with tab_cuant:
    st.subheader("Estadística descriptiva - Variables cuantitativas")
    var_num = st.selectbox("Selecciona una variable numérica", NUMERICAS, key="var_num")

    serie = df_f[var_num].dropna()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Media", f"{serie.mean():,.2f}")
    c2.metric("Mediana", f"{serie.median():,.2f}")
    c3.metric("Desv. estándar", f"{serie.std():,.2f}")
    c4.metric("Mínimo", f"{serie.min():,.2f}")
    c5.metric("Máximo", f"{serie.max():,.2f}")
    c6.metric("Varianza", f"{serie.var():,.2f}")

    with st.expander("Ver resumen estadístico completo (describe)"):
        st.dataframe(serie.describe().to_frame().T, use_container_width=True)

    st.markdown("#### Histograma con umbral personalizado")
    col_h1, col_h2 = st.columns([3, 1])
    with col_h2:
        n_bins = st.slider("Número de bins", 5, 100, 30)
        umbral_num = st.slider(
            "Umbral (línea de referencia)",
            float(serie.min()), float(serie.max()), float(serie.mean()),
        )
        color_hist = st.color_picker("Color del histograma", "#C0392B")
    with col_h1:
        fig_hist = px.histogram(
            df_f, x=var_num, nbins=n_bins,
            title=f"Distribución de {var_num}",
            color_discrete_sequence=[color_hist],
        )
        fig_hist.add_vline(
            x=umbral_num, line_dash="dash", line_color="black",
            annotation_text=f"Umbral = {umbral_num:.1f}", annotation_position="top",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("#### Boxplot por categoría")
    cat_boxplot = st.selectbox(
        "Agrupar boxplot por variable categórica", CATEGORICAS, key="cat_box"
    )
    fig_box = px.box(
        df_f, x=cat_boxplot, y=var_num, color=cat_boxplot,
        title=f"{var_num} agrupado por {cat_boxplot}",
    )
    fig_box.add_hline(
        y=umbral_num, line_dash="dash", line_color="black",
        annotation_text=f"Umbral = {umbral_num:.1f}",
    )
    st.plotly_chart(fig_box, use_container_width=True)

# --- Tab 4: Estadística cualitativa -------------------------------------------
with tab_cual:
    st.subheader("Estadística descriptiva - Variables cualitativas")
    var_cat = st.selectbox("Selecciona una variable categórica", CATEGORICAS, key="var_cat")

    conteo = df_f[var_cat].value_counts(dropna=True).rename("frecuencia").to_frame()
    conteo["porcentaje (%)"] = (conteo["frecuencia"] / conteo["frecuencia"].sum() * 100).round(2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Categorías únicas", df_f[var_cat].nunique())
    c2.metric("Moda", df_f[var_cat].mode().iloc[0] if not df_f[var_cat].mode().empty else "N/A")
    c3.metric("Registros totales", len(df_f))

    col_t, col_g = st.columns([1, 2])
    with col_t:
        st.markdown("**Tabla de frecuencias**")
        st.dataframe(conteo, use_container_width=True)
    with col_g:
        tipo_graf_cat = st.radio(
            "Tipo de gráfico", ["Barras", "Pastel (pie)"], horizontal=True
        )
        if tipo_graf_cat == "Barras":
            fig_cat = px.bar(
                conteo, x=conteo.index, y="frecuencia", color=conteo.index,
                title=f"Frecuencia de {var_cat}",
                labels={"x": var_cat},
            )
        else:
            fig_cat = px.pie(
                conteo, names=conteo.index, values="frecuencia",
                title=f"Proporción de {var_cat}",
            )
        st.plotly_chart(fig_cat, use_container_width=True)

# --- Tab 5: Serie de tiempo -----------------------------------------------------
with tab_series:
    st.subheader("Evolución mensual (serie de tiempo, 2019 - 2024)")
    st.caption(
        "Incluye un choque simulado (tipo pandemia) alrededor de 2020-2021 "
        "para observar cómo se recupera la tendencia con el tiempo."
    )

    col_cfg2, col_plot2 = st.columns([1, 3])
    with col_cfg2:
        var_serie = st.selectbox("Variable a graficar", NUMERICAS, key="var_serie", index=NUMERICAS.index("tasa_pobreza_monetaria"))
        agregacion = st.selectbox("Función de agregación", ["Promedio", "Mediana", "Suma"])
        comparar_comunas = st.checkbox("Comparar por comuna", value=False)
        comunas_comparar = None
        if comparar_comunas:
            comunas_comparar = st.multiselect(
                "Comunas a comparar", sorted(df_f["comuna"].unique()),
                default=sorted(df_f["comuna"].unique())[:3],
            )
        mostrar_umbral_serie = st.checkbox("Mostrar línea de umbral", value=False)
        umbral_serie = None
        if mostrar_umbral_serie:
            umbral_serie = st.slider(
                "Valor del umbral", float(df_f[var_serie].min()), float(df_f[var_serie].max()),
                float(df_f[var_serie].mean()), key="umbral_serie",
            )
        resaltar_choque = st.checkbox("Resaltar periodo de choque simulado (2020-2021)", value=True)

    func_agg = {"Promedio": "mean", "Mediana": "median", "Suma": "sum"}[agregacion]

    with col_plot2:
        if comparar_comunas and comunas_comparar:
            df_serie = (
                df_f[df_f["comuna"].isin(comunas_comparar)]
                .groupby([pd.Grouper(key="fecha_reporte", freq="MS"), "comuna"])[var_serie]
                .agg(func_agg)
                .reset_index()
            )
            fig_serie = px.line(
                df_serie, x="fecha_reporte", y=var_serie, color="comuna",
                markers=True, title=f"{agregacion} mensual de {var_serie} por comuna",
            )
        else:
            df_serie = (
                df_f.groupby(pd.Grouper(key="fecha_reporte", freq="MS"))[var_serie]
                .agg(func_agg)
                .reset_index()
            )
            fig_serie = px.line(
                df_serie, x="fecha_reporte", y=var_serie, markers=True,
                title=f"{agregacion} mensual de {var_serie} (general)",
            )

        if resaltar_choque:
            fig_serie.add_vrect(
                x0="2020-03-01", x1="2021-12-31", fillcolor="orange", opacity=0.15,
                line_width=0, annotation_text="Choque simulado", annotation_position="top left",
            )
        if mostrar_umbral_serie and umbral_serie is not None:
            fig_serie.add_hline(
                y=umbral_serie, line_dash="dash", line_color="crimson",
                annotation_text=f"Umbral = {umbral_serie:.1f}",
            )
        fig_serie.update_layout(height=520)
        st.plotly_chart(fig_serie, use_container_width=True)

# --- Tab 6: Gráficos dinámicos -----------------------------------------------
with tab_graf:
    st.subheader("Gráficos dinámicos personalizables")
    st.caption("Elige el tipo de gráfico, las variables y personaliza los detalles visuales.")

    col_cfg, col_plot = st.columns([1, 3])

    with col_cfg:
        tipo_grafico = st.selectbox(
            "Tipo de gráfico",
            ["Dispersión (scatter)", "Línea temporal", "Barras", "Histograma", "Boxplot", "Violín", "Pastel"],
        )

        todas_cols = NUMERICAS + CATEGORICAS + ["fecha_reporte"]

        eje_x = st.selectbox("Variable eje X", todas_cols, index=0)
        necesita_y = tipo_grafico in [
            "Dispersión (scatter)", "Línea temporal", "Barras", "Boxplot", "Violín"
        ]
        eje_y = None
        if necesita_y:
            eje_y = st.selectbox("Variable eje Y", NUMERICAS, index=2)

        color_por = st.selectbox("Colorear por (opcional)", ["Ninguno"] + CATEGORICAS)
        color_por = None if color_por == "Ninguno" else color_por

        plantilla = st.selectbox(
            "Estilo / plantilla",
            ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"],
        )
        titulo_custom = st.text_input("Título del gráfico", value=f"{tipo_grafico}: {eje_x}" + (f" vs {eje_y}" if eje_y else ""))

        mostrar_umbral = st.checkbox("Mostrar línea de umbral")
        valor_umbral = None
        orientacion_umbral = "horizontal"
        if mostrar_umbral:
            var_umbral_ref = eje_y if eje_y else "tasa_pobreza_monetaria"
            serie_ref = df_f[var_umbral_ref] if var_umbral_ref in NUMERICAS else df_f["tasa_pobreza_monetaria"]
            valor_umbral = st.slider(
                "Valor del umbral", float(serie_ref.min()), float(serie_ref.max()), float(serie_ref.mean())
            )
            orientacion_umbral = st.radio("Orientación del umbral", ["horizontal", "vertical"], horizontal=True)

    with col_plot:
        try:
            if tipo_grafico == "Dispersión (scatter)":
                fig = px.scatter(
                    df_f, x=eje_x, y=eje_y, color=color_por, template=plantilla,
                    title=titulo_custom, hover_data=CATEGORICAS,
                )
            elif tipo_grafico == "Línea temporal":
                df_agg = df_f.groupby(pd.Grouper(key="fecha_reporte", freq="MS"))[eje_y].mean().reset_index()
                fig = px.line(
                    df_agg, x="fecha_reporte", y=eje_y, template=plantilla,
                    title=titulo_custom, markers=True,
                )
            elif tipo_grafico == "Barras":
                df_agg = df_f.groupby(eje_x, as_index=False)[eje_y].mean()
                fig = px.bar(
                    df_agg, x=eje_x, y=eje_y, color=color_por if color_por in df_agg.columns else None,
                    template=plantilla, title=titulo_custom,
                )
            elif tipo_grafico == "Histograma":
                fig = px.histogram(
                    df_f, x=eje_x, color=color_por, template=plantilla, title=titulo_custom
                )
            elif tipo_grafico == "Boxplot":
                fig = px.box(
                    df_f, x=eje_x, y=eje_y, color=color_por, template=plantilla, title=titulo_custom
                )
            elif tipo_grafico == "Violín":
                fig = px.violin(
                    df_f, x=eje_x, y=eje_y, color=color_por, template=plantilla,
                    title=titulo_custom, box=True,
                )
            elif tipo_grafico == "Pastel":
                conteo_pie = df_f[eje_x].value_counts(dropna=True).reset_index()
                conteo_pie.columns = [eje_x, "frecuencia"]
                fig = px.pie(
                    conteo_pie, names=eje_x, values="frecuencia",
                    template=plantilla, title=titulo_custom,
                )

            if mostrar_umbral and valor_umbral is not None and tipo_grafico != "Pastel":
                if orientacion_umbral == "horizontal":
                    fig.add_hline(
                        y=valor_umbral, line_dash="dash", line_color="crimson",
                        annotation_text=f"Umbral = {valor_umbral:.1f}",
                    )
                else:
                    fig.add_vline(
                        x=valor_umbral, line_dash="dash", line_color="crimson",
                        annotation_text=f"Umbral = {valor_umbral:.1f}",
                    )

            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"No fue posible generar el gráfico con esta combinación de variables: {e}")

st.divider()
st.caption(
    "Dashboard construido con Streamlit + Plotly · Universidad EAFIT · Ciencia de Datos · 2026 · "
    "Datos 100% sintéticos generados en tiempo de ejecución."
)
