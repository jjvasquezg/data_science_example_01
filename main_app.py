"""
Dashboard interactivo COVID-19 (datos sintéticos)
==================================================
Este dashboard genera datos sintéticos de COVID-19 dentro de la propia
aplicación (no requiere archivos externos), con 9 columnas y correlaciones
realistas entre variables (edad, vacunación, síntomas y mortalidad),
estacionalidad y olas pandémicas en los casos nuevos, y permite al usuario:

- Regenerar los datos con distintos tamaños de muestra y semillas.
- Activar/desactivar la simulación de nulos y valores atípicos.
- Filtrar los datos interactivamente (fecha, país, región, género, estado,
  vacunación, síntomas y edad).
- Ver estadística cuantitativa (numérica) y cualitativa (categórica).
- Explorar gráficos dinámicos con Plotly, eligiendo variables, tipo de
  gráfico, umbrales (líneas de referencia) y personalización visual.

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
    page_title="Dashboard COVID-19 (Datos Sintéticos)",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAISES = [
    "Colombia", "México", "Argentina", "Brasil", "Chile",
    "Perú", "España", "Estados Unidos", "Ecuador", "Bolivia",
]
# Pesos relativos (aprox. proporcionales a población) para que el muestreo
# por país sea más realista que uno uniforme.
PAIS_PESOS = {
    "Estados Unidos": 14, "Brasil": 13, "México": 12, "Colombia": 8,
    "Argentina": 7, "Perú": 7, "España": 6, "Chile": 5, "Ecuador": 5, "Bolivia": 4,
}
_PESOS_ARR = np.array([PAIS_PESOS[p] for p in PAISES], dtype=float)
PAIS_PROBS = _PESOS_ARR / _PESOS_ARR.sum()

REGIONES = {
    "Colombia": ["Bogotá", "Antioquia", "Valle del Cauca", "Atlántico", "Santander"],
    "México": ["CDMX", "Jalisco", "Nuevo León", "Puebla", "Veracruz"],
    "Argentina": ["Buenos Aires", "Córdoba", "Santa Fe", "Mendoza", "Tucumán"],
    "Brasil": ["São Paulo", "Río de Janeiro", "Minas Gerais", "Bahía", "Paraná"],
    "Chile": ["Santiago", "Valparaíso", "Biobío", "Maule", "Araucanía"],
    "Perú": ["Lima", "Arequipa", "La Libertad", "Piura", "Cusco"],
    "España": ["Madrid", "Cataluña", "Andalucía", "Valencia", "País Vasco"],
    "Estados Unidos": ["California", "Texas", "Nueva York", "Florida", "Illinois"],
    "Ecuador": ["Guayas", "Pichincha", "Azuay", "Manabí", "El Oro"],
    "Bolivia": ["La Paz", "Santa Cruz", "Cochabamba", "Oruro", "Tarija"],
}

GENEROS = ["Masculino", "Femenino", "Otro"]
GENERO_PROBS = [0.48, 0.48, 0.04]
ESTADOS = ["Activo", "Recuperado", "Fallecido"]
VACUNADO_OPCIONES = ["Sí", "No"]
SINTOMAS_OPCIONES = ["Asintomático", "Leve", "Moderado", "Grave"]

CATEGORICAS = ["pais", "region", "genero", "estado_paciente", "vacunado", "sintomas"]
NUMERICAS = ["edad", "casos_nuevos"]


# ---------------------------------------------------------------------------
# Generación de datos sintéticos
# ---------------------------------------------------------------------------
def _muestreo_categorico_vectorizado(pesos: np.ndarray, categorias: np.ndarray, rng) -> np.ndarray:
    """Dado un arreglo (n, k) de pesos por fila y k categorías, devuelve un
    arreglo (n,) con la categoría muestreada para cada fila según sus propios
    pesos (permite que la probabilidad varíe registro a registro)."""
    pesos = pesos / pesos.sum(axis=1, keepdims=True)
    acumulado = np.cumsum(pesos, axis=1)
    r = rng.random(pesos.shape[0])
    idx = (r[:, None] < acumulado).argmax(axis=1)
    return categorias[idx], idx


@st.cache_data(show_spinner=False)
def generar_datos(n_registros: int, semilla: int, simular_imperfecciones: bool = True) -> pd.DataFrame:
    """Genera un dataset sintético de COVID-19 con 9 columnas de tipos de
    datos variados y con correlaciones realistas entre variables:

    - fecha_reporte (datetime)
    - pais, region, genero (categóricas)
    - edad, casos_nuevos (numéricas)
    - vacunado, sintomas, estado_paciente (categóricas, dependientes de edad,
      fecha y estado de vacunación, no puramente aleatorias)

    Si `simular_imperfecciones=True`, se inyectan nulos y valores atípicos
    controlados (como en datos reales) para que el usuario pueda practicar
    limpieza/exploración de datos.
    """
    rng = np.random.default_rng(semilla)

    # --- Fechas y estructura temporal -------------------------------------
    fecha_inicio = pd.Timestamp("2020-03-01")
    fecha_fin = pd.Timestamp("2023-12-31")
    rango_dias = (fecha_fin - fecha_inicio).days
    dias_offset = rng.integers(0, rango_dias, size=n_registros)
    fechas = fecha_inicio + pd.to_timedelta(dias_offset, unit="D")

    # --- Geografía: país (ponderado por población) y región asociada ------
    pais = rng.choice(PAISES, size=n_registros, p=PAIS_PROBS)
    region = np.empty(n_registros, dtype=object)
    for nombre_pais, lista_regiones in REGIONES.items():
        mascara = pais == nombre_pais
        n_mask = int(mascara.sum())
        if n_mask:
            region[mascara] = rng.choice(lista_regiones, size=n_mask)

    genero = rng.choice(GENEROS, size=n_registros, p=GENERO_PROBS)

    # --- Edad: mezcla de grupos etarios (más realista que una sola normal) -
    grupo = rng.choice(["joven", "adulto", "mayor"], size=n_registros, p=[0.30, 0.45, 0.25])
    edad = np.empty(n_registros, dtype=float)
    edad[grupo == "joven"] = rng.normal(22, 6, size=(grupo == "joven").sum())
    edad[grupo == "adulto"] = rng.normal(45, 10, size=(grupo == "adulto").sum())
    edad[grupo == "mayor"] = rng.normal(72, 9, size=(grupo == "mayor").sum())
    edad = np.clip(edad, 0, 100).astype(int)

    # --- Vacunación: la probabilidad crece con el tiempo (rollout 2021+) --
    dias_desde_2021 = (fechas - pd.Timestamp("2021-01-01")).days.to_numpy()
    prob_vacunado = 1 / (1 + np.exp(-(dias_desde_2021 - 180) / 90))
    vacunado_bool = rng.random(n_registros) < prob_vacunado
    vacunado = np.where(vacunado_bool, "Sí", "No")

    # --- Síntomas: dependen de la edad y de si está vacunado --------------
    riesgo = np.clip((edad / 100) + np.where(vacunado_bool, -0.15, 0.15), 0, 1)
    pesos_sintomas = np.column_stack([
        np.clip(0.55 - riesgo * 0.4, 0.05, None),   # Asintomático
        np.full(n_registros, 0.35),                  # Leve
        0.15 + riesgo * 0.20,                         # Moderado
        riesgo * 0.30,                                 # Grave
    ])
    sintomas, idx_severidad = _muestreo_categorico_vectorizado(
        pesos_sintomas, np.array(SINTOMAS_OPCIONES), rng
    )
    severidad_norm = idx_severidad / (len(SINTOMAS_OPCIONES) - 1)  # 0..1

    # --- Estado del paciente: la mortalidad depende de edad, severidad y --
    # --- vacunación (mayor edad/severidad = más riesgo; vacunado = menos) -
    riesgo_muerte = np.clip(
        0.01 + (edad / 100) * 0.15 + severidad_norm * 0.15 - np.where(vacunado_bool, 0.06, 0.0),
        0.005, 0.55,
    )
    pesos_estado = np.column_stack([
        np.full(n_registros, 0.15),          # Activo
        1 - riesgo_muerte - 0.15,             # Recuperado
        riesgo_muerte,                         # Fallecido
    ])
    pesos_estado[:, 1] = np.clip(pesos_estado[:, 1], 0.05, None)
    estado_paciente, _ = _muestreo_categorico_vectorizado(
        pesos_estado, np.array(ESTADOS), rng
    )

    # --- Casos nuevos: base por país + estacionalidad + olas pandémicas ---
    # --- + efecto de menor reporte los fines de semana ---------------------
    lambda_base = rng.uniform(20, 80, size=len(PAISES))
    lambda_por_pais = pd.Series(dict(zip(PAISES, lambda_base)))
    lambda_pais_arr = pais_to_lambda = lambda_por_pais.reindex(pais).to_numpy()

    dia_anio = fechas.dayofyear.to_numpy()
    estacional = 1 + 0.3 * np.sin(2 * np.pi * dia_anio / 365)

    dias_totales = dias_offset.astype(float)
    ola1 = np.exp(-((dias_totales - 120) ** 2) / (2 * 60 ** 2))   # ola inicial (2020)
    ola2 = np.exp(-((dias_totales - 400) ** 2) / (2 * 80 ** 2))   # segunda ola (2021)
    ola3 = np.exp(-((dias_totales - 650) ** 2) / (2 * 70 ** 2))   # ola delta/ómicron
    factor_olas = 1 + 2.5 * (ola1 + ola2 + ola3)

    dia_semana = fechas.dayofweek.to_numpy()
    factor_reporte = np.where(dia_semana >= 5, 0.6, 1.0)  # menos reportes en fin de semana

    lambda_final = np.clip(lambda_pais_arr * estacional * factor_olas * factor_reporte, 1, None)
    casos_nuevos = rng.poisson(lam=lambda_final)

    df = pd.DataFrame({
        "fecha_reporte": fechas,
        "pais": pais,
        "region": region,
        "edad": edad,
        "genero": genero,
        "vacunado": vacunado,
        "sintomas": sintomas,
        "casos_nuevos": casos_nuevos,
        "estado_paciente": estado_paciente,
    })

    # --- Imperfecciones controladas (nulos y atípicos), como en datos reales
    if simular_imperfecciones:
        df["edad"] = df["edad"].astype(float)

        idx_null_edad = rng.choice(n_registros, size=int(n_registros * 0.02), replace=False)
        df.loc[idx_null_edad, "edad"] = np.nan

        idx_null_vac = rng.choice(n_registros, size=int(n_registros * 0.015), replace=False)
        df.loc[idx_null_vac, "vacunado"] = None

        idx_null_region = rng.choice(n_registros, size=int(n_registros * 0.01), replace=False)
        df.loc[idx_null_region, "region"] = None

        idx_outlier = rng.choice(n_registros, size=max(1, int(n_registros * 0.004)), replace=False)
        df.loc[idx_outlier, "edad"] = rng.choice([-5, 104, 109, 118], size=len(idx_outlier))

    df = df.sort_values("fecha_reporte").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Barra lateral: configuración de datos y filtros
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")

st.sidebar.subheader("1. Generación de datos sintéticos")
n_registros = st.sidebar.slider(
    "Número de registros", min_value=500, max_value=10000, value=5000, step=500
)
semilla = st.sidebar.number_input(
    "Semilla aleatoria (seed)", min_value=0, max_value=99999, value=42, step=1
)
simular_imperfecciones = st.sidebar.checkbox(
    "Simular imperfecciones reales (nulos y atípicos)", value=True,
    help="Inyecta valores faltantes y atípicos controlados en edad/región/vacunación, "
         "útil para practicar limpieza y exploración de datos.",
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
paises_sel = st.sidebar.multiselect("País", PAISES, default=PAISES)
regiones_todas = sorted({r for lista in REGIONES.values() for r in lista})
regiones_sel = st.sidebar.multiselect("Región", regiones_todas, default=regiones_todas)
generos_sel = st.sidebar.multiselect("Género", GENEROS, default=GENEROS)
estados_sel = st.sidebar.multiselect("Estado del paciente", ESTADOS, default=ESTADOS)
vacunados_sel = st.sidebar.multiselect("Vacunado", VACUNADO_OPCIONES, default=VACUNADO_OPCIONES)
sintomas_sel = st.sidebar.multiselect("Síntomas", SINTOMAS_OPCIONES, default=SINTOMAS_OPCIONES)
rango_edad = st.sidebar.slider(
    "Rango de edad", min_value=0, max_value=100, value=(0, 100)
)
incluir_nulos = st.sidebar.checkbox(
    "Incluir registros con región/vacunación nulas en los filtros", value=True
)

# Aplicar filtros
if len(rango_fechas) == 2:
    f_ini, f_fin = pd.Timestamp(rango_fechas[0]), pd.Timestamp(rango_fechas[1])
else:
    f_ini, f_fin = df["fecha_reporte"].min(), df["fecha_reporte"].max()

filtro_region = df["region"].isin(regiones_sel) | (incluir_nulos & df["region"].isna())
filtro_vacunado = df["vacunado"].isin(vacunados_sel) | (incluir_nulos & df["vacunado"].isna())

df_f = df[
    (df["fecha_reporte"] >= f_ini)
    & (df["fecha_reporte"] <= f_fin)
    & (df["pais"].isin(paises_sel))
    & filtro_region
    & (df["genero"].isin(generos_sel))
    & (df["estado_paciente"].isin(estados_sel))
    & filtro_vacunado
    & (df["sintomas"].isin(sintomas_sel))
    & (df["edad"].between(rango_edad[0], rango_edad[1]) | (incluir_nulos & df["edad"].isna()))
].copy()

# ---------------------------------------------------------------------------
# Encabezado y KPIs
# ---------------------------------------------------------------------------
st.title("🦠 Dashboard Interactivo COVID-19 (Datos Sintéticos)")
st.caption(
    "Los datos son 100% simulados dentro de la aplicación con NumPy/Pandas, "
    "solo con fines educativos y de demostración."
)

if df_f.empty:
    st.warning("No hay registros que coincidan con los filtros seleccionados.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Registros filtrados", f"{len(df_f):,}")
k2.metric("Casos nuevos (total)", f"{int(df_f['casos_nuevos'].sum()):,}")
k3.metric("Edad promedio", f"{df_f['edad'].mean():.1f} años")
tasa_fallecidos = (df_f["estado_paciente"] == "Fallecido").mean() * 100
k4.metric("Tasa de fallecidos", f"{tasa_fallecidos:.1f}%")
tasa_vacunados = (df_f["vacunado"] == "Sí").mean() * 100
k5.metric("Tasa de vacunación", f"{tasa_vacunados:.1f}%")

st.divider()

# ---------------------------------------------------------------------------
# Tabs principales
# ---------------------------------------------------------------------------
tab_datos, tab_cuant, tab_cual, tab_graf = st.tabs(
    ["📄 Vista general", "🔢 Estadística cuantitativa", "🔤 Estadística cualitativa", "📊 Gráficos dinámicos"]
)

# --- Tab 1: Vista general -------------------------------------------------
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
        file_name="covid_sintetico_filtrado.csv", mime="text/csv"
    )

# --- Tab 2: Estadística cuantitativa ---------------------------------------
with tab_cuant:
    st.subheader("Estadística descriptiva - Variables cuantitativas")
    var_num = st.selectbox("Selecciona una variable numérica", NUMERICAS, key="var_num")

    serie = df_f[var_num]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Media", f"{serie.mean():.2f}")
    c2.metric("Mediana", f"{serie.median():.2f}")
    c3.metric("Desv. estándar", f"{serie.std():.2f}")
    c4.metric("Mínimo", f"{serie.min():.0f}")
    c5.metric("Máximo", f"{serie.max():.0f}")
    c6.metric("Varianza", f"{serie.var():.2f}")

    with st.expander("Ver resumen estadístico completo (describe)"):
        st.dataframe(serie.describe().to_frame().T, use_container_width=True)

    st.markdown("#### Histograma con umbral personalizado")
    col_h1, col_h2 = st.columns([3, 1])
    with col_h2:
        n_bins = st.slider("Número de bins", 5, 100, 30)
        umbral_num = st.slider(
            "Umbral (línea de referencia)",
            float(serie.min()), float(serie.max()),
            float(serie.mean()),
        )
        color_hist = st.color_picker("Color del histograma", "#636EFA")
    with col_h1:
        fig_hist = px.histogram(
            df_f, x=var_num, nbins=n_bins,
            title=f"Distribución de {var_num}",
            color_discrete_sequence=[color_hist],
        )
        fig_hist.add_vline(
            x=umbral_num, line_dash="dash", line_color="red",
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
        y=umbral_num, line_dash="dash", line_color="red",
        annotation_text=f"Umbral = {umbral_num:.1f}",
    )
    st.plotly_chart(fig_box, use_container_width=True)

# --- Tab 3: Estadística cualitativa ----------------------------------------
with tab_cual:
    st.subheader("Estadística descriptiva - Variables cualitativas")
    var_cat = st.selectbox("Selecciona una variable categórica", CATEGORICAS, key="var_cat")

    conteo = df_f[var_cat].value_counts().rename("frecuencia").to_frame()
    conteo["porcentaje (%)"] = (conteo["frecuencia"] / conteo["frecuencia"].sum() * 100).round(2)

    c1, c2, c3 = st.columns(3)
    c1.metric("Categorías únicas", df_f[var_cat].nunique())
    c2.metric("Moda", df_f[var_cat].mode().iloc[0])
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

# --- Tab 4: Gráficos dinámicos ----------------------------------------------
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
            eje_y = st.selectbox("Variable eje Y", NUMERICAS, index=0)

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
            var_umbral_ref = eje_y if eje_y else "casos_nuevos"
            serie_ref = df_f[var_umbral_ref] if var_umbral_ref in NUMERICAS else df_f["casos_nuevos"]
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
                df_agg = df_f.groupby(pd.Grouper(key="fecha_reporte", freq="W"))[eje_y].sum().reset_index()
                fig = px.line(
                    df_agg, x="fecha_reporte", y=eje_y, template=plantilla,
                    title=titulo_custom, markers=True,
                )
            elif tipo_grafico == "Barras":
                df_agg = df_f.groupby(eje_x, as_index=False)[eje_y].sum()
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
                conteo_pie = df_f[eje_x].value_counts().reset_index()
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
st.caption("Dashboard construido con Streamlit + Plotly · Datos 100% sintéticos generados en tiempo de ejecución.")