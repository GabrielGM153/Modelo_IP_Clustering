import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
st.set_page_config(
    page_title="Dashboard Económico MIP",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos para gráficos
plt.style.use('ggplot')

# ==========================================
# 1. CARGA DE DATOS 
# ==========================================
@st.cache_data
def load_data():
    # Rutas:
    path_eaim = 'C:/Users/gabri/OneDrive/_Proyectos_Python/Modelo IP/Datos/EAIM.xlsx'
    path_igae = 'C:/Users/gabri/OneDrive/_Proyectos_Python/Modelo IP/Datos/IGAE.xlsx'
    path_desempleo = 'C:/Users/gabri/OneDrive/_Proyectos_Python/Modelo IP/Datos/Tasadedesempleo.xlsx'

    try:
        # Intentamos cargar los archivos reales
        df_eaim = pd.read_excel(path_eaim, sheet_name='tr_eaim_cifra_2018_2023')
        df_igae = pd.read_excel(path_igae)
        df_desempleo = pd.read_excel(path_desempleo)
        
        # Pequeña limpieza si carga los reales
        if 'Fecha' in df_igae.columns:
            df_igae['Fecha'] = pd.to_datetime(df_igae['Fecha'])
            df_igae['ANIO'] = df_igae['Fecha'].dt.year
            
        return df_eaim, df_igae, df_desempleo, "Reales"

    except FileNotFoundError:
        
        
        np.random.seed(42)
        sectores = [f"Sector_{i}" for i in range(100)]
        anios = [2018, 2019, 2020, 2021, 2022, 2023]
        data_eaim = []
        
        for anio in anios:
            for sec in sectores:
                pbt = np.random.uniform(1e6, 1e8)
                k000a = pbt * np.random.uniform(0.4, 0.8) 
                vat = pbt - k000a
                potot = pbt / np.random.uniform(1000, 5000)
                data_eaim.append([sec, anio, pbt, k000a, vat, potot])
                
        df_eaim = pd.DataFrame(data_eaim, columns=['CODIGO_ACTIVIDAD', 'ANIO', 'PBT', 'K000A', 'VAT', 'POTOT'])
        
        fechas = pd.date_range(start='2018-01-01', end='2023-12-01', freq='MS')
        df_igae = pd.DataFrame({
            'Fecha': fechas,
            'ANIO': fechas.year,
            'IGAE Total_ crecimiento anual': np.random.normal(2, 3, len(fechas))
        })
        df_igae.loc[df_igae['Fecha'].dt.year == 2020, 'IGAE Total_ crecimiento anual'] -= 8
        
        df_desempleo = pd.DataFrame({
            'Periodos': fechas.strftime('%Y/%m'),
            'Serie desestacionalizada Porcentaje de la Población Económicamente Activa Mensual': np.random.uniform(3, 5, len(fechas))
        })
        
        return df_eaim, df_igae, df_desempleo, "Sintéticos"

# Ejecutar carga
df_eaim, df_igae, df_desempleo, tipo_datos = load_data()

# Sidebar de información
with st.sidebar:
    st.title("Configuración")
    if tipo_datos == "Reales":
        st.success("Conectado a Base de Datos Local (Excel)")
    else:
        st.warning("Usando Datos Simulados (No se encontraron los archivos en la ruta especificada)")
    
    st.info("Este tablero permite analizar la estructura productiva bajo la óptica de Leontief asistida por algoritmos de agrupamiento.")

# ==========================================
# 2. CUERPO PRINCIPAL
# ==========================================

st.title("🇲🇽 Análisis Estructural de la Industria Manufacturera")
st.markdown("### Convergencia entre Modelos Económicos y Ciencia de Datos")

tab1, tab2, tab3 = st.tabs(["Micro: Clustering Sectorial", "Macro: Ciclos Económicos", "Marco Teórico"])

# ------------------------------------------------------------------
# TAB 1: MICROECONOMÍA (CLUSTERING)
# ------------------------------------------------------------------
with tab1:
    col_desc, col_filtros = st.columns([3, 1])
    with col_desc:
        st.markdown("""
        **Objetivo:** Identificar patrones latentes en la función de producción de los sectores.
        En lugar de clasificar por "nombre" (ej. Alimentos, Textiles), clasificamos por **comportamiento matemático** (eficiencia, uso de insumos, valor agregado).
        """)
    
    # Preparación de datos (Año base 2022 o el último disponible)
    anio_analisis = 2022
    df_cluster = df_eaim[df_eaim['ANIO'] == anio_analisis].copy()
    
    # Feature Engineering (Creación de variables económicas)
    df_cluster['Coef_Insumo'] = df_cluster['K000A'] / df_cluster['PBT']
    df_cluster['Coef_VA'] = df_cluster['VAT'] / df_cluster['PBT']
    df_cluster['Productividad'] = df_cluster['PBT'] / df_cluster['POTOT']
    
    # Limpieza de infinitos o nulos generados por división entre cero
    df_cluster = df_cluster.replace([np.inf, -np.inf], np.nan).dropna(subset=['Coef_Insumo', 'Coef_VA', 'Productividad'])

    # Sidebar interno para el modelo
    with col_filtros:
        st.subheader("Parámetros K-Means")
        k = st.slider("Número de Clusters:", 2, 6, 4)
    
    # Modelado
    features = ['Coef_Insumo', 'Coef_VA', 'Productividad']
    X = df_cluster[features]
    
    # Estandarización (Z-Score)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA para visualización 2D
    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(pca_res, columns=['PC1', 'PC2'])
    
    # Clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_cluster['Cluster'] = clusters
    df_pca['Cluster'] = clusters
    
    # --- Visualización ---
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Mapa de Similitud Productiva (PCA)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            x='PC1', y='PC2', 
            hue='Cluster', 
            data=df_pca, 
            palette='viridis', 
            s=100, alpha=0.8, 
            edgecolor='k',
            ax=ax
        )
        # Centroides
        # (Nota: Los centroides están en el espacio original, habría que transformarlos para pintarlos aquí, 
        #  por simplicidad visualizamos solo los puntos)
        
        plt.title(f'Sectores Agrupados por Tecnología de Producción (k={k})')
        plt.xlabel(f'Componente Principal 1 (Varianza: {pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'Componente Principal 2 (Varianza: {pca.explained_variance_ratio_[1]:.1%})')
        st.pyplot(fig)
        st.caption("*Cada punto es un sector industrial. Puntos cercanos comparten estructuras de costos y productividad similares.*")

    with c2:
        st.subheader("ADN de los Clusters")
        # Promedios por cluster
        resumen = df_cluster.groupby('Cluster')[features].mean()
        
        # Formateo para mostrar
        st.dataframe(resumen.style.background_gradient(cmap='Blues').format("{:.2f}"))
        
        st.markdown("#### Interpretación Económica")
        # Lógica dinámica simple de interpretación
        max_prod = resumen['Productividad'].idxmax()
        max_ins = resumen['Coef_Insumo'].idxmax()
        
        st.success(f"**Cluster {max_prod}: Alta Tecnología / Capital Intensivo**\nMuestra la mayor productividad laboral. Probablemente industrias muy automatizadas.")
        st.warning(f"**Cluster {max_ins}: Ensamble / Manufactura Básica**\nAlta dependencia de insumos intermedios y menor valor agregado.")

# ------------------------------------------------------------------
# TAB 2: MACROECONOMÍA 
# ------------------------------------------------------------------
with tab2:
    st.header("Dinámica Histórica: Manufactura vs Economía Nacional")
    st.markdown("""
    Para entender la relación real, hemos normalizado los datos utilizando **Números Índice (Base 100 = 2018)**.
    Esto permite comparar la velocidad de recuperación de la manufactura frente al IGAE general.
    """)
    
    # 1. Procesamiento de datos para la gráfica
    # Agrupar IGAE por año (promedio del crecimiento o nivel)
    # NOTA: El IGAE que tienes es "Variación anual". Para hacerlo índice necesitamos reconstruir una serie base 100.
    # Como aproximación, usaremos la Producción Manufacturera Total y simularemos un índice para el IGAE
    # asumiendo que el 2018 es el 100 y aplicando las tasas de crecimiento.
    
    # A) Manufactura
    manuf_anual = df_eaim.groupby('ANIO')['PBT'].sum().reset_index()
    base_manuf = manuf_anual.loc[manuf_anual['ANIO'] == 2018, 'PBT'].values[0]
    manuf_anual['Indice_Manuf'] = (manuf_anual['PBT'] / base_manuf) * 100
    
    # B) IGAE (Reconstrucción de índice simple)
    igae_anual = df_igae.groupby('ANIO')['IGAE Total_ crecimiento anual'].mean().reset_index()
    # Creamos un índice artificial comenzando en 100 y aplicando los crecimientos promedio
    igae_index = [100]
    for x in igae_anual['IGAE Total_ crecimiento anual'].iloc[1:]:
        igae_index.append(igae_index[-1] * (1 + x/100))
    # Ajustamos longitud si es necesario o usamos mapeo directo si los años coinciden
    # Para simplificar visualización directa de la tasa proporcionada:
    
    # UNIÓN
    df_macro = pd.merge(manuf_anual, igae_anual, on='ANIO')
    
    # --- GRÁFICO 1: CO-MOVIMIENTO (Índices o Tasas) ---
    # Vamos a graficar Producción (Barras) vs IGAE (Línea) para ver correlación visual
    
    fig2, ax1 = plt.subplots(figsize=(12, 6))
    
    # Eje izquierdo: Producción Monetaria
    color1 = '#1f77b4' # Azul
    ax1.set_xlabel('Año', fontsize=12)
    ax1.set_ylabel('Producción Manufacturera (Miles de Millones $)', color=color1, fontsize=12)
    bars = ax1.bar(df_macro['ANIO'], df_macro['PBT']/1e9, color=color1, alpha=0.6, label='Producción Manuf. (Nivel)')
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Eje derecho: IGAE (Ciclo económico)
    ax2 = ax1.twinx()
    color2 = '#d62728' # Rojo
    ax2.set_ylabel('IGAE (Tasa de Crecimiento %)', color=color2, fontsize=12)
    line = ax2.plot(df_macro['ANIO'], df_macro['IGAE Total_ crecimiento anual'], color=color2, linewidth=3, marker='o', label='Ciclo Económico (IGAE)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Línea cero para el IGAE
    ax2.axhline(0, color='gray', linestyle='--', linewidth=1)
    
    plt.title('Resiliencia Industrial: Producción vs Ciclo Económico', fontsize=16)
    st.pyplot(fig2)
    
    st.markdown("""
    **¿Qué estamos viendo?**
    * **Las barras azules** representan el tamaño real de la producción manufacturera (dinero).
    * **La línea roja** representa la "salud" de la economía general (crecimiento del IGAE).
    * **Análisis:** Si las barras caen cuando la línea roja cae (ej. 2020), el sector es **procíclico**. Si las barras crecen más rápido que la línea roja en la recuperación, el sector es un **motor de reactivación**.
    """)
    
    # --- GRÁFICO 2: CORRELACIÓN ---
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Matriz de Correlación (Heatmap)")
        
        # Procesamos desempleo para unirlo
        df_desemp_anual = df_desempleo.copy()
        # Extraer año si es string 'YYYY/MM' o si es fecha
        try:
            if df_desemp_anual['Periodos'].dtype == 'object':
                df_desemp_anual['ANIO'] = df_desemp_anual['Periodos'].astype(str).str[:4].astype(int)
            else:
                df_desemp_anual['ANIO'] = pd.to_datetime(df_desemp_anual['Periodos']).dt.year
        except:
             # Fallback simple si la columna tiene otro nombre en el excel real
             st.error("Verificar nombre de columna fecha en Desempleo")
             df_desemp_anual['ANIO'] = 2018 # Dummy
             
        # Buscamos la columna de tasa (asumimos que es la segunda columna si hay duda)
        col_tasa = [c for c in df_desempleo.columns if 'Porcentaje' in c or 'Tasa' in c][0]
        desemp_agrupado = df_desemp_anual.groupby('ANIO')[col_tasa].mean().reset_index()
        
        df_corr = pd.merge(df_macro, desemp_agrupado, on='ANIO')
        
        # Calculamos correlación
        corr_matrix = df_corr[['PBT', 'IGAE Total_ crecimiento anual', col_tasa]].corr()
        
        fig3, ax3 = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", ax=ax3)
        st.pyplot(fig3)
        
    with c2:
        st.markdown("#### Interpretación Estadística")
        st.info("""
        1. **Correlación PBT vs IGAE:**
           - Cercana a +1: La manufactura es el motor de la economía.
           - Cercana a 0: La manufactura se mueve independiente al país.
        
        2. **Correlación PBT vs Desempleo:**
           - Esperamos que sea **Negativa** (Ley de Okun): A mayor producción, menor desempleo.
           - Si es positiva, indica una "Recuperación sin empleo" (aumento de productividad por automatización, no por contratación).
        """)

# ------------------------------------------------------------------
# TAB 3: MARCO TEÓRICO PROFUNDO
# ------------------------------------------------------------------
with tab3:
    st.header("Integración Epistemológica: Economía Estructural y Ciencia de Datos")
    
    st.markdown("""
    Este proyecto no es una simple visualización; es un ejercicio de **Econometría Estructural asistida por Aprendizaje de Máquina**. A continuación se detalla la convergencia de ambas disciplinas.
    """)
    
    expander1 = st.expander("1. De la Contabilidad Nacional a los Vectores de Características", expanded=True)
    with expander1:
        st.markdown(r"""
        **Teoría Económica (Wassily Leontief):**
        La economía asume que la producción de un sector $j$ ($X_j$) está determinada por una función de producción de coeficientes fijos:
        
        $$ X_j = \sum_{i} x_{ij} + V_j $$
        
        Donde $x_{ij}$ son los insumos intermedios y $V_j$ el valor agregado (trabajo + capital).
        
        **Aporte de Data Science (Feature Engineering):**
        Para el algoritmo, los valores monetarios absolutos introducen sesgo de escala (sectores grandes vs pequeños). La Ciencia de Datos normaliza esto transformando la identidad contable en **Coeficientes Técnicos ($a_{ij}$)**:
        
        $$ a_{ij} = \frac{x_{ij}}{X_j} $$
        
        Esto convierte a cada sector industrial en un vector adimensional comparable en un espacio n-dimensional, permitiendo comparar una tortillería con una armadora de autos basándose puramente en su **"receta" de producción**.
        """)
        
    expander2 = st.expander("2. Reducción de Dimensionalidad (PCA) en Economía")
    with expander2:
        st.markdown("""
        **El Problema Económico:**
        Analizar 200 ramas industriales con 10 variables cada una implica un espacio de 2000 dimensiones. El cerebro humano y los modelos de regresión lineal clásica (OLS) sufren con la multicolinealidad.
        
        **La Solución Algorítmica (PCA):**
        El Análisis de Componentes Principales descompone la matriz de covarianza de los sectores para encontrar "ejes latentes".
        * **PC1 (Eje X):** Generalmente captura la **Intensidad de Insumos**. Sectores a la derecha requieren cadenas de suministro masivas.
        * **PC2 (Eje Y):** Generalmente captura la **Generación de Valor**. Sectores arriba dependen más del capital humano o propiedad intelectual.
        """)
        
    expander3 = st.expander("3. Clustering K-Means vs Clasificación SCIAN/NAICS")
    with expander3:
        st.markdown(r"""
        **Clasificación Tradicional (SCIAN):**
        Agrupa por "tipo de producto" (ej. Zapatos de cuero y Bolsos de cuero van juntos). Es subjetiva y a priori.
        
        **Segmentación No Supervisada (K-Means):**
        Agrupa por "estructura de costos". 
        El algoritmo minimiza la inercia intra-cluster (Varianza interna):
        
        $$ J = \sum_{i=1}^{k} \sum_{x \in S_i} ||x - \mu_i||^2 $$
        
        **Hallazgo:** Un algoritmo puede agrupar la *Industria Farmacéutica* con la *Industria Aeroespacial* en el mismo cluster (Alta Productividad/Alto Valor Agregado), aunque produzcan cosas totalmente distintas. Esto es invaluable para diseñar **política industrial transversal**.
        """)

st.markdown("---")
st.caption("Desarrollado para el Análisis Económico | Economía Aplicada y Ciencia de Datos")