import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Configuración de la página
st.set_page_config(
    page_title="Dashboard Modelo Insumo-Producto",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para estética
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1, h2, h3 {
        color: #FAFAFA;
    }
    .stPlotlyChart {
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Título Principal
st.title("📊 Análisis Económico: Modelo Insumo-Producto y Macroeconomía")
st.markdown("---")

# Rutas de archivos (Ajustar según entorno local)
PATH_EAIM = 'c:/Users/gabri/OneDrive/_Proyectos_Python/Modelo IP/Datos/EAIM.xlsx'
PATH_IGAE = 'c:/Users/gabri/OneDrive/_Proyectos_Python/Modelo IP/Datos/IGAE.xlsx'
PATH_DESEMPLEO = 'c:/Users/gabri/OneDrive/_Proyectos_Python/Modelo IP/Datos/Tasadedesempleo.xlsx'

@st.cache_data
def load_data():
    try:
        # Cargar EAIM
        df_eaim = pd.read_excel(PATH_EAIM, sheet_name='tr_eaim_cifra_2018_2023')
        
        # Cargar IGAE
        df_igae = pd.read_excel(PATH_IGAE)
        df_igae['Fecha'] = pd.to_datetime(df_igae['Fecha'])
        
        # Cargar Desempleo
        df_desempleo = pd.read_excel(PATH_DESEMPLEO)
        # Limpiar formato de fecha 'YYYY/MM'
        df_desempleo['Fecha'] = pd.to_datetime(df_desempleo['Periodos'], format='%Y/%m')
        col_desempleo = 'Serie desestacionalizada Porcentaje de la Población Económicamente Activa Mensual'
        df_desempleo.rename(columns={col_desempleo: 'Tasa_Desempleo'}, inplace=True)
        
        return df_eaim, df_igae, df_desempleo
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None, None, None

df_eaim, df_igae, df_desempleo = load_data()

if df_eaim is not None:
    # --- SIDEBAR ---
    st.sidebar.header("Configuración")
    anio_analisis = st.sidebar.selectbox("Seleccionar Año para Análisis Sectorial", sorted(df_eaim['ANIO'].unique(), reverse=True))
    
    # --- SECCIÓN 1: MACROECONOMÍA ---
    st.header("1. Contexto Macroeconómico Histórico")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Actividad Económica (IGAE)")
        fig_igae = px.line(df_igae, x='Fecha', y='IGAE Total_ crecimiento anual', 
                           title='Crecimiento Anual del IGAE',
                           template='plotly_dark',
                           color_discrete_sequence=['#00CC96'])
        fig_igae.update_layout(hovermode="x unified")
        st.plotly_chart(fig_igae, use_container_width=True)
        
    with col2:
        st.subheader("Tasa de Desempleo")
        fig_desempleo = px.line(df_desempleo, x='Fecha', y='Tasa_Desempleo', 
                                title='Tasa de Desempleo (% PEA)',
                                template='plotly_dark',
                                color_discrete_sequence=['#EF553B'])
        fig_desempleo.update_layout(hovermode="x unified")
        st.plotly_chart(fig_desempleo, use_container_width=True)

    # --- SECCIÓN 2: ANÁLISIS SECTORIAL (MIP) ---
    st.markdown("---")
    st.header(f"2. Análisis Sectorial Insumo-Producto ({anio_analisis})")
    
    # Procesamiento de datos sectoriales
    cols_interes = ['CODIGO_ACTIVIDAD', 'ANIO', 'PBT', 'K000A', 'VAT', 'POTOT']
    df_ip = df_eaim[cols_interes].copy()
    cols_num = ['PBT', 'K000A', 'VAT', 'POTOT']
    for col in cols_num:
        df_ip[col] = pd.to_numeric(df_ip[col], errors='coerce')
    df_ip.dropna(subset=cols_num, inplace=True)
    
    df_anio = df_ip[df_ip['ANIO'] == anio_analisis].copy()
    
    # Cálculos MIP
    df_anio['Coef_Insumo'] = df_anio['K000A'] / df_anio['PBT']
    df_anio['Coef_ValorAgregado'] = df_anio['VAT'] / df_anio['PBT']
    df_anio['Productividad_Laboral'] = df_anio['PBT'] / df_anio['POTOT']
    
    # PCA y Clustering
    features = ['Coef_Insumo', 'Coef_ValorAgregado', 'Productividad_Laboral']
    X = df_anio[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X_scaled)
    df_anio['PC1'] = principalComponents[:, 0]
    df_anio['PC2'] = principalComponents[:, 1]
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_anio['Cluster'] = kmeans.fit_predict(X_scaled)
    df_anio['Cluster'] = df_anio['Cluster'].astype(str) # Para que sea categórico en el gráfico
    
    # Visualización PCA
    col3, col4 = st.columns([2, 1])
    
    with col3:
        st.subheader("Mapa de Tipologías Industriales (PCA + Clustering)")
        fig_pca = px.scatter(df_anio, x='PC1', y='PC2', color='Cluster',
                             hover_data=['CODIGO_ACTIVIDAD', 'Coef_Insumo', 'Coef_ValorAgregado'],
                             title='Distribución de Sectores por Estructura Productiva',
                             template='plotly_dark',
                             color_discrete_sequence=px.colors.qualitative.Bold)
        fig_pca.update_traces(marker=dict(size=10, opacity=0.8))
        st.plotly_chart(fig_pca, use_container_width=True)
        
    with col4:
        st.subheader("Perfil de los Clusters")
        cluster_summary = df_anio.groupby('Cluster')[features].mean().reset_index()
        
        # Radar Chart para perfiles
        categories = features
        fig_radar = go.Figure()

        for i, row in cluster_summary.iterrows():
            # Escalar valores para el radar (solo visualización)
            # Nota: Esto es simplificado, idealmente se normaliza 0-1
            values = row[categories].values.flatten().tolist()
            values += values[:1] # Cerrar el loop
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=f'Cluster {row["Cluster"]}'
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                visible=True,
                range=[0, max(cluster_summary[features].max())]
                )),
            showlegend=True,
            template='plotly_dark',
            title="Características Promedio"
        )
        # Como las escalas son muy diferentes (Productividad es miles, Coeficientes < 1), 
        # mejor usamos un Heatmap o Bar Chart normalizado
        
        # Heatmap alternativo más claro
        cluster_summary_melt = cluster_summary.melt(id_vars='Cluster', var_name='Variable', value_name='Valor Promedio')
        fig_bar = px.bar(cluster_summary_melt, x='Variable', y='Valor Promedio', color='Cluster', barmode='group',
                         template='plotly_dark', title="Comparativa de Clusters")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Tabla de Datos
    st.markdown("---")
    st.subheader("Datos Detallados por Sector")
    st.dataframe(df_anio[['CODIGO_ACTIVIDAD', 'PBT', 'Coef_Insumo', 'Coef_ValorAgregado', 'Productividad_Laboral', 'Cluster']], use_container_width=True)

else:
    st.warning("No se pudieron cargar los datos. Verifique las rutas de los archivos.")
