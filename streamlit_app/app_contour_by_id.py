"""
Aplicación Streamlit para interpolación 2D de parámetros geotécnicos
usando plot_contour_between_id_minmax.

Esta aplicación genera mapas de contorno limitados por polígonos formados
a partir de las cotas mínimas y máximas por cada ID de sondeo.

Características:
- Carga de archivo único (CSV o Excel) con X, Y, parámetros e ID
- Mapeo interactivo de columnas (X, Y, Z, ID)
- Visualización de contornos limitados por polígono min/max por ID
- Configuración completa de parámetros de interpolación
- Exportación de figuras PNG, grillas CSV y datos del polígono
- Botón para cargar ejemplo de datos

Autor: Camilo Gardi
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import json

# Importar funciones de utils
from utils import (
    read_table,
    normalize_column_names,
    get_numeric_columns,
    plot_contour_between_id_minmax,
    export_interpolated_grid_to_csv,
    figure_to_bytes,
    polygon_to_geojson
)

# Configuración de página
st.set_page_config(
    page_title="Contornos por Sondeo - Interpolación 2D",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("🗺️ Interpolación 2D por Sondeos - Desarrollado po Camilo Garcia")
st.markdown("""
Esta aplicación genera **mapas de contorno** de parámetros geotécnicos limitados
por un **polígono envolvente** basado en las cotas mínimas y máximas de cada sondeo.


### 📋 Características principales:
- ✅ Interpolación limitada por polígono min/max por ID de sondeo
- ✅ Soporte para múltiples parámetros
- ✅ Configuración completa de interpolación (método, resolución, niveles)
- ✅ Exportación de PNG, CSV y datos del polígono
- ✅ Ejemplo de datos incluido

### 📁 Requisitos del archivo:
- **Formato**: CSV o Excel
- **Columnas requeridas**:
  - **X** (abscisa/coordenada Este)
  - **Y** (cota/elevación)
  - **ID** (identificador de sondeo) - **OBLIGATORIO**
  - Una o más columnas con parámetros a interpolar
""")

# ============================================================================
# SECCIÓN 1: CARGA DE ARCHIVO O EJEMPLO
# ============================================================================
st.header("📁 Paso 1: Cargar datos")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Sube tu archivo con datos de sondeos (CSV o Excel)",
        type=['csv', 'xls', 'xlsx'],
        help="El archivo debe contener columnas X, Y, ID y uno o más parámetros numéricos"
    )

with col2:
    st.markdown("&nbsp;")  # Espaciado
    load_example = st.button(
        "📂 Cargar ejemplo",
        type="secondary",
        use_container_width=True,
        help="Carga un archivo de ejemplo para probar la aplicación"
    )

# Manejar carga de ejemplo
df_raw = None
file_source = None

if load_example:
    try:
        example_path = "streamlit_app/examples/example_table.csv"
        df_raw = pd.read_csv(example_path)
        file_source = "ejemplo"
        st.success(f"✅ Ejemplo cargado: {len(df_raw)} filas, {len(df_raw.columns)} columnas")
        with st.expander("ℹ️ Información del ejemplo"):
            st.markdown("""
            **Archivo de ejemplo:** `example_table.csv`
            
            Contiene datos sintéticos de 10 sondeos (P-01 a P-10) con:
            - **abscisa**: Coordenada X (100-200m)
            - **cota**: Coordenada Y/elevación (45-52m)
            - **id**: Identificador de sondeo
            - **Parámetros**: qc, gamma, LL, IP, humedad
            
            Este ejemplo es ideal para probar la funcionalidad de la aplicación.
            """)
    except Exception as e:
        st.error(f"Error al cargar el ejemplo: {str(e)}")
        st.stop()

elif uploaded_file is not None:
    try:
        with st.spinner("Leyendo archivo..."):
            df_raw = read_table(uploaded_file)
            file_source = "subido"
        st.success(f"✅ Archivo cargado: {len(df_raw)} filas, {len(df_raw.columns)} columnas")
    except Exception as e:
        st.error(f"❌ Error al leer el archivo: {str(e)}")
        if "openpyxl" in str(e).lower():
            st.info("💡 Instala openpyxl con: `pip install openpyxl`")
        elif "xlrd" in str(e).lower():
            st.info("💡 Instala xlrd con: `pip install xlrd` o exporta el archivo a CSV")
        st.stop()

if df_raw is None:
    st.info("👆 Por favor, sube un archivo o carga el ejemplo para continuar")
    st.stop()

# Normalizar nombres de columnas
df = normalize_column_names(df_raw)

# Vista previa
with st.expander("🔍 Vista previa de datos", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"Mostrando primeras 10 filas de {len(df)} totales")

# ============================================================================
# SECCIÓN 2: MAPEO DE COLUMNAS
# ============================================================================
st.header("🗂️ Paso 2: Mapear columnas")

st.markdown("""
Indica qué columnas contienen las coordenadas **X**, **Y**, el **ID de sondeo** 
y qué **parámetro(s)** deseas interpolar.
""")

# Detectar columnas candidatas
col_options = list(df.columns)

# Detectar X (abscisa, este)
x_candidates = [c for c in col_options if any(k in c.lower() for k in ['x', 'abscisa', 'este', 'easting'])]
x_default = x_candidates[0] if x_candidates else col_options[0]

# Detectar Y (cota, elevación, norte)
y_candidates = [c for c in col_options if any(k in c.lower() for k in ['y', 'cota', 'elevacion', 'elevation', 'norte', 'northing'])]
y_default = y_candidates[0] if y_candidates else (col_options[1] if len(col_options) > 1 else col_options[0])

# Detectar ID
id_candidates = [c for c in col_options if any(k in c.lower() for k in ['id', 'nombre', 'name', 'sondeo', 'sondaje', 'drilling'])]
id_default = id_candidates[0] if id_candidates else (col_options[2] if len(col_options) > 2 else col_options[0])

# UI para selección de columnas
col1, col2, col3 = st.columns(3)

with col1:
    x_col = st.selectbox(
        "Columna X (abscisa) *",
        options=col_options,
        index=col_options.index(x_default),
        help="Coordenada X o abscisa (coordenada Este)"
    )

with col2:
    y_col = st.selectbox(
        "Columna Y (cota/elevación) *",
        options=col_options,
        index=col_options.index(y_default),
        help="Coordenada Y, cota o elevación"
    )

with col3:
    id_col = st.selectbox(
        "Columna ID (sondeo) *",
        options=col_options,
        index=col_options.index(id_default),
        help="Identificador de sondeo/punto (OBLIGATORIO para esta función)"
    )

# Validar que las columnas sean diferentes
if len(set([x_col, y_col, id_col])) < 3:
    st.error("❌ Las columnas X, Y e ID deben ser diferentes. Selecciona columnas distintas.")
    st.stop()

# Obtener columnas de parámetros disponibles (excluyendo X, Y, ID)
exclude_cols = [x_col, y_col, id_col]
param_cols = get_numeric_columns(df, exclude=exclude_cols)

if not param_cols:
    st.error("❌ No se encontraron columnas de parámetros numéricos para interpolar.")
    st.stop()

st.success(f"✅ Columnas mapeadas correctamente. {len(param_cols)} parámetros disponibles: {', '.join(param_cols)}")

# ============================================================================
# SECCIÓN 3: SELECCIÓN DE PARÁMETROS
# ============================================================================
st.header("📊 Paso 3: Seleccionar parámetro(s) a interpolar")

selected_params = st.multiselect(
    "Selecciona uno o más parámetros",
    options=param_cols,
    default=param_cols[:1],  # Seleccionar el primero por defecto
    help="Se generará un mapa de contorno independiente por cada parámetro seleccionado"
)

if not selected_params:
    st.info("👆 Selecciona al menos un parámetro para continuar")
    st.stop()

# ============================================================================
# SECCIÓN 4: VALIDACIÓN Y ESTADÍSTICAS
# ============================================================================
st.header("📈 Paso 4: Validación de datos")

# Validar datos: eliminar filas con X, Y o ID faltantes
df_clean = df[[x_col, y_col, id_col] + selected_params].copy()
initial_count = len(df_clean)
df_clean = df_clean.dropna(subset=[x_col, y_col, id_col])
final_count = len(df_clean)

if initial_count > final_count:
    st.warning(f"⚠️ Se eliminaron {initial_count - final_count} filas con valores faltantes en X, Y o ID")

# Verificar número mínimo de puntos y sondeos
n_points = len(df_clean)
n_ids = df_clean[id_col].nunique()

if n_points < 3:
    st.error(f"❌ Se requieren al menos 3 puntos válidos para interpolar. Actualmente: {n_points}")
    st.stop()

if n_ids < 2:
    st.error(f"❌ Se requieren al menos 2 sondeos (IDs únicos) para generar el polígono. Actualmente: {n_ids}")
    st.stop()

# Mostrar estadísticas
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Puntos válidos", f"{n_points}")
with col2:
    st.metric("Sondeos únicos", f"{n_ids}")
with col3:
    avg_points_per_id = n_points / n_ids
    st.metric("Promedio puntos/sondeo", f"{avg_points_per_id:.1f}")

# Estadísticas de parámetros seleccionados
with st.expander("📊 Estadísticas de parámetros seleccionados"):
    stats_df = df_clean[selected_params].describe().T
    stats_df['missing'] = df_clean[selected_params].isna().sum()
    st.dataframe(stats_df, use_container_width=True)

# ============================================================================
# SECCIÓN 5: CONFIGURACIÓN DE INTERPOLACIÓN
# ============================================================================
st.header("⚙️ Paso 5: Configurar interpolación")

st.markdown("Ajusta los parámetros de interpolación en el **panel lateral** →")

# Sidebar con configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    st.subheader("🔲 Resolución de grilla")
    nx = st.slider("Resolución X (nx)", 50, 500, 300, 50, help="Número de puntos en dirección X")
    ny = st.slider("Resolución Y (ny)", 50, 500, 300, 50, help="Número de puntos en dirección Y")
    
    # Advertencia de resolución excesiva
    total_points = nx * ny
    if total_points > 1_000_000:
        st.error(f"⚠️ Resolución muy alta: {total_points:,} puntos. Recomendado < 1,000,000")
        st.warning("Reduce nx o ny para evitar problemas de memoria/rendimiento")
    elif total_points > 500_000:
        st.warning(f"⚠️ Resolución alta: {total_points:,} puntos. Puede ser lento.")
    
    st.subheader("🎨 Método de interpolación")
    prefer_method = st.radio(
        "Método preferido",
        options=['cubic', 'linear'],
        index=0,
        help="Cubic es más suave pero puede fallar; si falla, usa linear automáticamente"
    )
    
    clip_to_range = st.checkbox(
        "Recortar al rango de datos",
        value=True,
        help="Evita overshooting en la interpolación (valores fuera del rango real)"
    )
    
    st.subheader("📐 Niveles de contorno")
    n_levels = st.slider("Número de niveles", 5, 30, 14, 1, help="Número de niveles de contorno a mostrar")
    
    st.subheader("🎨 Visualización")
    cmap = st.selectbox(
        "Mapa de colores",
        options=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                'coolwarm', 'RdYlBu_r', 'Spectral_r', 'jet'],
        index=0
    )
    
    scatter_size = st.slider("Tamaño puntos datos", 5, 20, 8, 1)
    
    invert_yaxis = st.checkbox(
        "Invertir eje Y",
        value=False,
        help="Útil si Y representa profundidad (mayor valor = más profundo)"
    )
    
    st.subheader("📏 Límites de Y (opcional)")
    apply_y_limits = st.checkbox("Aplicar límites de Y", value=False)
    
    y_limits = None
    if apply_y_limits:
        y_min_data = float(df_clean[y_col].min())
        y_max_data = float(df_clean[y_col].max())
        
        y_min = st.number_input(
            "Y mínimo",
            value=y_min_data,
            help="Límite inferior para el eje Y"
        )
        y_max = st.number_input(
            "Y máximo",
            value=y_max_data,
            help="Límite superior para el eje Y"
        )
        
        if y_min >= y_max:
            st.error("Y mínimo debe ser menor que Y máximo")
        else:
            y_limits = (y_min, y_max)
    
    st.subheader("📐 Tamaño de figura")
    figsize_width = st.slider("Ancho (pulgadas)", 6, 20, 10, 1)
    figsize_height = st.slider("Alto (pulgadas)", 4, 16, 6, 1)
    figsize = (figsize_width, figsize_height)

# ============================================================================
# SECCIÓN 6: GENERAR CONTORNOS
# ============================================================================
st.markdown("---")
generate_button = st.button("🚀 Generar mapas de contorno", type="primary", use_container_width=True)

if not generate_button:
    st.info("👆 Cuando estés listo, presiona el botón 'Generar mapas de contorno'")
    st.stop()

# ============================================================================
# SECCIÓN 7: GENERACIÓN E INTERPOLACIÓN
# ============================================================================
st.header("📊 Paso 6: Mapas de contorno generados")

# Generar contornos para cada parámetro seleccionado
for idx, param in enumerate(selected_params, start=1):
    
    st.subheader(f"Parámetro {idx}/{len(selected_params)}: {param}")
    
    # Filtrar datos válidos para este parámetro
    df_param = df_clean[[x_col, y_col, id_col, param]].dropna(subset=[param])
    
    if len(df_param) < 3:
        st.warning(f"⚠️ Parámetro '{param}': solo {len(df_param)} puntos válidos. Se omite (mínimo 3).")
        continue
    
    # Verificar número de IDs
    n_ids_param = df_param[id_col].nunique()
    if n_ids_param < 2:
        st.warning(f"⚠️ Parámetro '{param}': solo {n_ids_param} sondeo(s) con datos. Se requieren al menos 2.")
        continue
    
    # Generar contorno usando plot_contour_between_id_minmax
    try:
        with st.spinner(f"Generando contorno para '{param}'..."):
            fig, ax, poly = plot_contour_between_id_minmax(
                df_param,
                x_col=x_col,
                y_col=y_col,
                z_col=param,
                id_col=id_col,
                y_limits=y_limits,
                n_levels=n_levels,
                nx=nx,
                ny=ny,
                cmap=cmap,
                clip_to_range=clip_to_range,
                scatter_size=scatter_size,
                title=f'Mapa de contorno: {param} (min/max por {id_col})',
                figsize=figsize,
                prefer_method=prefer_method
            )
            
            # Invertir eje Y si se requiere
            if invert_yaxis:
                ax.invert_yaxis()
            
            fig.tight_layout()
            
    except Exception as e:
        st.error(f"❌ Error al generar contorno para '{param}': {str(e)}")
        import traceback
        with st.expander("Ver detalles del error"):
            st.code(traceback.format_exc())
        continue
    
    # Mostrar figura
    st.pyplot(fig)
    
    # Información del polígono
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Puntos válidos", f"{len(df_param)}")
    with col2:
        st.metric("Sondeos", f"{n_ids_param}")
    with col3:
        st.metric("Área polígono", f"{poly.area:.2f}")
    with col4:
        if hasattr(poly, 'bounds'):
            bounds = poly.bounds
            st.metric("Bounds", f"[{bounds[0]:.1f}, {bounds[2]:.1f}]")
    
    # ========================================================================
    # EXPORTACIÓN
    # ========================================================================
    st.markdown("### 💾 Exportar resultados")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Exportar PNG
        png_bytes = figure_to_bytes(fig, format='png', dpi=300)
        st.download_button(
            label="📥 Descargar PNG",
            data=png_bytes,
            file_name=f"contorno_{param}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col_exp2:
        # Exportar CSV de grilla interpolada
        # Recrear la grilla para exportar
        x_min, x_max = df_param[x_col].min(), df_param[x_col].max()
        y_min_grid, y_max_grid = df_param[y_col].min(), df_param[y_col].max()
        
        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min_grid, y_max_grid, ny)
        Xi, Yi = np.meshgrid(xi, yi)
        
        from scipy.interpolate import griddata
        points = df_param[[x_col, y_col]].values
        values = df_param[param].values
        
        if prefer_method == 'cubic':
            try:
                Zi = griddata(points, values, (Xi, Yi), method='cubic')
                if np.all(np.isnan(Zi)):
                    Zi = griddata(points, values, (Xi, Yi), method='linear')
            except:
                Zi = griddata(points, values, (Xi, Yi), method='linear')
        else:
            Zi = griddata(points, values, (Xi, Yi), method='linear')
        
        csv_str = export_interpolated_grid_to_csv(Xi, Yi, Zi, include_masked=True)
        
        st.download_button(
            label="📥 Descargar CSV (grilla)",
            data=csv_str,
            file_name=f"grilla_{param}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_exp3:
        # Exportar GeoJSON del polígono (si shapely está disponible)
        geojson_dict = polygon_to_geojson(poly)
        if geojson_dict is not None:
            geojson_str = json.dumps(geojson_dict, indent=2)
            st.download_button(
                label="📥 Descargar GeoJSON",
                data=geojson_str,
                file_name=f"poligono_{param}.geojson",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("GeoJSON no disponible (requiere shapely)")
    
    # Información adicional del polígono
    with st.expander("ℹ️ Información del polígono"):
        st.markdown(f"""
        **Tipo de geometría:** {poly.geom_type if hasattr(poly, 'geom_type') else 'N/A'}
        
        **Área:** {poly.area:.4f}
        
        **Bounds:** {poly.bounds if hasattr(poly, 'bounds') else 'N/A'}
        
        **Construcción del polígono:**
        - Se agrupan los datos por `{id_col}`
        - Para cada ID se calcula: centroide X, cota mínima, cota máxima
        - Los IDs se ordenan por centroide X
        - El polígono se construye uniendo:
          - Cotas máximas de izquierda a derecha
          - Cotas mínimas de derecha a izquierda
        """)
    
    # Separador entre parámetros
    if idx < len(selected_params):
        st.markdown("---")
    
    # Cerrar figura para liberar memoria
    plt.close(fig)

# ============================================================================
# MENSAJE FINAL
# ============================================================================
st.markdown("---")
st.success("✅ ¡Interpolación completada! Puedes descargar las figuras y datos desde arriba.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    <strong>Interpolación 2D por Sondeos</strong><br>
    Función: plot_contour_between_id_minmax | Desarrollado con Streamlit<br>
    Autor: Camilo Gardi
</div>
""", unsafe_allow_html=True)
