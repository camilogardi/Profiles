"""
Aplicación Streamlit para interpolación 2D de parámetros geotécnicos.

Acepta UNA sola tabla con coordenadas (X, Y) y múltiples columnas de parámetros,
permitiendo seleccionar qué variable(s) interpolar y visualizar en mapas de contorno.

Características:
- Carga de archivo único (CSV o Excel) con X, Y y parámetros
- Mapeo interactivo de columnas
- Selección múltiple de parámetros a interpolar
- Múltiples métodos de interpolación: griddata, RBF, IDW
- Enmascaramiento para evitar extrapolación (ConvexHull y/o distancia)
- Visualización de contornos con matplotlib
- Exportación de figuras PNG y grillas CSV

Extensiones futuras:
- Kriging con pykrige
- Exportación a GeoTIFF
- Visualización interactiva con plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO

# Importar funciones de utils e io_helpers
from utils import (
    read_file,
    read_table,
    normalize_column_names,
    get_numeric_columns,
    validate_data_for_interpolation,
    calculate_parameter_statistics,
    make_xy_grid,
    interpolate_xy_grid,
    create_convexhull_mask,
    create_distance_mask,
    apply_mask_to_grid,
    combine_masks,
    export_grid_to_dataframe,
    subsample_data
)

from io_helpers import (
    validate_file_uploaded,
    show_data_preview,
    create_column_mapping_ui,
    create_parameter_selection_ui,
    show_statistics_table,
    show_validation_warnings,
    check_grid_resolution_warning,
    create_interpolation_config_ui,
    create_download_buttons,
    show_progress_info,
    show_error_message,
    show_success_message
)

# Configuración de página
st.set_page_config(
    page_title="Interpolación 2D - Parámetros Geotécnicos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("🗺️ Interpolación 2D de Parámetros Geotécnicos")
st.markdown("""
Esta aplicación genera **mapas de contorno** (interpolación 2D) de parámetros geotécnicos
a partir de **una sola tabla** con coordenadas (X, Y) y múltiples parámetros medidos.

### 📋 Requisitos del archivo de entrada:
- **Formato**: CSV o Excel (xls, xlsx)
- **Columnas mínimas**:
  - Una columna con coordenadas X (abscisa, este)
  - Una columna con coordenadas Y (cota, elevación, norte)
  - Una o más columnas con parámetros numéricos a interpolar

### 🔧 Características principales:
- ✅ Mapeo flexible de columnas
- ✅ Selección múltiple de parámetros
- ✅ Interpolación: griddata, RBF, IDW
- ✅ Enmascaramiento automático (evita extrapolación)
- ✅ Exportación PNG y CSV
""")

# Sección 1: Carga de archivo
st.header("📁 Paso 1: Cargar archivo de datos")

file_uploaded = st.file_uploader(
    "Sube tu archivo con datos geotécnicos (CSV o Excel)",
    type=['csv', 'xls', 'xlsx'],
    help="El archivo debe contener al menos columnas X, Y y uno o más parámetros numéricos"
)

if not validate_file_uploaded(file_uploaded, "archivo de datos"):
    st.stop()

# Leer archivo
try:
    with st.spinner("Leyendo archivo..."):
        df_raw = read_file(file_uploaded)
        df = normalize_column_names(df_raw)
    show_success_message(f"Archivo cargado: {len(df)} filas, {len(df.columns)} columnas")
except Exception as e:
    show_error_message("Error al leer el archivo", str(e))
    st.stop()

# Vista previa
with st.expander("🔍 Vista previa de datos", expanded=False):
    show_data_preview(df, "Primeras filas del archivo")

# Sección 2: Mapeo de columnas
st.header("🗂️ Paso 2: Mapear columnas")

st.markdown("""
Indica qué columnas contienen las coordenadas **X (abscisa)** e **Y (cota/elevación)**.
Las demás columnas numéricas serán consideradas como parámetros disponibles para interpolar.
""")

x_col, y_col = create_column_mapping_ui(df)

# Validar que X e Y sean diferentes
if x_col == y_col:
    st.error("❌ Las columnas X e Y deben ser diferentes. Por favor, selecciona columnas distintas.")
    st.stop()

# Obtener columnas de parámetros disponibles (excluyendo X, Y y posible ID)
exclude_cols = [x_col, y_col]
# Intentar detectar columna ID
id_candidates = [c for c in df.columns if any(k in c.lower() for k in ['id', 'nombre', 'name', 'sondeo'])]
id_col = id_candidates[0] if id_candidates else None
if id_col:
    exclude_cols.append(id_col)
    st.info(f"💡 Se detectó columna de identificación: **{id_col}** (se excluirá de la interpolación)")

param_cols = get_numeric_columns(df, exclude=exclude_cols)

if not param_cols:
    st.error("❌ No se encontraron columnas de parámetros numéricos para interpolar. "
             "Verifica que tu archivo contenga al menos una columna numérica además de X e Y.")
    st.stop()

st.success(f"✅ Se detectaron {len(param_cols)} parámetros disponibles: {', '.join(param_cols)}")

# Sección 3: Selección de parámetros
st.header("📊 Paso 3: Seleccionar parámetros a interpolar")

selected_params = create_parameter_selection_ui(param_cols)

if not selected_params:
    st.info("👆 Selecciona al menos un parámetro para continuar")
    st.stop()

# Sección 4: Validación y estadísticas
st.header("📈 Paso 4: Validación de datos")

# Validar datos
with st.spinner("Validando datos..."):
    df_clean, warnings = validate_data_for_interpolation(
        df, x_col, y_col, selected_params, min_points=3
    )

# Mostrar advertencias
show_validation_warnings(warnings)

# Verificar si hay suficientes datos
if 'insufficient_points' in warnings:
    st.error("❌ No hay suficientes puntos válidos para interpolar. Se requieren al menos 3 puntos con X, Y válidos.")
    st.stop()

# Calcular y mostrar estadísticas
stats_df = calculate_parameter_statistics(df_clean, selected_params)
show_statistics_table(stats_df)

# Información de puntos disponibles
n_points = len(df_clean)
st.info(f"📍 Total de puntos válidos para interpolación: **{n_points}**")

# Submuestreo si hay demasiados puntos
if n_points > 10000:
    st.warning(f"⚠️ El archivo contiene {n_points} puntos. Para mejor rendimiento, "
               f"se recomienda trabajar con menos de 10,000 puntos.")
    
    if st.checkbox("Aplicar submuestreo", value=False):
        col1, col2 = st.columns(2)
        with col1:
            subsample_method = st.radio(
                "Método de submuestreo",
                options=['random', 'grid'],
                format_func=lambda x: {
                    'random': 'Aleatorio (rápido)',
                    'grid': 'Por rejilla espacial (mantiene distribución)'
                }[x],
                help="Aleatorio: selección aleatoria simple. Grid: divide en celdas y toma puntos representativos."
            )
        with col2:
            max_points = st.slider("Número máximo de puntos", 1000, 10000, 5000, 500)
        
        df_clean = subsample_data(
            df_clean, 
            max_points=max_points, 
            method=subsample_method,
            x_col=x_col,
            y_col=y_col
        )
        st.success(f"✅ Datos submuestreados a {len(df_clean)} puntos usando método '{subsample_method}'")

# Sección 5: Configuración de interpolación
st.header("⚙️ Paso 5: Configurar interpolación y visualización")

st.markdown("Usa el **panel lateral** para configurar:")
st.markdown("- 🔲 Resolución de la grilla")
st.markdown("- 🎨 Método de interpolación (griddata, RBF, IDW)")
st.markdown("- 🔍 Enmascaramiento (ConvexHull, distancia)")
st.markdown("- 🎨 Opciones de visualización (colores, niveles)")

# Crear configuración en sidebar
config = create_interpolation_config_ui()

# Advertencia de resolución
check_grid_resolution_warning(config['nx'], config['ny'])

# Botón para generar interpolación
st.markdown("---")
generate_button = st.button("🚀 Generar mapas de contorno", type="primary", use_container_width=True)

if not generate_button:
    st.info("👆 Cuando estés listo, presiona el botón 'Generar mapas de contorno'")
    st.stop()

# Sección 6: Generación de mapas de contorno
st.header("📊 Paso 6: Mapas de contorno generados")

# Calcular límites del dominio
x_min, x_max = df_clean[x_col].min(), df_clean[x_col].max()
y_min, y_max = df_clean[y_col].min(), df_clean[y_col].max()

# Crear grilla
grid_x, grid_y = make_xy_grid(x_min, x_max, y_min, y_max, config['nx'], config['ny'])

# Preparar puntos para interpolación
points_xy = df_clean[[x_col, y_col]].values

# Crear máscara base (si se requiere)
mask_base = None
if config['mask_method'] != 'none':
    with st.spinner("Creando máscara..."):
        if config['mask_method'] == 'convexhull':
            mask_base = create_convexhull_mask(points_xy, grid_x, grid_y)
            st.info("✅ Máscara ConvexHull aplicada")
        
        elif config['mask_method'] == 'distance':
            max_dist = config.get('max_distance', None)
            mask_base = create_distance_mask(points_xy, grid_x, grid_y, max_distance=max_dist)
            st.info(f"✅ Máscara por distancia aplicada")
        
        elif config['mask_method'] == 'both':
            mask_hull = create_convexhull_mask(points_xy, grid_x, grid_y)
            max_dist = config.get('max_distance', None)
            mask_dist = create_distance_mask(points_xy, grid_x, grid_y, max_distance=max_dist)
            mask_base = combine_masks(mask_hull, mask_dist, operation='and')
            st.info("✅ Máscara combinada (ConvexHull + distancia) aplicada")

# Interpolar cada parámetro seleccionado
for idx, param in enumerate(selected_params, start=1):
    
    show_progress_info(param, idx, len(selected_params))
    
    # Filtrar datos válidos para este parámetro
    df_param = df_clean[[x_col, y_col, param]].dropna(subset=[param])
    
    if len(df_param) < 3:
        st.warning(f"⚠️ Parámetro '{param}': solo {len(df_param)} puntos válidos. Se omite.")
        continue
    
    points_param = df_param[[x_col, y_col]].values
    values_param = df_param[param].values
    
    # Interpolar
    try:
        with st.spinner(f"Interpolando '{param}'..."):
            if config['interp_method'].startswith('griddata'):
                griddata_method = config['interp_method'].split('_')[1]
                grid_values = interpolate_xy_grid(
                    points_param, values_param, grid_x, grid_y,
                    method='griddata',
                    griddata_method=griddata_method
                )
            elif config['interp_method'] == 'rbf':
                grid_values = interpolate_xy_grid(
                    points_param, values_param, grid_x, grid_y,
                    method='rbf',
                    rbf_func=config.get('rbf_func', 'multiquadric')
                )
            elif config['interp_method'] == 'idw':
                grid_values = interpolate_xy_grid(
                    points_param, values_param, grid_x, grid_y,
                    method='idw',
                    idw_power=config.get('idw_power', 2.0)
                )
    except Exception as e:
        show_error_message(f"Error al interpolar parámetro '{param}'", str(e))
        continue
    
    # Aplicar máscara si existe
    if mask_base is not None:
        grid_values = apply_mask_to_grid(grid_values, mask_base)
    
    # Verificar valores finitos
    finite_count = np.sum(np.isfinite(grid_values))
    if finite_count < 10:
        st.error(f"❌ La interpolación de '{param}' produjo muy pocos valores finitos ({finite_count})")
        continue
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Contornos rellenos
    try:
        vmin = np.nanmin(grid_values)
        vmax = np.nanmax(grid_values)
        levels = np.linspace(vmin, vmax, config['n_levels'])
        
        contourf = ax.contourf(
            grid_x, grid_y, grid_values,
            levels=levels,
            cmap=config['cmap'],
            extend='both'
        )
        
        # Líneas de contorno
        contour = ax.contour(
            grid_x, grid_y, grid_values,
            levels=levels,
            colors='black',
            alpha=0.3,
            linewidths=0.5
        )
        
        # Colorbar
        cbar = fig.colorbar(contourf, ax=ax, label=param)
        cbar.ax.tick_params(labelsize=10)
        
    except Exception as e:
        show_error_message(f"Error al crear contornos para '{param}'", str(e))
        plt.close(fig)
        continue
    
    # Overlay: puntos de datos
    if config['show_points']:
        ax.scatter(
            df_param[x_col],
            df_param[y_col],
            c='white',
            s=30,
            edgecolors='black',
            linewidths=0.7,
            alpha=0.8,
            label='Puntos de datos',
            zorder=5
        )
    
    # Etiquetas de puntos (si se requiere y existe columna ID)
    if config['show_labels'] and id_col and id_col in df_param.columns:
        for _, row in df_param.iterrows():
            ax.annotate(
                str(row.get(id_col, '')),
                (row[x_col], row[y_col]),
                fontsize=6,
                alpha=0.7,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
    
    # Configuración de ejes
    ax.set_xlabel(f'{x_col} (X - Abscisa)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{y_col} (Y - Cota/Elevación)', fontsize=12, fontweight='bold')
    
    if config['invert_yaxis']:
        ax.invert_yaxis()
    
    ax.set_title(
        f'Mapa de contorno: {param}\nMétodo: {config["interp_method"]} | Máscara: {config["mask_method"]}',
        fontsize=14,
        fontweight='bold',
        pad=15
    )
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    if config['show_points']:
        ax.legend(loc='best', fontsize=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Mostrar figura
    st.pyplot(fig)
    
    # Estadísticas de interpolación
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Puntos válidos", f"{len(df_param)}")
    with col2:
        st.metric("Valor mínimo", f"{vmin:.3f}")
    with col3:
        st.metric("Valor máximo", f"{vmax:.3f}")
    
    # Botones de descarga
    st.markdown("### 💾 Exportar resultados")
    create_download_buttons(fig, grid_x, grid_y, grid_values, param)
    
    # Separador entre parámetros
    if idx < len(selected_params):
        st.markdown("---")
    
    # Cerrar figura para liberar memoria
    plt.close(fig)

# Mensaje final
st.markdown("---")
st.success("✅ ¡Interpolación completada! Puedes descargar las figuras y datos desde arriba.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Interpolación 2D de Parámetros Geotécnicos | Desarrollado con Streamlit
</div>
""", unsafe_allow_html=True)
