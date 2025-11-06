"""
Aplicación Streamlit para generación de perfiles geotécnicos.
Genera perfiles verticales (X vs Cota vs Parámetro) a partir de dos archivos:
- Archivo A: Cabeceras de sondeos (ID, x, y, cota)
- Archivo B: Ensayos por profundidad (ID, profundidad, parámetros)
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
    normalize_column_names,
    merge_headers_and_samples,
    calculate_z_param,
    compute_borehole_bounds,
    order_boreholes_by_x,
    make_xz_grid,
    interpolate_xz_grid,
    create_vertical_mask,
    apply_mask_to_grid,
    get_numeric_columns,
    validate_merged_data,
    export_grid_to_dataframe
)

from io_helpers import (
    validate_file_uploaded,
    create_column_mapping_ui,
    show_data_preview,
    show_validation_message,
    show_borehole_summary,
    show_missing_ids_warning,
    validate_numeric_data,
    check_grid_resolution_warning,
    create_sidebar_configuration
)


# Configuración de página
st.set_page_config(
    page_title="Perfiles Geotécnicos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("🗻 Generador de Perfiles Geotécnicos")
st.markdown("""
Esta aplicación genera **perfiles verticales** (sección X vs Elevación) de parámetros geotécnicos
a partir de datos de sondeos. Requiere **DOS archivos**:

1. **Archivo de cabeceras**: Información de cada sondeo (ID, x, y, cota_inicial)
2. **Archivo de ensayos**: Resultados de ensayos por profundidad (ID, profundidad, parámetros)
""")

# Sección 1: Carga de archivos
st.header("📁 Paso 1: Cargar archivos")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Archivo A: Cabeceras de sondeos")
    st.caption("Debe contener: ID_sondeo, x, y, cota")
    file_headers = st.file_uploader(
        "Subir archivo de cabeceras (CSV/Excel)",
        type=['csv', 'xls', 'xlsx'],
        key='file_headers'
    )

with col2:
    st.subheader("Archivo B: Ensayos por profundidad")
    st.caption("Debe contener: ID_sondeo, profundidad_ensayo, parámetros")
    file_samples = st.file_uploader(
        "Subir archivo de ensayos (CSV/Excel)",
        type=['csv', 'xls', 'xlsx'],
        key='file_samples'
    )

# Validar archivos subidos
if not validate_file_uploaded(file_headers, "Archivo de cabeceras"):
    st.stop()

if not validate_file_uploaded(file_samples, "Archivo de ensayos"):
    st.stop()

# Leer archivos
try:
    df_headers = read_file(file_headers)
    df_headers = normalize_column_names(df_headers)
    st.success(f"✅ Archivo de cabeceras cargado: {len(df_headers)} filas")
except Exception as e:
    st.error(f"Error al leer archivo de cabeceras: {e}")
    st.stop()

try:
    df_samples = read_file(file_samples)
    df_samples = normalize_column_names(df_samples)
    st.success(f"✅ Archivo de ensayos cargado: {len(df_samples)} filas")
except Exception as e:
    st.error(f"Error al leer archivo de ensayos: {e}")
    st.stop()

# Mostrar vista previa
with st.expander("🔍 Vista previa - Archivo de cabeceras"):
    show_data_preview(df_headers, "Cabeceras de sondeos")

with st.expander("🔍 Vista previa - Archivo de ensayos"):
    show_data_preview(df_samples, "Ensayos por profundidad")

# Sección 2: Mapeo de columnas
st.header("🗂️ Paso 2: Mapear columnas")

with st.form("column_mapping_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mapeo: Archivo de cabeceras")
        headers_cols = df_headers.columns.tolist()
        
        id_col_headers = st.selectbox(
            "Columna: ID del sondeo",
            options=headers_cols,
            index=0,
            key='id_headers'
        )
        x_col = st.selectbox(
            "Columna: Coordenada X",
            options=headers_cols,
            index=min(1, len(headers_cols)-1),
            key='x_col'
        )
        y_col = st.selectbox(
            "Columna: Coordenada Y",
            options=headers_cols,
            index=min(2, len(headers_cols)-1),
            key='y_col'
        )
        cota_col = st.selectbox(
            "Columna: Cota (elevación inicial)",
            options=headers_cols,
            index=min(3, len(headers_cols)-1),
            key='cota_col'
        )
    
    with col2:
        st.subheader("Mapeo: Archivo de ensayos")
        samples_cols = df_samples.columns.tolist()
        
        id_col_samples = st.selectbox(
            "Columna: ID del sondeo",
            options=samples_cols,
            index=0,
            key='id_samples'
        )
        profundidad_col = st.selectbox(
            "Columna: Profundidad del ensayo",
            options=samples_cols,
            index=min(1, len(samples_cols)-1),
            key='profundidad_col'
        )
    
    submitted = st.form_submit_button("✅ Aplicar mapeo y procesar datos")

if not submitted:
    st.info("👆 Configura el mapeo de columnas y presiona el botón para continuar")
    st.stop()

# Sección 3: Procesamiento de datos
st.header("⚙️ Paso 3: Procesamiento de datos")

# Validar columnas numéricas requeridas
df_headers_clean, n_removed_h = validate_numeric_data(
    df_headers,
    [x_col, y_col, cota_col]
)

df_samples_clean, n_removed_s = validate_numeric_data(
    df_samples,
    [profundidad_col]
)

if n_removed_h > 0:
    st.warning(f"⚠️ Se eliminaron {n_removed_h} filas del archivo de cabeceras por valores no numéricos")

if n_removed_s > 0:
    st.warning(f"⚠️ Se eliminaron {n_removed_s} filas del archivo de ensayos por valores no numéricos")

# Merge de datos
with st.spinner("Uniendo archivos..."):
    df_merged, missing_ids = merge_headers_and_samples(
        df_headers_clean,
        df_samples_clean,
        id_col_headers,
        id_col_samples
    )

# Mostrar advertencias de IDs faltantes
show_missing_ids_warning(missing_ids)

# Validar merge
is_valid, msg = validate_merged_data(
    df_merged,
    [id_col_headers, x_col, y_col, cota_col, profundidad_col]
)
show_validation_message(is_valid, msg)

if not is_valid:
    st.stop()

# Calcular z_param (elevación de cada ensayo)
df_merged = calculate_z_param(df_merged, cota_col, profundidad_col, 'z_param')

# Obtener columnas de parámetros disponibles
exclude_cols = [id_col_headers, id_col_samples, x_col, y_col, cota_col, profundidad_col, 'z_param']
param_columns = get_numeric_columns(df_merged, exclude=exclude_cols)

if not param_columns:
    st.error("❌ No se encontraron columnas de parámetros numéricos en el archivo de ensayos")
    st.stop()

st.success(f"✅ Datos procesados: {len(df_merged)} ensayos de {df_merged[id_col_headers].nunique()} sondeos")

# Calcular límites verticales de sondeos
borehole_bounds = compute_borehole_bounds(
    df_merged,
    id_col_headers,
    cota_col,
    profundidad_col
)

# Mostrar resumen de sondeos
show_borehole_summary(borehole_bounds, id_col_headers)

# Sección 4: Configuración de visualización
st.header("🎨 Paso 4: Configurar visualización")

# Obtener rangos
z_min = borehole_bounds['z_bottom'].min()
z_max = borehole_bounds['z_top'].max()
x_min_coord = df_headers_clean[x_col].min()
x_max_coord = df_headers_clean[x_col].max()

# Crear configuración en sidebar
config = create_sidebar_configuration(
    param_columns,
    x_min_coord,
    x_max_coord,
    z_min,
    z_max
)

# Advertencia de resolución
check_grid_resolution_warning(config['nx'], config['nz'])

# Botón para generar perfil
generate_button = st.button("🚀 Generar perfil", type="primary", use_container_width=True)

if not generate_button:
    st.info("👆 Configura los parámetros en el panel lateral y presiona 'Generar perfil'")
    st.stop()

# Sección 5: Generación de perfil
st.header("📊 Paso 5: Perfil generado")

with st.spinner("Generando perfil..."):
    
    # Ordenar sondeos para eje X
    try:
        borehole_positions, sorted_positions = order_boreholes_by_x(
            df_headers_clean,
            id_col_headers,
            x_col,
            y_col,
            method=config['order_method']
        )
    except Exception as e:
        st.error(f"Error al ordenar sondeos: {e}")
        st.stop()
    
    # Mapear posiciones X a cada ensayo
    df_merged['x_pos'] = df_merged[id_col_headers].map(borehole_positions)
    df_merged = df_merged.dropna(subset=['x_pos'])
    
    # Filtrar datos del parámetro seleccionado
    param_col = config['param_col']
    df_plot = df_merged[[id_col_headers, 'x_pos', 'z_param', param_col]].copy()
    df_plot = df_plot.dropna(subset=[param_col])
    
    if len(df_plot) < 3:
        st.error(f"❌ Insuficientes datos para el parámetro '{param_col}' (se requieren al menos 3 puntos)")
        st.stop()
    
    st.info(f"📍 Usando {len(df_plot)} puntos de ensayo para interpolación")
    
    # Crear grilla X-Z
    x_min = sorted_positions.min()
    x_max = sorted_positions.max()
    
    grid_x, grid_z = make_xz_grid(
        x_min, x_max,
        z_min, z_max,
        config['nx'], config['nz']
    )
    
    # Preparar datos para interpolación
    points_xz = df_plot[['x_pos', 'z_param']].values
    values = df_plot[param_col].values
    
    # Interpolar
    try:
        if config['interp_method'].startswith('griddata'):
            griddata_method = config['interp_method'].split('_')[1]
            grid_values = interpolate_xz_grid(
                points_xz, values, grid_x, grid_z,
                method='griddata',
                griddata_method=griddata_method
            )
        elif config['interp_method'] == 'rbf':
            grid_values = interpolate_xz_grid(
                points_xz, values, grid_x, grid_z,
                method='rbf',
                rbf_func=config.get('rbf_func', 'multiquadric')
            )
        elif config['interp_method'] == 'idw':
            grid_values = interpolate_xz_grid(
                points_xz, values, grid_x, grid_z,
                method='idw',
                idw_power=config.get('idw_power', 2.0)
            )
        else:
            st.error(f"Método desconocido: {config['interp_method']}")
            st.stop()
    except Exception as e:
        st.error(f"Error durante interpolación: {e}")
        st.stop()
    
    # Aplicar máscara si está habilitada
    if config['apply_mask']:
        try:
            mask = create_vertical_mask(
                grid_x, grid_z,
                borehole_bounds,
                borehole_positions,
                id_col_headers,
                max_horizontal_distance=config.get('max_h_distance')
            )
            grid_values = apply_mask_to_grid(grid_values, mask)
            st.info("✅ Máscara vertical aplicada")
        except Exception as e:
            st.warning(f"No se pudo aplicar máscara: {e}")
    
    # Verificar valores finitos
    finite_count = np.sum(np.isfinite(grid_values))
    if finite_count < 10:
        st.error(f"❌ La interpolación produjo muy pocos valores finitos ({finite_count})")
        st.stop()
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Contornos
    try:
        levels = np.linspace(
            np.nanmin(grid_values),
            np.nanmax(grid_values),
            config['n_levels']
        )
        contourf = ax.contourf(
            grid_x, grid_z, grid_values,
            levels=levels,
            cmap=config['cmap'],
            extend='both'
        )
        
        # Líneas de contorno
        contour_lines = ax.contour(
            grid_x, grid_z, grid_values,
            levels=levels,
            colors='black',
            alpha=0.3,
            linewidths=0.5
        )
        
        # Colorbar
        cbar = fig.colorbar(contourf, ax=ax, label=param_col)
        
    except Exception as e:
        st.error(f"Error al crear contornos: {e}")
        st.stop()
    
    # Overlay: puntos de ensayo
    if config['show_sample_points']:
        ax.scatter(
            df_plot['x_pos'],
            df_plot['z_param'],
            c='white',
            s=20,
            edgecolors='black',
            linewidths=0.5,
            alpha=0.7,
            label='Puntos de ensayo',
            zorder=5
        )
    
    # Overlay: líneas verticales y etiquetas de sondeos
    if config['show_borehole_labels']:
        for _, row in borehole_bounds.iterrows():
            borehole_id = row[id_col_headers]
            if borehole_id not in borehole_positions:
                continue
            
            x_pos = borehole_positions[borehole_id]
            z_top = row['z_top']
            z_bottom = row['z_bottom']
            
            # Línea vertical del sondeo
            ax.plot(
                [x_pos, x_pos],
                [z_bottom, z_top],
                'k-',
                linewidth=1.5,
                alpha=0.4,
                zorder=3
            )
            
            # Etiqueta
            ax.text(
                x_pos,
                z_top + (z_max - z_min) * 0.02,
                str(borehole_id),
                rotation=90,
                fontsize=8,
                ha='center',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
    
    # Configuración de ejes
    ax.set_xlabel('Posición X (ordenada)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Elevación Z (cota)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Perfil de {param_col} - Método: {config["interp_method"]}',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if config['show_sample_points']:
        ax.legend(loc='best')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Mostrar figura
    st.pyplot(fig)
    
    # Información adicional
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Puntos interpolados", f"{finite_count:,}")
    with col2:
        st.metric("Valor mínimo", f"{np.nanmin(grid_values):.3f}")
    with col3:
        st.metric("Valor máximo", f"{np.nanmax(grid_values):.3f}")

# Sección 6: Exportar resultados
st.header("💾 Paso 6: Exportar resultados")

col1, col2 = st.columns(2)

with col1:
    # Exportar figura PNG
    st.subheader("Descargar figura")
    buf_img = BytesIO()
    fig.savefig(buf_img, format='png', dpi=300, bbox_inches='tight')
    buf_img.seek(0)
    
    st.download_button(
        label="📥 Descargar figura (PNG)",
        data=buf_img,
        file_name=f"perfil_{param_col}.png",
        mime="image/png",
        use_container_width=True
    )

with col2:
    # Exportar grilla CSV
    st.subheader("Descargar datos interpolados")
    df_export = export_grid_to_dataframe(grid_x, grid_z, grid_values)
    
    buf_csv = BytesIO()
    df_export.to_csv(buf_csv, index=False)
    buf_csv.seek(0)
    
    st.download_button(
        label="📥 Descargar grilla (CSV)",
        data=buf_csv,
        file_name=f"grilla_{param_col}.csv",
        mime="text/csv",
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    Generador de Perfiles Geotécnicos | Desarrollado con Streamlit
</div>
""", unsafe_allow_html=True)
