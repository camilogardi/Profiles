"""
Funciones auxiliares para validación de entrada y helpers de UI para interpolación 2D.
Provee interfaces de usuario para:
- Validación de archivos subidos
- Mapeo de columnas X, Y
- Selección múltiple de parámetros a interpolar
- Configuración de interpolación y visualización
- Estadísticas y validaciones
"""

import pandas as pd
import streamlit as st
import numpy as np
from typing import List, Tuple, Dict, Optional


def validate_file_uploaded(file_obj, file_label: str) -> bool:
    """
    Valida que un archivo haya sido subido.
    
    Parameters
    ----------
    file_obj : file-like object or None
        Objeto de archivo subido.
    file_label : str
        Etiqueta descriptiva del archivo.
        
    Returns
    -------
    bool
        True si el archivo existe, False en caso contrario.
    """
    if file_obj is None:
        st.info(f"👆 Por favor, sube el archivo: {file_label}")
        return False
    return True


def show_data_preview(df: pd.DataFrame, title: str = "Vista previa de datos", n_rows: int = 10):
    """
    Muestra vista previa de un DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a mostrar.
    title : str
        Título de la sección.
    n_rows : int
        Número de filas a mostrar.
    """
    st.subheader(title)
    st.dataframe(df.head(n_rows), use_container_width=True)
    st.caption(f"Total de filas: {len(df)}, Total de columnas: {len(df.columns)}")


def create_column_mapping_ui(
    df: pd.DataFrame,
    default_x_col: Optional[str] = None,
    default_y_col: Optional[str] = None
) -> Tuple[str, str]:
    """
    Crea UI para mapear columnas X e Y del DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas a mapear.
    default_x_col : Optional[str]
        Nombre de columna por defecto para X.
    default_y_col : Optional[str]
        Nombre de columna por defecto para Y.
        
    Returns
    -------
    x_col, y_col : Tuple[str, str]
        Nombres de columnas seleccionadas para X e Y.
    """
    cols = df.columns.tolist()
    
    # Intentar detectar automáticamente columnas X e Y
    x_candidates = [c for c in cols if any(k in c.lower() for k in ['x', 'abscisa', 'este', 'easting'])]
    y_candidates = [c for c in cols if any(k in c.lower() for k in ['y', 'cota', 'elevacion', 'norte', 'northing', 'elevation'])]
    
    # Índices por defecto
    x_idx = 0
    y_idx = min(1, len(cols) - 1)
    
    if default_x_col and default_x_col in cols:
        x_idx = cols.index(default_x_col)
    elif x_candidates:
        x_idx = cols.index(x_candidates[0])
    
    if default_y_col and default_y_col in cols:
        y_idx = cols.index(default_y_col)
    elif y_candidates:
        y_idx = cols.index(y_candidates[0])
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_col = st.selectbox(
            "🔹 Columna: X (Abscisa)",
            options=cols,
            index=x_idx,
            help="Selecciona la columna que contiene las coordenadas X (abscisa, este)"
        )
    
    with col2:
        y_col = st.selectbox(
            "🔹 Columna: Y (Cota / Elevación)",
            options=cols,
            index=y_idx,
            help="Selecciona la columna que contiene las coordenadas Y (cota, elevación)"
        )
    
    return x_col, y_col


def create_parameter_selection_ui(
    available_params: List[str]
) -> List[str]:
    """
    Crea UI para seleccionar parámetros a interpolar.
    
    Parameters
    ----------
    available_params : List[str]
        Lista de parámetros disponibles.
        
    Returns
    -------
    List[str]
        Lista de parámetros seleccionados.
    """
    st.subheader("📊 Selección de parámetros")
    
    if not available_params:
        st.error("❌ No se encontraron parámetros numéricos disponibles")
        return []
    
    selected = st.multiselect(
        "Selecciona uno o más parámetros a interpolar:",
        options=available_params,
        default=[available_params[0]] if available_params else [],
        help="Puedes seleccionar múltiples parámetros. Se generará un gráfico por cada uno."
    )
    
    return selected


def show_statistics_table(stats_df: pd.DataFrame):
    """
    Muestra tabla de estadísticas de parámetros.
    
    Parameters
    ----------
    stats_df : pd.DataFrame
        DataFrame con estadísticas.
    """
    st.subheader("📈 Estadísticas de parámetros seleccionados")
    
    # Formatear números
    formatted_df = stats_df.copy()
    for col in ['Mínimo', 'Máximo', 'Media', 'Desv.Est.']:
        if col in formatted_df.columns:
            formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    
    st.dataframe(formatted_df, use_container_width=True, hide_index=True)


def show_validation_warnings(warnings: Dict[str, str]):
    """
    Muestra advertencias de validación.
    
    Parameters
    ----------
    warnings : Dict[str, str]
        Diccionario con advertencias.
    """
    if warnings:
        for key, msg in warnings.items():
            st.warning(f"⚠️ {msg}")


def check_grid_resolution_warning(nx: int, ny: int, threshold: int = 1000000):
    """
    Verifica si la resolución de grilla es muy alta y muestra advertencia.
    
    Parameters
    ----------
    nx, ny : int
        Resolución de grilla.
    threshold : int
        Umbral para mostrar advertencia.
    """
    total_points = nx * ny
    if total_points > threshold:
        st.warning(
            f"⚠️ La resolución de grilla es alta ({nx} x {ny} = {total_points:,} puntos). "
            f"Esto puede consumir mucha memoria y tiempo de cómputo. "
            f"Considera reducir la resolución para previsualizaciones rápidas."
        )


def create_interpolation_config_ui() -> Dict:
    """
    Crea UI de configuración de interpolación y visualización en sidebar.
    
    Returns
    -------
    Dict
        Diccionario con configuración seleccionada.
    """
    st.sidebar.header("⚙️ Configuración")
    
    config = {}
    
    # Resolución de grilla
    st.sidebar.subheader("🔲 Resolución de grilla")
    config['nx'] = st.sidebar.number_input(
        "Puntos en X",
        min_value=20,
        max_value=500,
        value=100,
        step=10,
        help="Resolución horizontal de la grilla"
    )
    config['ny'] = st.sidebar.number_input(
        "Puntos en Y",
        min_value=20,
        max_value=500,
        value=100,
        step=10,
        help="Resolución vertical de la grilla"
    )
    
    # Método de interpolación
    st.sidebar.subheader("🎨 Método de interpolación")
    
    interp_options = {
        'griddata_linear': 'Griddata - Linear (rápida, suave)',
        'griddata_nearest': 'Griddata - Nearest (preserva valores)',
        'griddata_cubic': 'Griddata - Cubic (muy suave)',
        'rbf': 'RBF - Radial Basis Function',
        'idw': 'IDW - Inverse Distance Weighting'
    }
    
    config['interp_method'] = st.sidebar.selectbox(
        "Método",
        options=list(interp_options.keys()),
        format_func=lambda x: interp_options[x],
        index=0
    )
    
    # Parámetros específicos según método
    if config['interp_method'] == 'rbf':
        config['rbf_func'] = st.sidebar.selectbox(
            "Función RBF",
            options=['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'],
            help="Función de base radial para interpolación"
        )
    
    if config['interp_method'] == 'idw':
        config['idw_power'] = st.sidebar.slider(
            "Potencia IDW",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5,
            help="Mayor potencia = más peso a puntos cercanos"
        )
    
    # Enmascaramiento
    st.sidebar.subheader("🔍 Enmascaramiento")
    st.sidebar.caption("Evita extrapolación fuera del dominio de datos")
    
    config['mask_method'] = st.sidebar.radio(
        "Método de máscara",
        options=['none', 'convexhull', 'distance', 'both'],
        format_func=lambda x: {
            'none': 'Sin máscara',
            'convexhull': 'ConvexHull (envolvente)',
            'distance': 'Por distancia',
            'both': 'Ambos (combinados)'
        }[x],
        help="ConvexHull: enmascara fuera del polígono convexo.\nDistancia: enmascara celdas lejanas a puntos de datos."
    )
    
    if config['mask_method'] in ['distance', 'both']:
        config['max_distance'] = st.sidebar.number_input(
            "Distancia máxima",
            min_value=0.0,
            value=0.0,
            help="0 = automático (basado en distribución de puntos)"
        )
        if config['max_distance'] == 0.0:
            config['max_distance'] = None
    
    # Visualización
    st.sidebar.subheader("🎨 Visualización")
    
    config['n_levels'] = st.sidebar.slider(
        "Niveles de contorno",
        min_value=5,
        max_value=50,
        value=15,
        step=1
    )
    
    # Colormap con opciones comunes
    common_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 
                    'coolwarm', 'RdYlBu_r', 'RdBu_r', 'seismic', 'jet']
    config['cmap'] = st.sidebar.selectbox(
        "Mapa de colores",
        options=common_cmaps,
        index=0
    )
    
    config['show_points'] = st.sidebar.checkbox(
        "Mostrar puntos de datos",
        value=True,
        help="Overlay de puntos de muestreo originales"
    )
    
    config['invert_yaxis'] = st.sidebar.checkbox(
        "Invertir eje Y",
        value=False,
        help="Útil para mostrar profundidad creciente hacia abajo"
    )
    
    config['show_labels'] = st.sidebar.checkbox(
        "Mostrar etiquetas de puntos",
        value=False,
        help="Requiere columna 'id' en los datos"
    )
    
    return config


def create_download_buttons(
    fig,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_values: np.ndarray,
    param_name: str
):
    """
    Crea botones de descarga para figura y datos.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura a exportar.
    grid_x, grid_y : np.ndarray
        Grillas de coordenadas.
    grid_values : np.ndarray
        Valores interpolados.
    param_name : str
        Nombre del parámetro.
    """
    from io import BytesIO
    import pandas as pd
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Descargar figura")
        buf_img = BytesIO()
        fig.savefig(buf_img, format='png', dpi=300, bbox_inches='tight')
        buf_img.seek(0)
        
        st.download_button(
            label="💾 Descargar PNG",
            data=buf_img,
            file_name=f"contorno_{param_name}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col2:
        st.subheader("📥 Descargar grilla CSV")
        
        # Crear DataFrame con grilla
        df_export = pd.DataFrame({
            'X': grid_x.ravel(),
            'Y': grid_y.ravel(),
            param_name: grid_values.ravel()
        })
        
        csv_buffer = BytesIO()
        df_export.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            label="💾 Descargar CSV",
            data=csv_buffer,
            file_name=f"grilla_{param_name}.csv",
            mime="text/csv",
            use_container_width=True
        )


def show_progress_info(param_name: str, current: int, total: int):
    """
    Muestra información de progreso para múltiples parámetros.
    
    Parameters
    ----------
    param_name : str
        Nombre del parámetro actual.
    current : int
        Número de parámetro actual.
    total : int
        Total de parámetros.
    """
    st.info(f"🔄 Procesando parámetro {current}/{total}: **{param_name}**")


def show_error_message(message: str, details: Optional[str] = None):
    """
    Muestra mensaje de error con detalles opcionales.
    
    Parameters
    ----------
    message : str
        Mensaje de error principal.
    details : Optional[str]
        Detalles adicionales del error.
    """
    st.error(f"❌ {message}")
    if details:
        with st.expander("Ver detalles del error"):
            st.code(details)


def show_success_message(message: str):
    """
    Muestra mensaje de éxito.
    
    Parameters
    ----------
    message : str
        Mensaje de éxito.
    """
    st.success(f"✅ {message}")
