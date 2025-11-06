"""
Funciones auxiliares para validación de entrada y helpers de UI.
"""

import pandas as pd
import streamlit as st
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
        st.warning(f"Por favor, sube el archivo: {file_label}")
        return False
    return True


def create_column_mapping_ui(
    df: pd.DataFrame,
    mapping_config: List[Tuple[str, str, str]],
    form_key: str = "column_mapping"
) -> Dict[str, str]:
    """
    Crea UI para mapear columnas del DataFrame a variables requeridas.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas a mapear.
    mapping_config : List[Tuple[str, str, str]]
        Lista de tuplas: (variable_name, label_description, default_col_index).
    form_key : str
        Clave única para el formulario.
        
    Returns
    -------
    Dict[str, str]
        Diccionario con mapeo variable_name -> nombre_columna_seleccionada.
    """
    cols = df.columns.tolist()
    mapping = {}
    
    for var_name, label, default_idx in mapping_config:
        if isinstance(default_idx, int):
            idx = min(default_idx, len(cols) - 1) if len(cols) > 0 else 0
        else:
            # Buscar por nombre
            idx = cols.index(default_idx) if default_idx in cols else 0
        
        selected = st.selectbox(
            label,
            options=cols,
            index=idx,
            key=f"{form_key}_{var_name}"
        )
        mapping[var_name] = selected
    
    return mapping


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
    st.dataframe(df.head(n_rows))
    st.caption(f"Total de filas: {len(df)}, Total de columnas: {len(df.columns)}")


def show_validation_message(is_valid: bool, message: str):
    """
    Muestra mensaje de validación.
    
    Parameters
    ----------
    is_valid : bool
        Si la validación fue exitosa.
    message : str
        Mensaje a mostrar.
    """
    if is_valid:
        st.success(message)
    else:
        st.error(message)


def show_borehole_summary(
    borehole_bounds: pd.DataFrame,
    id_col: str,
    show_n: int = 20
):
    """
    Muestra resumen de límites verticales de sondeos.
    
    Parameters
    ----------
    borehole_bounds : pd.DataFrame
        DataFrame con información de límites de sondeos.
    id_col : str
        Nombre de columna con ID.
    show_n : int
        Número máximo de filas a mostrar.
    """
    st.subheader("Resumen de sondeos")
    
    # Mostrar estadísticas generales
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de sondeos", len(borehole_bounds))
    with col2:
        st.metric("Z máximo (cota más alta)", f"{borehole_bounds['z_top'].max():.2f}")
    with col3:
        st.metric("Z mínimo (fondo más bajo)", f"{borehole_bounds['z_bottom'].min():.2f}")
    
    # Tabla detallada
    st.dataframe(
        borehole_bounds.head(show_n),
        use_container_width=True
    )
    
    if len(borehole_bounds) > show_n:
        st.caption(f"Mostrando {show_n} de {len(borehole_bounds)} sondeos")


def show_missing_ids_warning(missing_ids: List[str]):
    """
    Muestra advertencia sobre IDs sin coincidencia.
    
    Parameters
    ----------
    missing_ids : List[str]
        Lista de IDs que no tienen coincidencia.
    """
    if missing_ids:
        st.warning(
            f"⚠️ Se encontraron {len(missing_ids)} IDs en el archivo de ensayos "
            f"que no existen en el archivo de cabeceras. Estos registros serán ignorados."
        )
        with st.expander("Ver IDs sin coincidencia"):
            st.write(missing_ids)


def validate_numeric_data(
    df: pd.DataFrame,
    columns: List[str]
) -> Tuple[pd.DataFrame, int]:
    """
    Valida que las columnas especificadas sean numéricas y elimina filas con valores no válidos.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a validar.
    columns : List[str]
        Lista de columnas que deben ser numéricas.
        
    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame con solo filas válidas.
    n_removed : int
        Número de filas eliminadas.
    """
    df_clean = df.copy()
    original_len = len(df_clean)
    
    for col in columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    df_clean = df_clean.dropna(subset=columns)
    n_removed = original_len - len(df_clean)
    
    return df_clean, n_removed


def check_grid_resolution_warning(nx: int, nz: int, threshold: int = 500000):
    """
    Verifica si la resolución de grilla es muy alta y muestra advertencia.
    
    Parameters
    ----------
    nx, nz : int
        Resolución de grilla.
    threshold : int
        Umbral para mostrar advertencia.
    """
    total_points = nx * nz
    if total_points > threshold:
        st.warning(
            f"⚠️ La resolución de grilla es alta ({nx} x {nz} = {total_points:,} puntos). "
            f"Esto puede consumir mucha memoria y tiempo de cómputo. "
            f"Considera reducir la resolución para previsualizaciones rápidas."
        )


def get_interpolation_method_description(method: str) -> str:
    """
    Retorna descripción del método de interpolación.
    
    Parameters
    ----------
    method : str
        Nombre del método.
        
    Returns
    -------
    str
        Descripción del método.
    """
    descriptions = {
        'griddata_linear': 'Interpolación lineal (rápida, suave)',
        'griddata_nearest': 'Vecino más cercano (preserva valores discretos)',
        'griddata_cubic': 'Interpolación cúbica (muy suave, puede sobrepasar)',
        'rbf': 'Función de Base Radial (suave, buena para datos dispersos)',
        'idw': 'Ponderación por Distancia Inversa (promedio ponderado, sin sobrepaso)'
    }
    return descriptions.get(method, method)


def create_sidebar_configuration(
    param_columns: List[str],
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float
) -> Dict:
    """
    Crea UI de configuración en sidebar y retorna parámetros seleccionados.
    
    Parameters
    ----------
    param_columns : List[str]
        Lista de columnas de parámetros disponibles.
    x_min, x_max : float
        Rango de posiciones X.
    z_min, z_max : float
        Rango de elevaciones Z.
        
    Returns
    -------
    Dict
        Diccionario con configuración seleccionada.
    """
    st.sidebar.subheader("⚙️ Configuración de perfil")
    
    config = {}
    
    # Selección de parámetro
    config['param_col'] = st.sidebar.selectbox(
        "Parámetro a graficar",
        options=param_columns,
        help="Selecciona la columna del parámetro geotécnico a visualizar"
    )
    
    # Método de ordenación
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Ordenación de sondeos (eje X)")
    config['order_method'] = st.sidebar.radio(
        "Método de ordenación",
        options=['x', 'xy_sort', 'pca'],
        format_func=lambda x: {
            'x': 'Coordenada X real',
            'xy_sort': 'Ordenar por X, luego Y',
            'pca': 'Proyección PCA (eje principal)'
        }[x],
        help="PCA es útil para transectos oblicuos"
    )
    
    # Configuración de grilla
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔲 Resolución de grilla")
    config['nx'] = st.sidebar.number_input(
        "Puntos en X",
        min_value=20,
        max_value=1000,
        value=min(200, 200),
        step=10,
        help="Resolución horizontal"
    )
    config['nz'] = st.sidebar.number_input(
        "Puntos en Z",
        min_value=20,
        max_value=1000,
        value=min(200, 200),
        step=10,
        help="Resolución vertical"
    )
    
    # Método de interpolación
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Interpolación")
    config['interp_method'] = st.sidebar.selectbox(
        "Método",
        options=['griddata_linear', 'griddata_nearest', 'griddata_cubic', 'rbf', 'idw'],
        format_func=get_interpolation_method_description
    )
    
    # Parámetros específicos según método
    if config['interp_method'] == 'rbf':
        config['rbf_func'] = st.sidebar.selectbox(
            "Función RBF",
            options=['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic']
        )
    
    if config['interp_method'] == 'idw':
        config['idw_power'] = st.sidebar.slider(
            "Potencia IDW",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5
        )
    
    # Opciones de visualización
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Visualización")
    config['n_levels'] = st.sidebar.number_input(
        "Niveles de contorno",
        min_value=5,
        max_value=50,
        value=15,
        step=1
    )
    
    # Colormap
    from matplotlib import pyplot as plt
    colormaps = sorted([m for m in plt.colormaps()])
    default_cmap = 'viridis' if 'viridis' in colormaps else colormaps[0]
    config['cmap'] = st.sidebar.selectbox(
        "Mapa de colores",
        options=colormaps,
        index=colormaps.index(default_cmap)
    )
    
    # Opciones de máscara
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔍 Enmascaramiento")
    config['apply_mask'] = st.sidebar.checkbox(
        "Aplicar máscara vertical",
        value=True,
        help="Enmascara zonas sin cobertura vertical real"
    )
    
    if config['apply_mask']:
        config['max_h_distance'] = st.sidebar.number_input(
            "Distancia horizontal máxima",
            min_value=0.0,
            value=0.0,
            help="0 = automático (1.5x distancia al sondeo más cercano)"
        )
        if config['max_h_distance'] == 0.0:
            config['max_h_distance'] = None
    
    # Opciones de anotación
    st.sidebar.markdown("---")
    st.sidebar.subheader("📝 Anotaciones")
    config['show_borehole_labels'] = st.sidebar.checkbox(
        "Mostrar etiquetas de sondeos",
        value=True
    )
    config['show_sample_points'] = st.sidebar.checkbox(
        "Mostrar puntos de ensayo",
        value=True
    )
    
    return config
