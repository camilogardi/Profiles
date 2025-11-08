"""
Funciones utilitarias para interpolación 2D de parámetros geotécnicos.
Este módulo provee funcionalidades para:
- Lectura y normalización de archivos con una sola tabla
- Interpolación 2D en plano X-Y (abscisa vs cota)
- Múltiples métodos: griddata, RBF, IDW
- Enmascaramiento para evitar extrapolación (ConvexHull y distancia)
- Exportación de grillas interpoladas

Extensiones posibles:
- Añadir kriging con pykrige para datos geoestadísticos
- Exportar a GeoTIFF con coordenadas georreferenciadas
- Implementar submuestreo inteligente para datasets grandes
"""

import pandas as pd
import numpy as np
from scipy.interpolate import griddata, Rbf
from scipy.spatial import ConvexHull, cKDTree, distance
from typing import Tuple, List, Optional, Dict, Union


def read_table(filelike) -> pd.DataFrame:
    """
    Lee un archivo CSV o Excel y retorna un DataFrame con manejo robusto de errores.
    
    Detecta la extensión del archivo y utiliza el motor apropiado:
    - .csv: pandas.read_csv
    - .xlsx: pandas.read_excel con engine='openpyxl'
    - .xls: pandas.read_excel con engine='xlrd'
    
    Si no se puede detectar la extensión o falla la lectura, intenta en orden:
    1. Lectura como CSV
    2. Lectura como XLSX (openpyxl)
    3. Lectura como XLS (xlrd)
    
    Resetea el puntero del archivo (seek(0)) antes de cada intento de lectura
    para asegurar compatibilidad con objetos file-like de Streamlit.
    
    Parameters
    ----------
    filelike : file-like object
        Objeto de archivo con atributo .name y método .seek().
        Compatible con streamlit.UploadedFile.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con los datos del archivo.
        
    Raises
    ------
    ValueError
        Si el tipo de archivo no es soportado o no se puede leer.
    ImportError
        Si falta una dependencia necesaria para leer el archivo (openpyxl o xlrd).
        El mensaje incluye instrucciones claras de instalación y alternativas.
        
    Examples
    --------
    >>> from io import BytesIO
    >>> import pandas as pd
    >>> # Crear CSV en memoria
    >>> csv_data = "col1,col2\\n1,2\\n3,4"
    >>> filelike = BytesIO(csv_data.encode())
    >>> filelike.name = "test.csv"
    >>> df = read_table(filelike)
    >>> df.shape
    (2, 2)
    
    Notes
    -----
    Para archivos .xlsx se requiere: pip install openpyxl>=3.0.0
    Para archivos .xls se requiere: pip install xlrd>=2.0.1
    Como alternativa, puedes exportar el archivo a CSV.
    """
    # Detectar extensión si existe atributo .name
    extension = None
    if hasattr(filelike, 'name') and filelike.name:
        file_name = filelike.name.lower()
        if file_name.endswith('.csv'):
            extension = '.csv'
        elif file_name.endswith('.xlsx'):
            extension = '.xlsx'
        elif file_name.endswith('.xls'):
            extension = '.xls'
    
    # Intentar lectura basada en extensión detectada
    if extension == '.csv':
        if hasattr(filelike, 'seek'):
            filelike.seek(0)
        return pd.read_csv(filelike)
    
    elif extension == '.xlsx':
        if hasattr(filelike, 'seek'):
            filelike.seek(0)
        try:
            return pd.read_excel(filelike, engine='openpyxl')
        except ImportError:
            raise ImportError(
                "Para leer archivos .xlsx necesitas instalar openpyxl.\n"
                "Ejecuta: pip install openpyxl>=3.0.0\n\n"
                "Alternativa: Exporta tu archivo a formato .csv y vuelve a subirlo."
            )
    
    elif extension == '.xls':
        if hasattr(filelike, 'seek'):
            filelike.seek(0)
        try:
            return pd.read_excel(filelike, engine='xlrd')
        except ImportError:
            raise ImportError(
                "Para leer archivos .xls necesitas instalar xlrd.\n"
                "Ejecuta: pip install xlrd>=2.0.1\n\n"
                "Alternativa: Exporta tu archivo a formato .csv o .xlsx y vuelve a subirlo."
            )
        except Exception:
            # Si falla por otro motivo (archivo corrupto, etc.), intentar fallback
            pass
    
    # Si no se detectó extensión o es desconocida, o falló lectura directa,
    # intentar en orden con fallback
    errors = []
    
    # Intento 1: CSV
    try:
        if hasattr(filelike, 'seek'):
            filelike.seek(0)
        return pd.read_csv(filelike)
    except Exception as e:
        errors.append(f"CSV: {str(e)}")
    
    # Intento 2: XLSX con openpyxl
    try:
        if hasattr(filelike, 'seek'):
            filelike.seek(0)
        return pd.read_excel(filelike, engine='openpyxl')
    except ImportError:
        errors.append("XLSX: falta instalar openpyxl (pip install openpyxl>=3.0.0)")
    except Exception as e:
        errors.append(f"XLSX: {str(e)}")
    
    # Intento 3: XLS con xlrd
    try:
        if hasattr(filelike, 'seek'):
            filelike.seek(0)
        return pd.read_excel(filelike, engine='xlrd')
    except ImportError:
        errors.append("XLS: falta instalar xlrd (pip install xlrd>=2.0.1)")
    except Exception as e:
        errors.append(f"XLS: {str(e)}")
    
    # Si todos los intentos fallaron, lanzar error detallado
    file_name = filelike.name if hasattr(filelike, 'name') else 'archivo desconocido'
    raise ValueError(
        f"No se pudo leer el archivo '{file_name}'.\n\n"
        f"Intentos realizados:\n" + "\n".join(f"  - {err}" for err in errors) + "\n\n"
        f"Soluciones:\n"
        f"  1. Para archivos .xlsx instala: pip install openpyxl>=3.0.0\n"
        f"  2. Para archivos .xls instala: pip install xlrd>=2.0.1\n"
        f"  3. Como alternativa, exporta el archivo a formato CSV.\n"
        f"\nFormatos soportados: .csv, .xlsx, .xls"
    )


def read_file(file_obj) -> pd.DataFrame:
    """
    Lee un archivo CSV o Excel y retorna un DataFrame.
    
    Esta función es un wrapper de read_table() para mantener compatibilidad
    con código existente.
    
    Parameters
    ----------
    file_obj : file-like object
        Objeto de archivo subido desde Streamlit.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con los datos del archivo.
        
    Raises
    ------
    ValueError
        Si el tipo de archivo no es soportado.
    ImportError
        Si falta una dependencia necesaria para leer el archivo (openpyxl o xlrd).
        
    See Also
    --------
    read_table : Función principal con manejo robusto de errores.
    """
    return read_table(file_obj)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nombres de columnas (elimina espacios, convierte a minúsculas, 
    reemplaza espacios por guiones bajos).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con columnas a normalizar.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas normalizadas.
    """
    df = df.copy()
    df.columns = [
        str(col).strip().lower().replace(' ', '_').replace('-', '_')
        for col in df.columns
    ]
    return df


def get_numeric_columns(df: pd.DataFrame, exclude: List[str] = None) -> List[str]:
    """
    Obtiene lista de columnas numéricas de un DataFrame, excluyendo las especificadas.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a analizar.
    exclude : List[str], optional
        Lista de columnas a excluir.
        
    Returns
    -------
    List[str]
        Lista de nombres de columnas numéricas.
    """
    if exclude is None:
        exclude = []
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude]


def validate_data_for_interpolation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    param_cols: List[str],
    min_points: int = 3
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Valida y limpia datos para interpolación.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos.
    x_col : str
        Nombre de columna X.
    y_col : str
        Nombre de columna Y.
    param_cols : List[str]
        Lista de columnas de parámetros.
    min_points : int
        Número mínimo de puntos requeridos.
        
    Returns
    -------
    df_clean : pd.DataFrame
        DataFrame limpio con valores válidos.
    warnings : Dict[str, str]
        Diccionario con advertencias por parámetro.
    """
    df_clean = df.copy()
    warnings = {}
    
    # Convertir a numérico
    for col in [x_col, y_col] + param_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Eliminar filas con X o Y faltantes
    original_len = len(df_clean)
    df_clean = df_clean.dropna(subset=[x_col, y_col])
    removed = original_len - len(df_clean)
    
    if removed > 0:
        warnings['xy_missing'] = f"Se eliminaron {removed} filas por valores faltantes en X o Y"
    
    # Verificar puntos mínimos
    if len(df_clean) < min_points:
        warnings['insufficient_points'] = f"Insuficientes puntos válidos: {len(df_clean)} (mínimo: {min_points})"
    
    # Verificar cada parámetro
    for param in param_cols:
        valid_count = df_clean[param].notna().sum()
        if valid_count < min_points:
            warnings[f'{param}_insufficient'] = f"Parámetro '{param}': solo {valid_count} puntos válidos"
    
    return df_clean, warnings


def calculate_parameter_statistics(
    df: pd.DataFrame,
    param_cols: List[str]
) -> pd.DataFrame:
    """
    Calcula estadísticas básicas para cada parámetro.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con datos.
    param_cols : List[str]
        Lista de columnas de parámetros.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con estadísticas (min, max, mean, std, n_points).
    """
    stats = []
    for param in param_cols:
        if param in df.columns:
            values = df[param].dropna()
            stats.append({
                'Parámetro': param,
                'n_puntos': len(values),
                'Mínimo': values.min() if len(values) > 0 else np.nan,
                'Máximo': values.max() if len(values) > 0 else np.nan,
                'Media': values.mean() if len(values) > 0 else np.nan,
                'Desv.Est.': values.std() if len(values) > 0 else np.nan
            })
    
    return pd.DataFrame(stats)


def make_xy_grid(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crea una grilla 2D regular para el plano X-Y.
    
    Parameters
    ----------
    x_min, x_max : float
        Rango del eje X (abscisa).
    y_min, y_max : float
        Rango del eje Y (cota/elevación).
    nx, ny : int
        Número de puntos en X y Y.
        
    Returns
    -------
    grid_x, grid_y : np.ndarray
        Grillas 2D (meshgrid) para interpolación.
    """
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    grid_x, grid_y = np.meshgrid(x, y)
    return grid_x, grid_y


def interpolate_xy_grid(
    points_xy: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    method: str = 'griddata',
    griddata_method: str = 'linear',
    rbf_func: str = 'multiquadric',
    idw_power: float = 2.0
) -> np.ndarray:
    """
    Interpola valores en una grilla X-Y usando el método especificado.
    
    Parameters
    ----------
    points_xy : np.ndarray, shape (n, 2)
        Coordenadas (x, y) de los puntos de datos.
    values : np.ndarray, shape (n,)
        Valores del parámetro en cada punto.
    grid_x, grid_y : np.ndarray
        Grillas 2D para interpolación.
    method : str
        Método de interpolación: 'griddata', 'rbf', 'idw'.
    griddata_method : str
        Método para griddata: 'linear', 'nearest', 'cubic'.
    rbf_func : str
        Función para RBF: 'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'.
    idw_power : float
        Potencia para IDW (típicamente 2.0).
        
    Returns
    -------
    grid_values : np.ndarray
        Valores interpolados en la grilla.
        
    Raises
    ------
    ValueError
        Si el método no es reconocido o hay error en interpolación.
    """
    if method == 'griddata':
        grid_values = griddata(
            points_xy,
            values,
            (grid_x, grid_y),
            method=griddata_method
        )
        
    elif method == 'rbf':
        rbf = Rbf(
            points_xy[:, 0],
            points_xy[:, 1],
            values,
            function=rbf_func
        )
        grid_values = rbf(grid_x, grid_y)
        
    elif method == 'idw':
        grid_values = idw_interpolate(
            points_xy,
            values,
            grid_x,
            grid_y,
            power=idw_power
        )
        
    else:
        raise ValueError(f"Método de interpolación desconocido: {method}")
    
    return grid_values


def idw_interpolate(
    points_xy: np.ndarray,
    values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    power: float = 2.0,
    search_radius: Optional[float] = None
) -> np.ndarray:
    """
    Interpolación IDW (Inverse Distance Weighting) implementada de forma vectorizada.
    
    IDW es un método de interpolación ponderado por la distancia inversa que calcula
    valores interpolados como un promedio ponderado de los puntos conocidos, donde
    los pesos son inversamente proporcionales a la distancia elevada a una potencia.
    
    Utiliza scipy.spatial.cKDTree para búsquedas eficientes de vecinos.
    
    Parameters
    ----------
    points_xy : np.ndarray, shape (n, 2)
        Coordenadas (x, y) de puntos de datos.
    values : np.ndarray, shape (n,)
        Valores en cada punto.
    grid_x, grid_y : np.ndarray
        Grillas 2D para interpolación.
    power : float
        Potencia para IDW. Valores típicos: 1-3. Mayor potencia = más peso a puntos cercanos.
    search_radius : Optional[float]
        Radio de búsqueda opcional. Si se especifica, solo se consideran puntos
        dentro de este radio para la interpolación (mejora rendimiento en datasets grandes).
        
    Returns
    -------
    grid_values : np.ndarray
        Valores interpolados en la grilla.
        
    Notes
    -----
    La implementación usa scipy.spatial.cKDTree para cálculos eficientes de distancia,
    especialmente útil para datasets grandes.
    """
    # Aplanar grilla
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    
    # Crear KDTree para búsquedas eficientes
    tree = cKDTree(points_xy)
    
    if search_radius is not None:
        # Usar búsqueda por radio para mejorar eficiencia
        # query_ball_point retorna índices de vecinos dentro del radio
        weights_list = []
        values_interp = []
        
        for point in grid_points:
            indices = tree.query_ball_point(point, search_radius)
            
            if len(indices) == 0:
                # No hay puntos dentro del radio
                values_interp.append(np.nan)
            else:
                # Calcular distancias solo a puntos dentro del radio
                dists = np.linalg.norm(points_xy[indices] - point, axis=1)
                
                # Evitar división por cero
                epsilon = 1e-10
                dists = np.maximum(dists, epsilon)
                
                # Calcular pesos
                w = 1.0 / (dists ** power)
                w_sum = np.sum(w)
                w = w / w_sum
                
                # Interpolar
                val = np.dot(w, values[indices])
                values_interp.append(val)
        
        grid_values = np.array(values_interp).reshape(grid_x.shape)
    
    else:
        # Calcular distancias usando cKDTree (más eficiente que cdist para grandes datasets)
        distances, _ = tree.query(grid_points, k=len(points_xy))
        
        # Si solo hay un punto, query retorna un escalar en lugar de array
        if len(points_xy) == 1:
            distances = distances.reshape(-1, 1)
        
        # Evitar división por cero
        epsilon = 1e-10
        distances = np.maximum(distances, epsilon)
        
        # Calcular pesos: w = 1 / d^power
        weights = 1.0 / (distances ** power)
        
        # Normalizar pesos
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        weights = weights / weights_sum
        
        # Interpolar: valor = suma(w_i * value_i)
        grid_values = np.dot(weights, values)
        
        # Reshape a forma de grilla
        grid_values = grid_values.reshape(grid_x.shape)
    
    return grid_values


def create_convexhull_mask(
    points_xy: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray
) -> np.ndarray:
    """
    Crea máscara basada en ConvexHull de los puntos de datos.
    Las celdas fuera del hull convexo se marcan como inválidas (False).
    
    Parameters
    ----------
    points_xy : np.ndarray, shape (n, 2)
        Coordenadas (x, y) de puntos de datos.
    grid_x, grid_y : np.ndarray
        Grillas 2D.
        
    Returns
    -------
    mask : np.ndarray
        Máscara booleana (True = dentro del hull, False = fuera).
        
    Notes
    -----
    Requiere al menos 4 puntos no colineales para crear un ConvexHull en 2D.
    Si hay menos puntos, retorna máscara toda True (sin enmascarar).
    """
    if len(points_xy) < 4:
        # No hay suficientes puntos para ConvexHull, retornar todo válido
        return np.ones_like(grid_x, dtype=bool)
    
    try:
        hull = ConvexHull(points_xy)
        
        # Aplanar grilla
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
        
        # Verificar qué puntos están dentro del hull
        # Método: un punto está dentro si todas las ecuaciones del hull son <= 0
        mask_flat = np.all(
            hull.equations[:, :2] @ grid_points.T + hull.equations[:, 2:3] <= 1e-12,
            axis=0
        )
        
        # Reshape a forma de grilla
        mask = mask_flat.reshape(grid_x.shape)
        
    except Exception as e:
        # Si falla ConvexHull (puntos colineales, etc.), retornar todo válido
        mask = np.ones_like(grid_x, dtype=bool)
    
    return mask


def create_distance_mask(
    points_xy: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    max_distance: Optional[float] = None,
    k_neighbors: int = 1
) -> np.ndarray:
    """
    Crea máscara basada en distancia a los k vecinos más cercanos.
    Las celdas con distancia mayor al umbral se marcan como inválidas.
    
    Utiliza scipy.spatial.cKDTree para búsquedas eficientes de k-vecinos más cercanos.
    
    Parameters
    ----------
    points_xy : np.ndarray, shape (n, 2)
        Coordenadas (x, y) de puntos de datos.
    grid_x, grid_y : np.ndarray
        Grillas 2D.
    max_distance : float, optional
        Distancia máxima permitida. Si None, se calcula automáticamente como
        percentil 90 de distancias entre puntos vecinos * factor de seguridad.
    k_neighbors : int
        Número de vecinos a considerar (típicamente 1).
        
    Returns
    -------
    mask : np.ndarray
        Máscara booleana (True = válido, False = muy lejos de datos).
        
    Notes
    -----
    Usa scipy.spatial.cKDTree para búsquedas eficientes de k-vecinos,
    significativamente más rápido que cálculos de distancia completos para datasets grandes.
    """
    # Aplanar grilla
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    
    # Crear KDTree para búsquedas eficientes de k-vecinos
    tree = cKDTree(points_xy)
    
    # Consultar distancia al k-ésimo vecino más cercano
    k = min(k_neighbors, len(points_xy))
    kth_distances, _ = tree.query(grid_points, k=k)
    
    # Si k > 1, tomar la distancia máxima entre los k vecinos
    if k > 1:
        kth_distances = np.max(kth_distances, axis=1)
    
    # Determinar umbral
    if max_distance is None:
        # Calcular umbral automático: analizar distancias típicas entre puntos
        if len(points_xy) > 1:
            # Usar KDTree para calcular distancias entre vecinos
            neighbor_distances, _ = tree.query(points_xy, k=min(2, len(points_xy)))
            if len(points_xy) > 1:
                # Tomar distancia al vecino más cercano (excluyendo el punto mismo)
                neighbor_distances = neighbor_distances[:, 1] if neighbor_distances.ndim > 1 else neighbor_distances
            
            # Usar percentil 90 de distancias como referencia
            typical_distance = np.percentile(neighbor_distances, 90)
            max_distance = typical_distance * 1.5  # Factor de seguridad
        else:
            max_distance = np.inf  # Sin umbral si solo hay 1 punto
    
    # Crear máscara
    mask_flat = kth_distances <= max_distance
    mask = mask_flat.reshape(grid_x.shape)
    
    return mask


def apply_mask_to_grid(
    grid_values: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Aplica máscara a la grilla (marca como NaN las celdas inválidas).
    
    Parameters
    ----------
    grid_values : np.ndarray
        Valores interpolados.
    mask : np.ndarray
        Máscara booleana (True = válido, False = enmascarar).
        
    Returns
    -------
    masked_grid : np.ndarray
        Grilla con máscara aplicada (NaN donde mask=False).
    """
    masked_grid = grid_values.copy()
    masked_grid[~mask] = np.nan
    return masked_grid


def combine_masks(mask1: np.ndarray, mask2: np.ndarray, operation: str = 'and') -> np.ndarray:
    """
    Combina dos máscaras con operación lógica.
    
    Parameters
    ----------
    mask1, mask2 : np.ndarray
        Máscaras booleanas a combinar.
    operation : str
        Operación: 'and' (intersección) u 'or' (unión).
        
    Returns
    -------
    np.ndarray
        Máscara combinada.
    """
    if operation == 'and':
        return mask1 & mask2
    elif operation == 'or':
        return mask1 | mask2
    else:
        raise ValueError(f"Operación desconocida: {operation}. Use 'and' u 'or'.")


def export_grid_to_dataframe(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_values: np.ndarray,
    param_name: str = 'Value'
) -> pd.DataFrame:
    """
    Convierte grilla interpolada a DataFrame para exportar.
    
    Parameters
    ----------
    grid_x, grid_y : np.ndarray
        Grillas 2D de coordenadas.
    grid_values : np.ndarray
        Valores interpolados.
    param_name : str
        Nombre del parámetro para la columna de valores.
        
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas X, Y, y el parámetro.
    """
    df = pd.DataFrame({
        'X': grid_x.ravel(),
        'Y': grid_y.ravel(),
        param_name: grid_values.ravel()
    })
    
    # No eliminamos NaN para mantener la estructura de la grilla
    # El usuario puede filtrar si lo desea
    
    return df


def subsample_data(
    df: pd.DataFrame,
    max_points: int = 10000,
    method: str = 'random',
    x_col: Optional[str] = None,
    y_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Submuestrea el DataFrame si tiene demasiados puntos.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a submuestrear.
    max_points : int
        Número máximo de puntos deseados.
    method : str
        Método de submuestreo: 'random' (aleatorio) o 'grid' (por rejilla espacial).
    x_col : Optional[str]
        Nombre de columna X (requerido para método 'grid').
    y_col : Optional[str]
        Nombre de columna Y (requerido para método 'grid').
        
    Returns
    -------
    pd.DataFrame
        DataFrame submuestreado.
        
    Notes
    -----
    El submuestreo por rejilla (grid) divide el espacio en celdas y toma
    un punto representativo por celda, útil para mantener distribución espacial
    uniforme en el dataset submuestreado.
    """
    if len(df) <= max_points:
        return df
    
    if method == 'random':
        return df.sample(n=max_points, random_state=42)
    
    elif method == 'grid':
        # Submuestreo espacial: dividir en grid y tomar puntos representativos
        if x_col is None or y_col is None:
            # Si no se especifican columnas, usar random
            return df.sample(n=max_points, random_state=42)
        
        # Calcular número de celdas en cada dimensión
        n_cells = int(np.sqrt(max_points))
        
        # Crear bins para X e Y
        x_bins = pd.cut(df[x_col], bins=n_cells, labels=False)
        y_bins = pd.cut(df[y_col], bins=n_cells, labels=False)
        
        # Agrupar por celda
        df_temp = df.copy()
        df_temp['_cell_x'] = x_bins
        df_temp['_cell_y'] = y_bins
        
        # Tomar un punto aleatorio por celda
        df_sampled = (df_temp
                     .groupby(['_cell_x', '_cell_y'], dropna=False)
                     .sample(n=1, random_state=42)
                     .drop(columns=['_cell_x', '_cell_y']))
        
        # Si no llegamos a max_points, complementar con random
        if len(df_sampled) < max_points:
            remaining = max_points - len(df_sampled)
            df_remaining = df[~df.index.isin(df_sampled.index)]
            if len(df_remaining) > 0:
                df_extra = df_remaining.sample(n=min(remaining, len(df_remaining)), random_state=42)
                df_sampled = pd.concat([df_sampled, df_extra])
        
        return df_sampled.reset_index(drop=True)
    
    else:
        raise ValueError(f"Método de submuestreo desconocido: {method}. Use 'random' o 'grid'.")


def plot_contour_between_id_minmax(
    df,
    x_col='x',
    y_col='y',
    z_col='z',
    id_col='ID',
    y_limits=None,          # tuple (ymin, ymax) o None
    n_levels=14,            # número de niveles de contorno (o lista de niveles)
    nx=300,
    ny=300,
    cmap='viridis',
    clip_to_range=True,
    scatter_size=8,
    title='Interpolación 2D',
    figsize=(10,6),
    prefer_method='cubic'   # 'cubic' o 'linear' (cubic intentará y caerá a linear si falla)
):
    """
    Crea un contourf de z sobre (x,y) limitando la región a un polígono formado
    por las cotas máximas (upper) y mínimas (lower) por cada ID (ordenado por x).
    
    Esta función es ideal para generar mapas de contorno de parámetros geotécnicos
    a lo largo de sondeos, donde se quiere limitar la visualización a la región
    entre las cotas mínimas y máximas de cada sondeo.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con las columnas especificadas (x_col, y_col, z_col, id_col).
    x_col : str, default='x'
        Nombre de la columna con coordenadas X (abscisa).
    y_col : str, default='y'
        Nombre de la columna con coordenadas Y (cota/elevación).
    z_col : str, default='z'
        Nombre de la columna con valores del parámetro a interpolar.
    id_col : str, default='ID'
        Nombre de la columna con identificadores de sondeos/puntos.
    y_limits : tuple or None, default=None
        Tupla (ymin, ymax) para limitar el rango Y. Si se especifica,
        el polígono se intersecta con esta franja horizontal y el eje Y
        se fija a estos límites.
    n_levels : int or array-like, default=14
        Número de niveles de contorno (int) o lista explícita de niveles.
    nx : int, default=300
        Resolución de la malla de interpolación en dirección X.
    ny : int, default=300
        Resolución de la malla de interpolación en dirección Y.
    cmap : str, default='viridis'
        Nombre del colormap de matplotlib.
    clip_to_range : bool, default=True
        Si True, recorta los valores interpolados al rango real de z
        para evitar overshoot de la interpolación.
    scatter_size : int, default=8
        Tamaño de los puntos de datos en el gráfico.
    title : str, default='Interpolación 2D'
        Título del gráfico.
    figsize : tuple, default=(10, 6)
        Tamaño de la figura (ancho, alto) en pulgadas.
    prefer_method : str, default='cubic'
        Método de interpolación preferido: 'cubic' o 'linear'.
        Si 'cubic' falla, se utiliza 'linear' automáticamente.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figura de matplotlib con el contorno generado.
    ax : matplotlib.axes.Axes
        Ejes de matplotlib de la figura.
    poly : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Polígono usado como máscara. Puede ser un Polygon simple o
        MultiPolygon si la intersección con y_limits genera múltiples partes.
        
    Raises
    ------
    RuntimeError
        - Si no hay sondajes en la columna id_col
        - Si el polígono generado es degenerado (área cero)
        - Si la intersección con y_limits produce un polígono vacío
        
    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'abscisa': [100, 100, 150, 150, 200, 200],
    ...     'cota': [50, 48, 52, 50, 51, 49],
    ...     'qc': [2.5, 3.2, 2.8, 3.5, 2.6, 3.3],
    ...     'ID': ['P-01', 'P-01', 'P-02', 'P-02', 'P-03', 'P-03']
    ... })
    >>> fig, ax, poly = plot_contour_between_id_minmax(
    ...     df, x_col='abscisa', y_col='cota', z_col='qc', id_col='ID'
    ... )
    >>> print(f"Área del polígono: {poly.area:.2f}")
    
    Notes
    -----
    - La función agrupa los datos por id_col y calcula el centroide X,
      y las cotas mínima y máxima por cada ID.
    - Los IDs se ordenan por su centroide X para construir el polígono.
    - El polígono se construye uniendo las cotas máximas (de izquierda a derecha)
      con las cotas mínimas (de derecha a izquierda).
    - Si shapely.vectorized.contains está disponible, se usa para el cálculo
      eficiente de la máscara. Si no, se usa matplotlib.path.Path como fallback.
    - Se requiere shapely instalado. Si no está disponible, el fallback con
      matplotlib.path.Path se usará automáticamente.
    
    See Also
    --------
    scipy.interpolate.griddata : Interpolación 2D usada internamente
    matplotlib.pyplot.contourf : Visualización de contornos
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    
    # Intentar importar shapely; si falla, usar matplotlib.path como fallback
    try:
        from shapely.geometry import Polygon, box
        from shapely.ops import unary_union
        SHAPELY_AVAILABLE = True
    except ImportError:
        SHAPELY_AVAILABLE = False
        from matplotlib.path import Path
        # Definir clase dummy Polygon para compatibilidad
        class Polygon:
            def __init__(self, points):
                self.points = points
                self.exterior = type('obj', (object,), {'coords': points})()
                self.geom_type = 'Polygon'
                # Calcular área usando fórmula del trapecio
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                self.area = 0.5 * abs(sum(x[i]*y[i+1] - x[i+1]*y[i] for i in range(len(points)-1)))
                self.is_valid = True
                self.is_empty = False
                
            def buffer(self, distance):
                return self
                
            def intersection(self, other):
                # Implementación simplificada - retorna self
                return self
                
            @property
            def bounds(self):
                x = [p[0] for p in self.points]
                y = [p[1] for p in self.points]
                return (min(x), min(y), max(x), max(y))

    # extraer arrays
    x = df[x_col].values
    y = df[y_col].values
    z = df[z_col].values

    # 1) agrupar por ID y calcular cx, ymin, ymax
    groups = df.groupby(id_col)
    rows = []
    for sid, g in groups:
        cx = g[x_col].mean()
        ymin = g[y_col].min()
        ymax = g[y_col].max()
        rows.append((sid, float(cx), float(ymin), float(ymax)))
    if len(rows) == 0:
        raise RuntimeError(f"No hay sondajes en df['{id_col}']")

    rows_arr = np.array(rows, dtype=object)
    rows_sorted = rows_arr[rows_arr[:,1].astype(float).argsort()]

    upper = [(float(r[1]), float(r[3])) for r in rows_sorted]   # (cx, ymax)
    lower = [(float(r[1]), float(r[2])) for r in rows_sorted]   # (cx, ymin)

    # construir polígono: subir por upper (izq->der), bajar por lower (der->izq)
    poly_points = []
    poly_points.extend(upper)
    poly_points.extend(reversed(lower))
    if poly_points[0] != poly_points[-1]:
        poly_points.append(poly_points[0])

    poly = Polygon(poly_points)
    
    if SHAPELY_AVAILABLE:
        poly = poly.buffer(0)  # buffer(0) limpia geometrías degeneradas
        if (not poly.is_valid) or (poly.area == 0):
            # fallback a convex hull
            poly = Polygon(poly_points).convex_hull
            if poly.area == 0:
                raise RuntimeError("El polígono generado es degenerado (área cero). Revisa los datos de ID/x/y.")
    
    # si se especifican límites de y, intersectar el polígono con esa franja
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if y_limits is not None and SHAPELY_AVAILABLE:
        from shapely.geometry import box
        ymin_lim, ymax_lim = float(y_limits[0]), float(y_limits[1])
        clip_box = box(xmin, ymin_lim, xmax, ymax_lim)
        poly = poly.intersection(clip_box)
        if poly.is_empty:
            raise RuntimeError("La intersección del polígono con y_limits produjo un polígono vacío.")

    # 2) crear grid e interpolar
    xi = np.linspace(xmin, xmax, nx)
    yi = np.linspace(np.min(y), np.max(y), ny)
    Xi, Yi = np.meshgrid(xi, yi)

    points = np.vstack((x, y)).T
    Zi = None
    if prefer_method == 'cubic':
        try:
            Zi = griddata(points, z, (Xi, Yi), method='cubic')
            # si cubic falla devolviendo todo NaN, usar linear
            if np.all(np.isnan(Zi)):
                Zi = griddata(points, z, (Xi, Yi), method='linear')
        except Exception:
            Zi = griddata(points, z, (Xi, Yi), method='linear')
    else:
        Zi = griddata(points, z, (Xi, Yi), method='linear')

    # 3) máscara: usar shapely.contains_xy si está disponible (shapely >= 2.0); si no, usar matplotlib.path.Path
    mask_inside = None
    if SHAPELY_AVAILABLE:
        try:
            # preferir shapely.contains_xy (shapely >= 2.0)
            from shapely import contains_xy
            mask_inside = contains_xy(poly, Xi, Yi)
        except (ImportError, AttributeError):
            try:
                # fallback a shapely.vectorized.contains (shapely < 2.0, deprecated)
                from shapely import vectorized
                mask_inside = vectorized.contains(poly, Xi, Yi)
            except Exception:
                # fallback con matplotlib.path.Path
                from matplotlib.path import Path
                if poly.geom_type == 'Polygon':
                    path = Path(list(poly.exterior.coords))
                    pts = np.column_stack((Xi.ravel(), Yi.ravel()))
                    mask_flat = path.contains_points(pts)
                    mask_inside = mask_flat.reshape(Xi.shape)
                else:
                    # MultiPolygon: unir máscaras
                    mask_flat = np.zeros(Xi.size, dtype=bool)
                    pts = np.column_stack((Xi.ravel(), Yi.ravel()))
                    for p in poly.geoms:
                        path = Path(list(p.exterior.coords))
                        mask_flat |= path.contains_points(pts)
                    mask_inside = mask_flat.reshape(Xi.shape)
    else:
        # usar matplotlib.path.Path directamente
        from matplotlib.path import Path
        path = Path(poly.points)
        pts = np.column_stack((Xi.ravel(), Yi.ravel()))
        mask_flat = path.contains_points(pts)
        mask_inside = mask_flat.reshape(Xi.shape)

    # 4) enmascarar fuera del polígono o NaN
    Zi_masked = np.ma.array(Zi, mask=(~mask_inside) | np.isnan(Zi))

    # 5) opcional: clip al rango real de z para evitar overshoot
    if clip_to_range:
        zmin, zmax = np.nanmin(z), np.nanmax(z)
        data = Zi_masked.data.copy()
        data = np.clip(data, zmin, zmax)
        Zi_masked = np.ma.array(data, mask=Zi_masked.mask)

    # 6) niveles: si n_levels es int, generar linspace entre zmin y zmax
    if isinstance(n_levels, int):
        zmin, zmax = np.nanmin(z), np.nanmax(z)
        levels = np.linspace(zmin, zmax, n_levels)
    else:
        levels = np.asarray(n_levels)

    # 7) graficar
    fig, ax = plt.subplots(figsize=figsize)
    cf = ax.contourf(Xi, Yi, Zi_masked, levels=levels, cmap=cmap, extend='neither')
    cs = ax.contour(Xi, Yi, Zi_masked, levels=levels, colors='k', linewidths=0.5, alpha=0.6)
    

    # polígono borde
    if SHAPELY_AVAILABLE and poly.geom_type == 'Polygon':
        xp, yp = poly.exterior.xy
        ax.plot(xp, yp, color='k', linewidth=1.2, alpha=0.9, label='polígono (min/max por ID)')
    elif SHAPELY_AVAILABLE:
        # MultiPolygon
        for p in poly.geoms:
            xp, yp = p.exterior.xy
            ax.plot(xp, yp, color='k', linewidth=1.2, alpha=0.9)
    else:
        # Fallback sin shapely
        xp = [p[0] for p in poly.points]
        yp = [p[1] for p in poly.points]
        ax.plot(xp, yp, color='k', linewidth=1.2, alpha=0.9, label='polígono (min/max por ID)')

    

    # ejes y título
    if y_limits is not None:
        ax.set_ylim(y_limits)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    plt.colorbar(cf, ax=ax, label=z_col)
    

    plt.tight_layout()
    return fig, ax, poly


def export_interpolated_grid_to_csv(grid_x, grid_y, grid_values, include_masked=False):
    """
    Exporta la grilla interpolada a formato CSV (x, y, value).
    
    Parameters
    ----------
    grid_x : np.ndarray
        Coordenadas X de la grilla (meshgrid).
    grid_y : np.ndarray
        Coordenadas Y de la grilla (meshgrid).
    grid_values : np.ndarray or np.ma.MaskedArray
        Valores interpolados en la grilla.
    include_masked : bool, default=False
        Si True, incluye puntos enmascarados con valor NaN.
        Si False, solo exporta puntos válidos (no enmascarados).
        
    Returns
    -------
    str
        String con el contenido CSV (x,y,value).
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[0, 1], [0, 1]])
    >>> y = np.array([[0, 0], [1, 1]])
    >>> v = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> csv = export_interpolated_grid_to_csv(x, y, v)
    >>> print(csv.split('\\n')[0])  # header
    x,y,value
    """
    # Aplanar las grillas
    x_flat = grid_x.ravel()
    y_flat = grid_y.ravel()
    
    # Manejar MaskedArray
    if isinstance(grid_values, np.ma.MaskedArray):
        v_flat = grid_values.filled(np.nan).ravel()
    else:
        v_flat = grid_values.ravel()
    
    # Crear DataFrame
    df_export = pd.DataFrame({
        'x': x_flat,
        'y': y_flat,
        'value': v_flat
    })
    
    # Filtrar NaN si no se requieren
    if not include_masked:
        df_export = df_export.dropna(subset=['value'])
    
    # Convertir a CSV
    return df_export.to_csv(index=False)


def figure_to_bytes(fig, format='png', dpi=300):
    """
    Convierte una figura de matplotlib a bytes para descarga.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figura de matplotlib a convertir.
    format : str, default='png'
        Formato de imagen: 'png', 'jpg', 'svg', 'pdf'.
    dpi : int, default=300
        Resolución en puntos por pulgada (para formatos raster).
        
    Returns
    -------
    bytes
        Bytes de la imagen en el formato especificado.
        
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 2, 3])
    >>> img_bytes = figure_to_bytes(fig)
    >>> isinstance(img_bytes, bytes)
    True
    """
    from io import BytesIO
    
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def polygon_to_geojson(poly):
    """
    Convierte un polígono de shapely a formato GeoJSON.
    
    Parameters
    ----------
    poly : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Polígono a convertir.
        
    Returns
    -------
    dict
        Diccionario con la geometría en formato GeoJSON.
        
    Examples
    --------
    >>> from shapely.geometry import Polygon
    >>> poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    >>> geojson = polygon_to_geojson(poly)
    >>> geojson['type']
    'Feature'
    
    Notes
    -----
    Requiere shapely instalado. Si no está disponible, retorna None.
    """
    try:
        from shapely.geometry import mapping
        return {
            'type': 'Feature',
            'geometry': mapping(poly),
            'properties': {
                'area': poly.area,
                'bounds': poly.bounds
            }
        }
    except ImportError:
        return None
