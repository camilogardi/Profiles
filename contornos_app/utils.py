"""
Utility functions for geotechnical contour interpolation.
"""
import pandas as pd
import numpy as np
from scipy.interpolate import griddata, Rbf


def read_table(file_obj):
    """
    Read a CSV or Excel file into a pandas DataFrame.
    
    Parameters
    ----------
    file_obj : file-like object
        The uploaded file object from Streamlit.
        
    Returns
    -------
    pd.DataFrame
        The loaded dataframe.
    """
    file_name = file_obj.name.lower()
    if file_name.endswith('.csv'):
        return pd.read_csv(file_obj)
    elif file_name.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_obj)
    else:
        raise ValueError(f"Unsupported file type: {file_name}")


def make_grid(xmin, xmax, ymin, ymax, nx, ny):
    """
    Create a regular grid for interpolation.
    
    Parameters
    ----------
    xmin, xmax : float
        Min and max x coordinates.
    ymin, ymax : float
        Min and max y coordinates.
    nx, ny : int
        Number of grid points in x and y directions.
        
    Returns
    -------
    grid_x, grid_y : 2D arrays
        Meshgrid arrays for interpolation.
    """
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(x, y)
    return grid_x, grid_y


def interpolate_grid(points, values, grid_x, grid_y, method='griddata', 
                     griddata_method='linear', rbf_func='multiquadric'):
    """
    Interpolate scattered data onto a regular grid.
    
    Parameters
    ----------
    points : ndarray, shape (n, 2)
        Data point coordinates (x, y).
    values : ndarray, shape (n,)
        Data values at each point.
    grid_x, grid_y : 2D arrays
        Grid coordinates for interpolation.
    method : str
        Interpolation method: 'griddata' or 'rbf'.
    griddata_method : str
        Method for griddata: 'linear', 'nearest', or 'cubic'.
    rbf_func : str
        RBF function type for RBF interpolation.
        
    Returns
    -------
    grid_z : 2D array
        Interpolated values on the grid.
    """
    if method == 'griddata':
        grid_z = griddata(points, values, (grid_x, grid_y), method=griddata_method)
    elif method == 'rbf':
        rbf = Rbf(points[:, 0], points[:, 1], values, function=rbf_func)
        grid_z = rbf(grid_x, grid_y)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
    
    return grid_z


def idw_interpolate(points, values, grid_x, grid_y, power=2.0):
    """
    Inverse Distance Weighting (IDW) interpolation.
    
    Parameters
    ----------
    points : ndarray, shape (n, 2)
        Data point coordinates (x, y).
    values : ndarray, shape (n,)
        Data values at each point.
    grid_x, grid_y : 2D arrays
        Grid coordinates for interpolation.
    power : float
        Power parameter for IDW (typically 2.0).
        
    Returns
    -------
    grid_z : 2D array
        Interpolated values on the grid.
        
    Notes
    -----
    For very large grids (>1000x1000), this function may consume significant memory
    due to the distance matrix. Consider using lower resolution grids for large areas.
    """
    # Flatten the grid
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    
    # Compute distances from each grid point to all data points using vectorized operations
    # Shape: (n_grid_points, n_data_points)
    # Note: This creates a large intermediate array in memory
    distances = np.sqrt(np.sum((points[None, :, :] - grid_points[:, None, :]) ** 2, axis=2))
    
    # Avoid division by zero - if distance is very small, use the point value directly
    epsilon = 1e-10
    distances = np.maximum(distances, epsilon)
    
    # Compute weights
    weights = 1.0 / (distances ** power)
    
    # Normalize weights
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights = weights / weights_sum
    
    # Compute interpolated values
    grid_values = np.dot(weights, values)
    
    # Reshape to grid shape
    grid_z = grid_values.reshape(grid_x.shape)
    
    return grid_z


def compute_z_bounds(df):
    """
    Compute the minimum and maximum z (elevation) bounds from the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'z_top' and 'z_bottom' columns.
        
    Returns
    -------
    z_min, z_max : float
        Minimum and maximum z values.
    """
    z_min = df['z_bottom'].min()
    z_max = df['z_top'].max()
    return z_min, z_max
