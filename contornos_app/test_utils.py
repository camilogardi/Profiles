"""
Unit tests for utility functions in the geotechnical contours application.
Run with: python -m pytest test_utils.py
"""
import numpy as np
import pandas as pd
from io import BytesIO
import pytest

from utils import (
    read_table,
    make_grid,
    interpolate_grid,
    idw_interpolate,
    compute_z_bounds
)


class TestReadTable:
    """Tests for read_table function."""
    
    def test_read_csv(self):
        """Test reading CSV data."""
        csv_data = "col1,col2,col3\n1,2,3\n4,5,6\n"
        csv_bytes = BytesIO(csv_data.encode())
        csv_bytes.name = "test.csv"
        
        df = read_table(csv_bytes)
        assert len(df) == 2
        assert list(df.columns) == ['col1', 'col2', 'col3']
    
    def test_unsupported_format(self):
        """Test that unsupported formats raise ValueError."""
        file_obj = BytesIO(b"data")
        file_obj.name = "test.txt"
        
        with pytest.raises(ValueError, match="Unsupported file type"):
            read_table(file_obj)


class TestMakeGrid:
    """Tests for make_grid function."""
    
    def test_grid_shape(self):
        """Test that grid has correct shape."""
        grid_x, grid_y = make_grid(0, 100, 0, 100, 10, 20)
        assert grid_x.shape == (20, 10)
        assert grid_y.shape == (20, 10)
    
    def test_grid_bounds(self):
        """Test that grid covers specified bounds."""
        xmin, xmax = 10, 50
        ymin, ymax = 20, 80
        grid_x, grid_y = make_grid(xmin, xmax, ymin, ymax, 5, 5)
        
        assert np.isclose(grid_x.min(), xmin)
        assert np.isclose(grid_x.max(), xmax)
        assert np.isclose(grid_y.min(), ymin)
        assert np.isclose(grid_y.max(), ymax)


class TestInterpolateGrid:
    """Tests for interpolate_grid function."""
    
    def setup_method(self):
        """Set up test data."""
        self.points = np.array([[0, 0], [10, 10], [10, 0], [0, 10]])
        self.values = np.array([1.0, 2.0, 3.0, 4.0])
        self.grid_x, self.grid_y = make_grid(-1, 11, -1, 11, 10, 10)
    
    def test_griddata_linear(self):
        """Test linear griddata interpolation."""
        result = interpolate_grid(
            self.points, self.values, self.grid_x, self.grid_y,
            method='griddata', griddata_method='linear'
        )
        assert result.shape == (10, 10)
    
    def test_griddata_nearest(self):
        """Test nearest neighbor griddata interpolation."""
        result = interpolate_grid(
            self.points, self.values, self.grid_x, self.grid_y,
            method='griddata', griddata_method='nearest'
        )
        assert result.shape == (10, 10)
        assert not np.any(np.isnan(result))
    
    def test_rbf(self):
        """Test RBF interpolation."""
        result = interpolate_grid(
            self.points, self.values, self.grid_x, self.grid_y,
            method='rbf', rbf_func='multiquadric'
        )
        assert result.shape == (10, 10)
    
    def test_unknown_method(self):
        """Test that unknown methods raise ValueError."""
        with pytest.raises(ValueError, match="Unknown interpolation method"):
            interpolate_grid(
                self.points, self.values, self.grid_x, self.grid_y,
                method='invalid_method'
            )


class TestIDWInterpolate:
    """Tests for idw_interpolate function."""
    
    def test_basic_interpolation(self):
        """Test basic IDW interpolation."""
        points = np.array([[0, 0], [10, 10], [10, 0], [0, 10]])
        values = np.array([1.0, 2.0, 3.0, 4.0])
        grid_x, grid_y = make_grid(-1, 11, -1, 11, 10, 10)
        
        result = idw_interpolate(points, values, grid_x, grid_y, power=2.0)
        
        assert result.shape == (10, 10)
        assert not np.any(np.isnan(result))
    
    def test_exact_at_data_points(self):
        """Test that IDW returns exact values at data points."""
        points = np.array([[5.0, 5.0]])
        values = np.array([10.0])
        grid_x, grid_y = make_grid(5, 5, 5, 5, 1, 1)
        
        result = idw_interpolate(points, values, grid_x, grid_y, power=2.0)
        
        assert np.isclose(result[0, 0], 10.0, atol=0.1)
    
    def test_different_powers(self):
        """Test IDW with different power parameters."""
        points = np.array([[0, 0], [10, 10]])
        values = np.array([0.0, 10.0])
        grid_x, grid_y = make_grid(0, 10, 0, 10, 5, 5)
        
        result1 = idw_interpolate(points, values, grid_x, grid_y, power=1.0)
        result2 = idw_interpolate(points, values, grid_x, grid_y, power=4.0)
        
        # Higher power should create steeper gradients
        assert not np.allclose(result1, result2)


class TestComputeZBounds:
    """Tests for compute_z_bounds function."""
    
    def test_basic_bounds(self):
        """Test basic z bounds computation."""
        df = pd.DataFrame({
            'z_top': [10, 20, 30],
            'z_bottom': [0, 5, 10]
        })
        
        z_min, z_max = compute_z_bounds(df)
        
        assert z_min == 0
        assert z_max == 30
    
    def test_single_row(self):
        """Test bounds with single row."""
        df = pd.DataFrame({
            'z_top': [50],
            'z_bottom': [30]
        })
        
        z_min, z_max = compute_z_bounds(df)
        
        assert z_min == 30
        assert z_max == 50


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
