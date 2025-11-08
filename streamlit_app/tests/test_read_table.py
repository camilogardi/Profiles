"""
Pruebas unitarias para la función read_table().

Este módulo verifica la lectura correcta de archivos CSV y Excel (XLSX/XLS)
en memoria, validando que read_table() funciona con objetos BytesIO y maneja
correctamente las dependencias opcionales (openpyxl, xlrd).

Para ejecutar las pruebas:
    pip install -r streamlit_app/requirements.txt
    pip install pytest
    pytest -v streamlit_app/tests/test_read_table.py
"""

import pytest
import pandas as pd
from io import BytesIO
import sys
import os

# Añadir el directorio padre al path para poder importar utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import read_table


def create_sample_dataframe():
    """
    Crea un DataFrame de prueba pequeño (3 filas, 3 columnas).
    
    Returns
    -------
    pd.DataFrame
        DataFrame con datos de ejemplo.
    """
    return pd.DataFrame({
        'abscisa': [100.0, 150.0, 200.0],
        'cota': [50.5, 52.3, 51.8],
        'qc': [2.5, 2.8, 2.6]
    })


def test_read_table_csv():
    """
    Test: Lectura exitosa de archivo CSV desde BytesIO.
    
    Verifica que:
    - read_table() puede leer CSV desde memoria
    - El shape del DataFrame resultante es correcto (3 filas, 3 columnas)
    - Los datos se leen correctamente
    """
    # Crear DataFrame de prueba
    df_original = create_sample_dataframe()
    
    # Convertir a CSV en memoria
    csv_buffer = BytesIO()
    df_original.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)  # Resetear puntero al inicio
    
    # Simular atributo .name para compatibilidad con Streamlit UploadedFile
    csv_buffer.name = 'test_file.csv'
    
    # Leer usando read_table
    df_result = read_table(csv_buffer)
    
    # Verificaciones
    assert df_result.shape == (3, 3), f"Shape esperado (3, 3), obtenido {df_result.shape}"
    assert list(df_result.columns) == ['abscisa', 'cota', 'qc'], \
        f"Columnas esperadas ['abscisa', 'cota', 'qc'], obtenidas {list(df_result.columns)}"
    assert df_result['abscisa'].tolist() == [100.0, 150.0, 200.0], \
        "Los valores de 'abscisa' no coinciden"
    
    print("✓ Test CSV: Lectura exitosa")


def test_read_table_xlsx():
    """
    Test: Lectura exitosa de archivo XLSX desde BytesIO.
    
    Verifica que:
    - read_table() puede leer XLSX desde memoria usando engine='openpyxl'
    - El shape del DataFrame resultante es correcto (3 filas, 3 columnas)
    - Los datos se leen correctamente
    
    Si openpyxl no está instalado, el test falla con ImportError apropiado.
    """
    try:
        import openpyxl
    except ImportError:
        pytest.skip("openpyxl no está instalado. Instala con: pip install openpyxl>=3.0.0")
    
    # Crear DataFrame de prueba
    df_original = create_sample_dataframe()
    
    # Convertir a XLSX en memoria usando openpyxl
    xlsx_buffer = BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine='openpyxl') as writer:
        df_original.to_excel(writer, index=False, sheet_name='Sheet1')
    xlsx_buffer.seek(0)  # Resetear puntero al inicio
    
    # Simular atributo .name
    xlsx_buffer.name = 'test_file.xlsx'
    
    # Leer usando read_table
    df_result = read_table(xlsx_buffer)
    
    # Verificaciones
    assert df_result.shape == (3, 3), f"Shape esperado (3, 3), obtenido {df_result.shape}"
    assert list(df_result.columns) == ['abscisa', 'cota', 'qc'], \
        f"Columnas esperadas ['abscisa', 'cota', 'qc'], obtenidas {list(df_result.columns)}"
    assert df_result['abscisa'].tolist() == [100.0, 150.0, 200.0], \
        "Los valores de 'abscisa' no coinciden"
    
    print("✓ Test XLSX: Lectura exitosa (openpyxl instalado)")


def test_read_table_xls_handling():
    """
    Test: Manejo correcto de archivos XLS.
    
    Este test verifica que read_table() intenta usar xlrd para archivos .xls
    y proporciona un mensaje de error claro si xlrd no está instalado.
    
    Nota: xlrd>=2.0 solo soporta formato .xls antiguo (no .xlsx).
    Para archivos Excel modernos, usar .xlsx con openpyxl.
    """
    # Para este test, simplemente verificamos que si se pasa un .xls sin xlrd,
    # se obtiene un ImportError con mensaje apropiado
    
    # Crear un archivo que simula ser .xls pero es realmente CSV
    # (ya que crear un .xls real requiere xlwt u otra librería)
    df_original = create_sample_dataframe()
    
    csv_buffer = BytesIO()
    df_original.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    csv_buffer.name = 'test_file.xls'  # Simular extensión .xls
    
    # Si xlrd está instalado, debería fallar al intentar leer CSV como XLS
    # Si xlrd NO está instalado, debería dar ImportError con mensaje claro
    # En ambos casos, el fallback debería intentar CSV y tener éxito
    
    try:
        df_result = read_table(csv_buffer)
        # Si llegó aquí, el fallback a CSV funcionó
        assert df_result.shape == (3, 3), "Fallback a CSV debería funcionar"
        print("✓ Test XLS: Fallback a CSV funcionó correctamente")
    except ImportError as e:
        # Si hay ImportError, verificar que el mensaje es claro
        error_msg = str(e)
        assert 'xlrd' in error_msg.lower(), \
            f"ImportError debería mencionar xlrd, mensaje: {error_msg}"
        print(f"✓ Test XLS: ImportError apropiado detectado: {error_msg[:100]}...")


def test_read_table_fallback():
    """
    Test: Verificar que el fallback funciona cuando no se detecta extensión.
    
    Crea un archivo sin extensión reconocible y verifica que read_table()
    intenta leer como CSV automáticamente.
    """
    df_original = create_sample_dataframe()
    
    # CSV sin nombre o con nombre sin extensión
    csv_buffer = BytesIO()
    df_original.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    csv_buffer.name = 'test_file'  # Sin extensión
    
    # Debería intentar CSV como fallback
    df_result = read_table(csv_buffer)
    
    assert df_result.shape == (3, 3), "Fallback debería leer CSV correctamente"
    print("✓ Test Fallback: Lectura sin extensión funcionó")


def test_read_table_seek_functionality():
    """
    Test: Verificar que read_table() resetea el puntero del archivo correctamente.
    
    Lee el mismo buffer dos veces para asegurar que seek(0) funciona.
    """
    df_original = create_sample_dataframe()
    
    csv_buffer = BytesIO()
    df_original.to_csv(csv_buffer, index=False)
    csv_buffer.name = 'test_file.csv'
    
    # Primera lectura
    df_result1 = read_table(csv_buffer)
    assert df_result1.shape == (3, 3), "Primera lectura falló"
    
    # El buffer ahora está al final. read_table debería hacer seek(0) internamente
    # pero para segunda lectura necesitamos hacerlo manualmente
    csv_buffer.seek(0)
    
    # Segunda lectura
    df_result2 = read_table(csv_buffer)
    assert df_result2.shape == (3, 3), "Segunda lectura falló"
    
    print("✓ Test Seek: Múltiples lecturas funcionan correctamente")


def test_read_table_invalid_file():
    """
    Test: Verificar manejo de archivos inválidos.
    
    Intenta leer un archivo con contenido que no es CSV/Excel válido.
    Nota: pandas.read_csv puede leer casi cualquier cosa como CSV de una columna,
    por lo que este test verifica que al menos no falla catastróficamente.
    """
    # Crear buffer con contenido que no es un CSV estructurado
    invalid_buffer = BytesIO(b"This is not a valid CSV or Excel file 12345!@#$%")
    invalid_buffer.name = 'invalid.dat'
    
    # Pandas podría leer esto como CSV de una sola fila/columna
    # Lo importante es que no crashee
    try:
        df_result = read_table(invalid_buffer)
        # Si se leyó, verificar que al menos devolvió un DataFrame
        assert isinstance(df_result, pd.DataFrame), "Debería devolver un DataFrame"
        print("✓ Test Invalid: pandas leyó el contenido como CSV (comportamiento esperado)")
    except (ValueError, pd.errors.ParserError) as e:
        # Si lanza error, debería ser descriptivo
        error_msg = str(e)
        print(f"✓ Test Invalid: Error apropiado detectado: {error_msg[:100]}...")


def test_read_table_missing_openpyxl():
    """
    Test: Verificar mensaje de error cuando falta openpyxl para XLSX.
    
    Este test verifica que si openpyxl no está instalado y se intenta
    leer un .xlsx, el error es claro e incluye instrucciones.
    
    Nota: Este test es informativo si openpyxl está instalado.
    """
    # Solo ejecutar si openpyxl NO está instalado
    try:
        import openpyxl
        pytest.skip("openpyxl está instalado, no se puede probar ImportError")
    except ImportError:
        pass
    
    # Crear buffer fingiendo ser XLSX
    fake_xlsx = BytesIO(b"PK\x03\x04")  # Firma de archivo ZIP (XLSX es un ZIP)
    fake_xlsx.name = 'test.xlsx'
    
    # Debería lanzar ImportError con mensaje claro
    with pytest.raises(ImportError) as exc_info:
        read_table(fake_xlsx)
    
    error_msg = str(exc_info.value)
    assert 'openpyxl' in error_msg.lower(), \
        f"Error debería mencionar openpyxl: {error_msg}"
    assert 'pip install' in error_msg.lower(), \
        f"Error debería incluir instrucciones de instalación: {error_msg}"
    
    print("✓ Test Missing openpyxl: Error apropiado cuando falta dependencia")


# Función principal para ejecutar tests con pytest
if __name__ == '__main__':
    # Ejecutar con pytest si está disponible
    print("=" * 70)
    print("Pruebas unitarias para read_table()")
    print("=" * 70)
    print("\nEjecutando pruebas...\n")
    
    # Ejecutar cada test manualmente para ver resultados
    try:
        test_read_table_csv()
    except Exception as e:
        print(f"✗ Test CSV falló: {e}")
    
    try:
        test_read_table_xlsx()
    except Exception as e:
        print(f"✗ Test XLSX falló: {e}")
    
    try:
        test_read_table_xls_handling()
    except Exception as e:
        print(f"✗ Test XLS falló: {e}")
    
    try:
        test_read_table_fallback()
    except Exception as e:
        print(f"✗ Test Fallback falló: {e}")
    
    try:
        test_read_table_seek_functionality()
    except Exception as e:
        print(f"✗ Test Seek falló: {e}")
    
    try:
        test_read_table_invalid_file()
    except Exception as e:
        print(f"✗ Test Invalid falló: {e}")
    
    print("\n" + "=" * 70)
    print("Tests completados. Usa 'pytest -v' para ejecución completa.")
    print("=" * 70)
