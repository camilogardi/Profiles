# Generador de Perfiles Geotécnicos

Una aplicación Streamlit para generar **perfiles verticales** (X vs Elevación vs Parámetro) de parámetros geotécnicos a partir de datos de sondeos.

## 🎯 Características Principales

- **Entrada de dos archivos separados:**
  - Archivo A: Cabeceras de sondeos (ID, x, y, cota inicial)
  - Archivo B: Ensayos por profundidad (ID, profundidad, parámetros medidos)

- **Visualización de perfiles verticales:**
  - Sección transversal X-Z (posición horizontal vs elevación)
  - Interpolación 2D en el plano vertical
  - Enmascaramiento automático de zonas sin cobertura vertical

- **Múltiples métodos de interpolación:**
  - Griddata: linear, nearest, cubic
  - RBF (Radial Basis Function): multiquadric, inverse, gaussian, linear, cubic, quintic
  - IDW (Inverse Distance Weighting): con potencia configurable

- **Opciones de ordenación de sondeos:**
  - Por coordenada X real
  - Por X luego Y (ordenación secuencial)
  - Por proyección PCA (útil para transectos oblicuos)

- **Exportación:**
  - Figuras PNG de alta resolución (300 dpi)
  - Grilla interpolada en formato CSV

## 📋 Requisitos

### Dependencias

```bash
pip install -r streamlit_app/requirements.txt
```

Dependencias principales:
- Python >= 3.8
- streamlit >= 1.28.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- scikit-learn >= 1.3.0
- matplotlib >= 3.7.0
- openpyxl >= 3.1.0

## 🚀 Instalación y Ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/camilogardi/Profiles.git
cd Profiles
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# En Windows:
venv\Scripts\activate

# En Linux/Mac:
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r streamlit_app/requirements.txt
```

### 4. Ejecutar la aplicación

```bash
streamlit run streamlit_app/app.py
```

La aplicación se abrirá en `http://localhost:8501`

## 📁 Formato de Archivos de Entrada

### Archivo A: Cabeceras de Sondeos

Columnas requeridas (nombres personalizables en UI):
- **ID_sondeo**: Identificador único del sondeo
- **x**: Coordenada Este (Easting)
- **y**: Coordenada Norte (Northing)
- **cota**: Elevación de la cabeza del sondeo

Ejemplo: `examples/example_headers.csv`

```csv
ID,x,y,cota
S-01,100,200,50.5
S-02,150,205,52.3
S-03,200,198,51.8
```

### Archivo B: Ensayos por Profundidad

Columnas requeridas:
- **ID_sondeo**: Identificador del sondeo (debe coincidir con Archivo A)
- **profundidad**: Profundidad del ensayo desde la cota
- **parámetro(s)**: Columnas con valores geotécnicos (peso_unitario, SPT, etc.)

Ejemplo: `examples/example_samples.csv`

```csv
ID,profundidad,peso_unitario,limite_liquido,SPT,humedad
S-01,0.5,18.5,35,10,22
S-01,2.0,19.2,38,15,25
S-01,4.0,19.8,42,18,28
```

## 🎨 Flujo de Uso

1. **Cargar archivos**: Sube ambos archivos (cabeceras y ensayos)
2. **Mapear columnas**: Indica qué columnas corresponden a cada variable
3. **Revisar datos**: Verifica el resumen de sondeos y límites verticales
4. **Configurar**: Usa el panel lateral para ajustar interpolación y visualización
5. **Generar perfil**: Presiona el botón para crear el perfil vertical
6. **Exportar**: Descarga la figura PNG o los datos interpolados CSV

## 📊 Ordenación de Sondeos

- **Coordenada X real**: Usa coordenadas X directamente
- **Ordenar por X, luego Y**: Ordena secuencialmente
- **Proyección PCA**: Proyecta sobre eje principal (útil para transectos oblicuos)

### ¿Por qué usar PCA?
PCA encuentra automáticamente la dirección de máxima variación en las coordenadas de los sondeos. Es útil cuando:
- Los sondeos forman un transecto diagonal
- No están alineados con los ejes X/Y cardinales
- Se quiere visualizar un perfil a lo largo de cualquier dirección

## 🔧 Archivos de Ejemplo

El directorio `examples/` contiene datos de prueba:
- `example_headers.csv`: 10 sondeos
- `example_samples.csv`: 50 ensayos con 4 parámetros

## 📚 Documentación Adicional

Ver [USAGE.md](USAGE.md) para documentación detallada sobre el uso de la aplicación original de contornos horizontales.

---

## Aplicaciones Incluidas

Este repositorio contiene dos aplicaciones:

1. **streamlit_app/app.py** - Generador de Perfiles Verticales (NUEVO)
2. **contornos_app/contornos.py** - Herramienta de Contornos Horizontales (original)

Para más información sobre la aplicación de contornos horizontales, consulta [USAGE.md](USAGE.md).
