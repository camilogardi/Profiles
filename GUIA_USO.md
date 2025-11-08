# Guía de Uso - Interpolación 2D de Parámetros Geotécnicos

## Inicio Rápido

### 1. Preparar tus datos

Crea un archivo CSV o Excel con la siguiente estructura:

```csv
id,x,y,parametro1,parametro2,...
P-01,100.5,50.2,2.5,18.5
P-02,150.3,52.1,3.2,19.1
...
```

**Requisitos:**
- Al menos 3 filas con datos válidos
- Columnas X e Y con coordenadas numéricas
- Una o más columnas con parámetros numéricos

### 2. Ejecutar la aplicación

```bash
cd Profiles
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
streamlit run streamlit_app/app.py
```

### 3. Usar la interfaz

1. **Cargar archivo**: Haz clic en "Browse files" y selecciona tu CSV o Excel
2. **Mapear columnas**: 
   - Indica cuál columna es X (abscisa/este)
   - Indica cuál columna es Y (cota/elevación/norte)
3. **Seleccionar parámetros**: Marca los parámetros que quieres interpolar
4. **Configurar (sidebar)**:
   - Ajusta resolución de grilla (100×100 es buen punto de partida)
   - Elige método de interpolación (recomendado: Griddata - Linear)
   - Selecciona máscara (recomendado: ConvexHull)
5. **Generar**: Presiona el botón "Generar mapas de contorno"
6. **Exportar**: Descarga las figuras PNG o los datos CSV

## Configuración Avanzada

### Resolución de Grilla

- **50×50**: Vista rápida, baja calidad
- **100×100**: Uso general (recomendado)
- **200×200**: Alta calidad para figuras finales
- **>300×300**: Solo para datasets grandes con muchos puntos

### Métodos de Interpolación

#### Griddata Linear (recomendado)
- **Ventajas**: Rápido, estable, resultados suaves
- **Cuándo usar**: Primera opción para mayoría de casos
- **Limitaciones**: Puede dejar huecos (NaN) fuera de triangulación

#### Griddata Nearest
- **Ventajas**: Preserva valores discretos
- **Cuándo usar**: Datos categóricos o discretos
- **Limitaciones**: Aspecto "escalonado"

#### Griddata Cubic
- **Ventajas**: Muy suave
- **Cuándo usar**: Cuando necesitas máxima suavidad
- **Limitaciones**: Puede producir sobrepaso, más lento

#### RBF (Radial Basis Function)
- **Ventajas**: Excelente para datos dispersos
- **Cuándo usar**: Pocos puntos (<50), distribución irregular
- **Limitaciones**: Lento con muchos puntos, puede sobrepasar

#### IDW (Inverse Distance Weighting)
- **Ventajas**: Simple, rápido, sin sobrepaso
- **Cuándo usar**: Datos con tendencias locales fuertes
- **Limitaciones**: Puede crear "ojos de buey" alrededor de puntos

### Enmascaramiento

#### Sin máscara
- Muestra interpolación en toda la grilla
- Útil para: exploración inicial

#### ConvexHull (recomendado)
- Enmascara fuera del polígono convexo de los datos
- Útil para: mayoría de casos, evita extrapolación obvia
- Requiere: al menos 4 puntos no colineales

#### Por distancia
- Enmascara celdas lejanas a puntos de datos
- Útil para: datasets con huecos internos
- Parámetro: distancia máxima (0 = automático)

#### Ambos combinados
- Aplica ambas máscaras simultáneamente
- Útil para: máxima confiabilidad
- Más conservador: descarta más celdas

## Interpretación de Resultados

### Colores
- Rojo/Amarillo: Valores altos
- Verde/Azul: Valores medios
- Azul oscuro/Violeta: Valores bajos

### Puntos blancos
- Ubicaciones de tus datos originales
- Usa para verificar que la interpolación tiene sentido

### Áreas grises/blancas
- Zonas enmascaradas (sin datos confiables)
- No interprete valores en estas áreas

### Líneas de contorno negras
- Isolíneas de valores constantes
- Útiles para identificar gradientes

## Solución de Problemas

### "Insufficient points"
**Causa**: Menos de 3 puntos válidos  
**Solución**: 
- Verifica que X, Y y parámetros sean columnas numéricas
- Revisa si hay filas con valores faltantes
- Asegúrate de haber seleccionado las columnas correctas

### Error en ConvexHull
**Causa**: Puntos colineales o muy pocos puntos  
**Solución**: 
- Cambia a máscara "Por distancia"
- Verifica que tengas al menos 4 puntos no alineados

### Toda la grilla es NaN
**Causa**: Máscara muy restrictiva  
**Solución**: 
- Cambia a "Sin máscara" temporalmente
- Si usas máscara por distancia, aumenta distancia máxima
- Verifica distribución espacial de tus puntos

### Interpolación produce patrones extraños
**Causa**: Outliers o método inadecuado  
**Solución**: 
- Revisa tus datos por valores atípicos
- Prueba método IDW (más conservador)
- Aumenta resolución de grilla
- Verifica que X, Y estén correctos

### La aplicación es muy lenta
**Causa**: Resolución alta o RBF con muchos puntos  
**Solución**: 
- Reduce resolución a 50×50 para pruebas
- Usa Griddata en lugar de RBF
- Considera submuestrear datos si tienes >5000 puntos

## Consejos y Mejores Prácticas

### Preparación de Datos

1. **Limpia tus datos antes de importar**
   - Elimina filas duplicadas
   - Verifica valores atípicos
   - Asegura consistencia de unidades

2. **Usa nombres de columnas descriptivos**
   - Preferible: "resistencia_punta_qc" vs "col1"
   - Evita caracteres especiales y tildes en nombres

3. **Incluye columna ID si es posible**
   - Facilita identificación de puntos
   - Útil para troubleshooting

### Flujo de Trabajo Recomendado

1. **Primera iteración**: Griddata Linear + ConvexHull + 100×100
2. **Revisar**: ¿Los resultados tienen sentido?
3. **Ajustar**: Si es necesario, probar otros métodos
4. **Finalizar**: Aumentar resolución para figura final (200×200)
5. **Exportar**: Descargar PNG y CSV

### Validación de Resultados

1. **Verificar rangos**
   - ¿Min/max coinciden con datos originales?
   - ¿Hay valores fuera de rango esperado?

2. **Inspeccionar puntos**
   - ¿La interpolación pasa por/cerca de puntos reales?
   - ¿Hay gradientes suaves entre puntos?

3. **Revisar zonas enmascaradas**
   - ¿Tiene sentido la extensión de la máscara?
   - ¿Hay datos suficientes en toda el área?

## Ejemplos de Uso

### Ejemplo 1: Mapa de resistencia por punta (CPT)

```csv
sondeo,este,norte,qc,fs
CPT-01,100.0,200.0,2.5,50
CPT-01,100.0,200.0,3.2,55
CPT-02,150.0,205.0,2.8,52
...
```

Configuración:
- X: este
- Y: norte  
- Parámetro: qc
- Método: Griddata Linear
- Máscara: ConvexHull

### Ejemplo 2: Perfil de propiedades del suelo

```csv
id,abscisa,cota,gamma,LL,IP
S-01,0,50.5,18.5,35,12
S-01,0,48.0,19.1,38,14
S-02,10,51.2,18.7,36,13
...
```

Configuración:
- X: abscisa
- Y: cota
- Parámetros: gamma, LL, IP (todos)
- Método: IDW power=2.0
- Máscara: Ambos combinados
- Invertir eje Y: ✓ (para mostrar profundidad hacia abajo)

### Ejemplo 3: Mapa de nivel freático

```csv
pozo,x,y,nivel_freatico
PZ-01,100,200,5.2
PZ-02,150,210,4.8
PZ-03,200,190,6.1
...
```

Configuración:
- X: x
- Y: y
- Parámetro: nivel_freatico
- Método: RBF thin_plate
- Máscara: Por distancia (automático)

## Referencia Rápida

### Atajos de Teclado (Streamlit)
- `r`: Recargar aplicación
- `Ctrl+K`: Abrir comando rápido

### Formatos de Archivo Soportados
- CSV (`,` o `;` como separador)
- Excel 97-2003 (.xls)
- Excel 2007+ (.xlsx)

### Límites Prácticos
- Máximo puntos recomendado: 10,000
- Máximo resolución: 500×500
- Tamaño archivo: <200 MB

## Recursos Adicionales

- **Documentación scipy.interpolate**: https://docs.scipy.org/doc/scipy/reference/interpolate.html
- **Tutorial ConvexHull**: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.ConvexHull.html
- **Guía Streamlit**: https://docs.streamlit.io/

## Soporte

Si encuentras problemas:
1. Revisa esta guía de solución de problemas
2. Verifica el README.md del proyecto
3. Abre un issue en GitHub con:
   - Descripción del problema
   - Captura de pantalla
   - Muestra de tus datos (sin información sensible)
   - Configuración usada

---

**Versión**: 1.0  
**Última actualización**: Noviembre 2024
