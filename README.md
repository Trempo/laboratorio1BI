# Laboratorio 1 - Clasificación

### Elaborado por Camilo Salinas, Nicolas Orjuela y Felipe Bedoya.

## Descripción y análisis de perfilamiento de los datos y de las tareas sugeridas de transformación.

Para obtener una visión del estado actual de la calidad y la distribución de los datos se importo el CSV a PowerBI para poder ver estas métricas. Desde el Power Query Editor, y desde el contexto del Codebook pudimos ver que los datos son de tipo numérico. Aún así, algunas columnas tienen ciertos datos que no pueden ser procesados, como datos vacíos o datos no numéricos.

![Perfilamiento de datos PowerBI](./img/perfilamiento_powerbi.png)

Dentro del PowerQuery pudimos ver que debido a que al final de las lineas del CSV existe una repetición del carácter de terminación, existen 5 columnas vaciás. Se recomienda eliminarlas.

Directamente en el Notebook pudimos ver un poco mejor estas métricas para poder decidir que era lo mas pertinente.

![Heatmap de datos Notebook](./img/heatmap_isna.png)

Aquí podemos ver que los datos NaN (es decir, los vacíos o caracteres que no pueden ser procesados) no representan un porcentaje importante en los datos y se sugiere como primera tarea de transformación eliminarlos.

Podemos en el Notebook ver una gráfica de barras que nos indica el balanceo de los datos, contabilizando cuantos datos hay para cada una de las etiquetas del dataset.

![Desbalanceo de datos Notebook](./img/desbalanceo.png)

Como parte de la utilidad entregada a SaludAlpes, se incluye un tablero de PowerBI en donde esta la caracterización de los datos y un resumen del comportamiento, datos vacíos y relación entre estos.

![Tablero PowerBI](./img/tablero.png)

El archivo del tablero es **tablero LAB1.pbix** en el repositorio.

Empezar a ver relaciones de los datos es importante para poder escoger correctamente los features del modelo. Se recomienda un análisis mas avanzado de correlación y una limpieza de columnas.

Finalmente se recomienda hacer una estandarización o normalización de los datos (dependiendo de a cual beneficie el modelo) y convertir los datos al tipo más pequeño para reducir la complejidad computacional de los modelos.
