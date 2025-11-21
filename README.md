# Modelo de Insumo Producto y Clustering
-----------------------------------------
[Acceso a la aplicación](https://modeloipclustering-datascienceandeconomics.streamlit.app/)

Este proyecto integra la Teoría Económica Estructural (Modelo Insumo-Producto de Leontief) con técnicas de Ciencia de Datos (Aprendizaje No Supervisado) para analizar la estructura productiva de la industria manufacturera.

Para su desarrollo, se toma como referencia principal (y también como marco teórico) el trabajo tanto empirico como teórico de Aroche (2019) y el fundamento de economía matemática realizado por Chiang y Wainwright (2006) 

En este contexto, Aroche (2019) comenta que Kaldor (1966) entiende al sector manufacturero como motor económico. En este sentido, el autor identifica el por qué el sector manufacturero es tan importante en términos de crecimiento económico; esto es, porque (y lo menciona de igual manera), este sector tiene gran importancia en la demanda de insumos, que posteriormente convierte en producto que expande la actividad económica dentro de otros sectores; es un impulso de crecimiento para los demás sectores.

En la lectura el modelo IP se refiere al modelo de insumo producto, de Leontief, mismo que describe cómo es que los sectores dentro de una economía están interconectados.

Siguiendo esta idea, el modelo IP sigue la premisa básica, que es paralela a lo que comenta Aroche (2019), con respecto al sector manufacturero, en el sentido de que cada sector produce bienes y servicios que sirven como insumos para otros sectores, esta dinámica tiene un comportamiento de oferta y demanda entre sectores.

Y siguiendo dicho concepto, con la que se trabaja en el artículo, el modelo IP lo identifica el autor como un modelo que sirve para la aplicación a corto plazo basado en la reproducción simple, en donde el análisis tiene una naturaleza matricial; en este modelo se dividen las matrices y vectores para encontrar el grado de dependencia de ciertos sectores  otros. De ahí es donde entra en acción práctica la ley de Kaldor-Verdroom, misma que se refiere a cómo es que una división entre sectores (siendo esta el sector manufacturero y los otros); se puede potenciar el crecimiento económico teniendo al sector manufacturero como base. 

Para Kaldor-Verdroon representa una expansión económica a gran escala.

Específicamente en relación con México, y siguiendo el análisis del autor, este tuvo ciertos problemas en cuanto al planteamiento del análisis, por lo que incluyó tres matrices:

•	Intercambios intermedios.

•	Importaciones intermedias.

•	Intercambios totales.

### Aproximación teórica y ejemplo:

https://github.com/user-attachments/assets/9a2c7aa0-a83e-40dd-87eb-5e726baffd9b

### Interfaz Interactiva:

- Desarrollada en Streamlit.

- Permite ajustar el número de clusters ($k$) en tiempo real para dinamizar el análisis.
