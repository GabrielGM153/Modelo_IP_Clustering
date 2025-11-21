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

### Evolución del PIB:
<img width="1172" height="526" alt="Evolución del PIB" src="https://github.com/user-attachments/assets/8dd97598-ac17-4688-badd-f31533a6027d" />

### Composición del PIB
<img width="1218" height="503" alt="composición del pib" src="https://github.com/user-attachments/assets/ef9d41bb-c747-4864-8a75-b7020104d3c9" />

### IGAE: Tasas de crecimiento anual total
<img width="1134" height="522" alt="igae_tasas_crec" src="https://github.com/user-attachments/assets/5e2ddd71-0060-4d5b-8fe6-68e6a0aecb55" />

### Tasa de desempleo
<img width="1147" height="513" alt="Captura de pantalla 2025-11-21 150723" src="https://github.com/user-attachments/assets/5caed3a4-6130-4e1c-b661-5773e8be263c" />

### El sector manufacturero (EAIM)

<img width="1238" height="639" alt="Captura de pantalla 2025-11-21 150857" src="https://github.com/user-attachments/assets/f60f8d00-e714-46b0-be2f-c6ea40113ca5" />


### Interfaz Interactiva:

- Desarrollada en Streamlit.

- Permite ajustar el número de clusters ($k$) en tiempo real para dinamizar el análisis.

