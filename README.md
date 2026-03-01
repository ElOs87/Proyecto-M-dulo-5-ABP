README.md - Informe Técnico: Análisis Exploratorio ComercioYA
Fundamentación Técnica
El presente proyecto ejecuta un Análisis Exploratorio de Datos (EDA) sobre el flujo transaccional simulado del e-commerce ComercioYA. El objetivo principal consiste en identificar las variables que determinan el volumen de ventas y estructurar un modelo predictivo base. Implementé técnicas de evaluación paramétrica y visualización avanzada, asegurando el cumplimiento de los supuestos estadísticos antes de formular recomendaciones operativas para la gerencia.

Hallazgos Estratégicos y Modelado
El escaneo inicial de las distribuciones evidencia una alta asimetría en el monto de ventas. La evaluación mediante diagramas de caja aísla valores atípicos superiores, los cuales representan transacciones de alto volumen que exigen un tratamiento diferenciado en el análisis del flujo de caja corporativo.

Al evaluar las dependencias matemáticas, la matriz de correlación confirma una asociación lineal positiva entre la cantidad de visitas al portal y el monto transado. El modelo de regresión múltiple ajustado mediante Mínimos Cuadrados Ordinarios (OLS) cuantifica y valida esta relación. El análisis demuestra que las visitas web impactan directamente en la conversión financiera, mientras que la calificación del usuario (reseñas) presenta un efecto marginal en el ticket de compra inmediato. Esto sugiere que la tracción de ingresos depende fuertemente del tráfico estructurado por sobre la percepción cualitativa a corto plazo.

Recomendaciones Directas
Con base en los resultados del modelo de regresión, sugiero reasignar partidas presupuestarias hacia la captación directa de tráfico, ya que el volumen de visitas es el predictor de ingresos más robusto. Asimismo, la tasa de devoluciones del 15% detectada exige constituir una provisión fiscal en el balance, reconociendo este pasivo contingente para proteger la liquidez y la rentabilidad neta de la operación comercial.

Referencias Bibliográficas
Alkemy. (2024). Manual #4: Regresiones lineales. Módulo 5: Análisis Exploratorio de Datos.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2023). An Introduction to Statistical Learning with Applications in Python. Springer.

Seaborn Developers. (2024). Visualizing statistical relationships. https://seaborn.pydata.org/tutorial/relational.html
