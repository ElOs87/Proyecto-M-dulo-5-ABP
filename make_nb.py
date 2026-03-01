import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = [
    nbf.v4.new_markdown_cell('# PROYECTO ABP MÓDULO 4: ANÁLISIS EXPLORATORIO DE DATOS - COMERCIOYA\n**Autor:** Osvaldo [Ingeniero Comercial UBB 2011, Mag. Ing. Industrial U. Central]\n**Fecha:** 18-02-2026 | Dataset: Ventas e-commerce simuladas (500 registros)'),
    nbf.v4.new_code_cell('''import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from matplotlib.ticker import FuncFormatter

plt.style.use('seaborn-v0_8')
colores_corp = ['#FF6B6B', '#1f77b4', '#2ca02c']
sns.set_palette(colores_corp)

def formato_millones(x, pos):
    if x >= 1e6:
        return f'${x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'${x*1e-3:.0f}K'
    else:
        return f'${x:.0f}'

formatter_millones = FuncFormatter(formato_millones)
print("Librerías cargadas exitosamente.")'''),
    nbf.v4.new_markdown_cell('## LECCIÓN 1: ANÁLISIS EXPLORATORIO INICIAL (EDA BÁSICO)'),
    nbf.v4.new_code_cell('''np.random.seed(42)
n = 500
zonas = np.random.choice(['Norte', 'Centro', 'Sur'], n, p=[0.3, 0.4, 0.3])
montos = np.random.lognormal(7, 1.5, n).round(0) * 1000
visitas = np.random.poisson(5, n)
devoluciones = np.random.choice([0, 1], n, p=[0.85, 0.15])
resenas = np.random.normal(4.2, 0.8, n).clip(1, 5).round(1)

df = pd.DataFrame({
'ID_Cliente': range(1, n+1),
'Zona': zonas,
'Monto': montos,
'Visitas': visitas,
'Devolucion': devoluciones,
'Resena': resenas
})
print(df.head())
df.to_csv('dataset_comercioya.csv', index=False)'''),
    nbf.v4.new_markdown_cell('## LECCIÓN 2: ESTADÍSTICA DESCRIPTIVA'),
    nbf.v4.new_code_cell('''num_cols = ['Monto', 'Visitas', 'Resena']
print(df[num_cols].describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, col in enumerate(num_cols):
    sns.boxplot(data=df, y=col, ax=axes[i], color=colores_corp[0], width=0.4)
    axes[i].set_title(f'Boxplot {col}')
    sns.despine(ax=axes[i])
    if col == 'Monto':
        axes[i].yaxis.set_major_formatter(formatter_millones)

plt.tight_layout()
plt.show()'''),
    nbf.v4.new_markdown_cell('## LECCIÓN 3: CORRELACIONES'),
    nbf.v4.new_code_cell('''corr_matrix = df[['Monto', 'Visitas', 'Resena', 'Devolucion']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Blues', center=0, square=True)
plt.title('Matriz de Correlación - Variables Numéricas')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Visitas', y='Monto', hue='Zona', alpha=0.7, palette=colores_corp)
plt.title('Relación Visitas vs Monto por Zona')
plt.gca().yaxis.set_major_formatter(formatter_millones)
sns.despine()
plt.show()'''),
    nbf.v4.new_markdown_cell('## LECCIÓN 4: REGRESIONES LINEALES'),
    nbf.v4.new_code_cell('''X = df[['Visitas', 'Resena', 'Devolucion']]
X = sm.add_constant(X)
y = df['Monto']
model = sm.OLS(y, X).fit()
print(model.summary())

plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Visitas', y='Monto', scatter_kws={'alpha':0.5, 'color': colores_corp[1]}, line_kws={'color': colores_corp[0]})
plt.title(f'Regresión Monto ~ Visitas (R²={model.rsquared:.3f})')
plt.gca().yaxis.set_major_formatter(formatter_millones)
sns.despine()
plt.show()'''),
    nbf.v4.new_markdown_cell('## LECCIÓN 5: SEABORN AVANZADO'),
    nbf.v4.new_code_cell('''g = sns.pairplot(df[['Monto', 'Visitas', 'Resena', 'Zona']], hue='Zona', palette=colores_corp, plot_kws={'alpha':0.7})
for ax in g.axes.flatten():
    if ax.get_ylabel() == 'Monto':
        ax.yaxis.set_major_formatter(formatter_millones)
    if ax.get_xlabel() == 'Monto':
        ax.xaxis.set_major_formatter(formatter_millones)
plt.suptitle('Pairplot Completo por Zona', y=1.02)
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Zona', y='Monto', palette=colores_corp, inner='quartile')
plt.title('Distribución Monto por Zona (Violinplot)')
plt.gca().yaxis.set_major_formatter(formatter_millones)
sns.despine()
plt.show()'''),
    nbf.v4.new_markdown_cell('## LECCIÓN 6: MATPLOTLIB DASHBOARD'),
    nbf.v4.new_code_cell('''fig, axes = plt.subplots(2, 2, figsize=(15, 12), layout='constrained')

axes[0,0].hist(df['Monto'], bins=30, alpha=0.8, color=colores_corp[1], edgecolor='white')
axes[0,0].set_title('Distribución Monto (Histograma)', fontsize=14)
axes[0,0].xaxis.set_major_formatter(formatter_millones)
sns.despine(ax=axes[0,0])

zona_ventas = df.groupby('Zona')['Monto'].sum()
barras = axes[0,1].bar(zona_ventas.index, zona_ventas.values, color=colores_corp, edgecolor='white')
axes[0,1].set_title('Ventas Totales por Zona', fontsize=14)
axes[0,1].bar_label(barras, labels=[f'${v*1e-6:.1f}M' for v in zona_ventas.values], padding=3)
axes[0,1].yaxis.set_major_formatter(formatter_millones)
sns.despine(ax=axes[0,1])

sns.regplot(data=df, x='Resena', y='Monto', ax=axes[1,0], scatter_kws={'s':25, 'alpha': 0.6, 'color': colores_corp[1]}, line_kws={'color': colores_corp[0]})
axes[1,0].set_title('Monto vs Reseña', fontsize=14)
axes[1,0].yaxis.set_major_formatter(formatter_millones)
sns.despine(ax=axes[1,0])

dev_pie = df['Devolucion'].value_counts()
axes[1,1].pie(dev_pie.values, labels=['Sin Devolución', 'Con Devolución'], autopct='%1.1f%%', colors=[colores_corp[2], colores_corp[0]], startangle=90)
axes[1,1].set_title('Tasa Devoluciones', fontsize=14)

fig.suptitle('DASHBOARD EDA COMERCIOYA', fontsize=18, fontweight='bold')
plt.show()''')
]

nb['cells'] = cells
with open('Cuaderno_ComercioYA.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
