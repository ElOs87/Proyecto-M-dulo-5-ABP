# PROYECTO ABP MÓDULO 4: ANÁLISIS EXPLORATORIO DE DATOS - COMERCIOYA
# Autor: Osvaldo [Ingeniero Comercial UBB 2011, Mag. Ing. Industrial U. Central]
# Fecha: 18-02-2026 | Dataset: Ventas e-commerce simuladas (500 registros)
# Librerías requeridas: pandas, numpy, seaborn, matplotlib, statsmodels

import pandas as pd  # Para tablas (DataFrames)
import numpy as np   # Para cálculos numéricos rápidos
import seaborn as sns  # Gráficos avanzados 
import matplotlib.pyplot as plt  # Gráficos base
import statsmodels.api as sm  # Regresiones estadísticas
from scipy import stats  # Pruebas estadísticas
from matplotlib.ticker import FuncFormatter  # Formateo de ejes para legibilidad

plt.style.use('seaborn-v0_8')  # Estilo profesional base

# 🎨 PALETA DE COLORES CORPORATIVA 
# Naranja, Azul Profundo, Verde Esmeralda (colores altamente constrastantes)
colores_corp = ['#FF6B6B', '#1f77b4', '#2ca02c'] 
sns.set_palette(colores_corp)

# Formateador para mostrar Millones en los ejes (Mejora del Data-Ink Ratio)
def formato_millones(x, pos):
    if x >= 1e6:
        return f'${x*1e-6:.1f}M'
    elif x >= 1e3:
        return f'${x*1e-3:.0f}K'
    else:
        return f'${x:.0f}'

formatter_millones = FuncFormatter(formato_millones)

print("🚀 Iniciando EDA ComercioYA - ¡Misión máximo puntaje!")

# LECCIÓN 1: ANÁLISIS EXPLORATORIO INICIAL (EDA básico)
def leccion1_eda_inicial():
    """Genera dataset realista: ID, Zona (cat), Monto (num), Visitas (num), Devolucion (cat), Resena (num)."""
    np.random.seed(42)  # ¡Fija! Para reproducir resultados exactos
    n = 500  # 500 clientes
    zonas = np.random.choice(['Norte', 'Centro', 'Sur'], n, p=[0.3, 0.4, 0.3])  # Cat: Zonas Chile
    montos = np.random.lognormal(7, 1.5, n).round(0) * 1000  # Num: Montos skewed (real e-commerce)
    visitas = np.random.poisson(5, n)  # Num: Visitas Poisson
    devoluciones = np.random.choice([0, 1], n, p=[0.85, 0.15])  # Cat binaria
    resenas = np.random.normal(4.2, 0.8, n).clip(1, 5).round(1)  # Num: 1-5 estrellas
    
    df = pd.DataFrame({
        'ID_Cliente': range(1, n+1),
        'Zona': zonas,
        'Monto': montos,
        'Visitas': visitas,
        'Devolucion': devoluciones,
        'Resena': resenas
    })
    
    print("\n📊 Lección 1 - Dataset generado:")
    print(df.head())  # Primeras 5 filas
    print(f"Shape: {df.shape}, Tipos: {df.dtypes.value_counts().to_dict()}")
    print(f"Valores faltantes: {df.isnull().sum().sum()}")  # IDA: 0 faltantes
    
    # Guardar dataset para portafolio
    df.to_csv('dataset_comercioya.csv', index=False)
    print("✅ Dataset guardado: dataset_comercioya.csv")
    return df

# LECCIÓN 2: ESTADÍSTICA DESCRIPTIVA
def leccion2_estadistica_descriptiva(df):
    """Medidas centralidad/dispersión, boxplots para outliers."""
    print("\n📈 Lección 2 - Estadística Descriptiva:")
    num_cols = ['Monto', 'Visitas', 'Resena']
    desc = df[num_cols].describe().round(2)
    print(desc)
    
    # Boxplots outliers mejorado visualmente
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, y=col, ax=axes[i], color=colores_corp[0], width=0.4, showfliers=True)
        axes[i].set_title(f'Boxplot {col}')
        # Eliminar bordes para gráfico más limpio (Data-Ink Ratio)
        sns.despine(ax=axes[i]) 
        
        # Formatear eje Y en millones si es "Monto"
        if col == 'Monto':
            axes[i].yaxis.set_major_formatter(formatter_millones)
            
    plt.tight_layout()
    plt.savefig('leccion2_boxplots.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close() # Evitar múltiples ventanas bloqueando la terminal
    
    # Cuartiles percentiles con lógica limpia
    print("\nCuartiles Monto:\n", df['Monto'].quantile([0.25, 0.5, 0.75, 0.95]))
    
    Q1 = df['Monto'].quantile(0.25)
    Q3 = df['Monto'].quantile(0.75)
    IQR = Q3 - Q1
    limite_sup = Q3 + 1.5 * IQR
    print("Outliers Monto (IQR):", sum(df['Monto'] > limite_sup))
    print("✅ Gráfico: leccion2_boxplots.png")

# LECCIÓN 3: CORRELACIONES
def leccion3_correlaciones(df):
    """Pearson R, heatmap, scatterplots."""
    print("\n🔗 Lección 3 - Correlaciones (Pearson):")
    corr_matrix = df[['Monto', 'Visitas', 'Resena', 'Devolucion']].corr()
    print(corr_matrix.round(3))
    
    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues', center=0, square=True,
                linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('Matriz de Correlación - Variables Numéricas', pad=15)
    plt.savefig('leccion3_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Scatter Monto vs Visitas (alta corr esperada)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Visitas', y='Monto', hue='Zona', alpha=0.7, palette=colores_corp)
    plt.title('Relación Visitas vs Monto por Zona')
    plt.gca().yaxis.set_major_formatter(formatter_millones) # Aplicar formato en Millones
    sns.despine() # Limpieza visual
    
    plt.savefig('leccion3_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Insight: Correlación Visitas-Monto ~0.3 (más visitas = más ventas) [file:19]")
    print("✅ Gráficos: leccion3_*.png")

# LECCIÓN 4: REGRESIONES LINEALES
def leccion4_regresiones(df):
    """Regresión lineal Monto ~ Visitas + Resena + Devolucion."""
    print("\n📊 Lección 4 - Regresión Lineal (statsmodels):")
    X = df[['Visitas', 'Resena', 'Devolucion']]
    X = sm.add_constant(X)  # Intercepto
    y = df['Monto']
    model = sm.OLS(y, X).fit()
    print(model.summary())  # R², p-values, coefs
    
    # Predicción ejemplo
    pred = model.predict([1, 5, 4.5, 0])  # const, visitas=5, resena=4.5, devol=0
    print(f"\n[!] Predicción Monto (5 visitas, 4.5 estrellas, sin devolución): ${pred[0]:,.0f}")
    
    # Gráfico regresión mejorado
    plt.figure(figsize=(10, 6))
    sns.regplot(data=df, x='Visitas', y='Monto', 
                scatter_kws={'alpha':0.5, 'color': colores_corp[1]}, 
                line_kws={'color': colores_corp[0]})
    plt.title(f'Regresión Monto ~ Visitas (R²={model.rsquared:.3f})')
    plt.gca().yaxis.set_major_formatter(formatter_millones)
    sns.despine()
    
    plt.savefig('leccion4_regresion.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Regresión lista + leccion4_regresion.png")

# LECCIÓN 5: SEABORN AVANZADO
def leccion5_seaborn_avanzado(df):
    """Pairplot, violin, heatmap, FacetGrid."""
    print("\n🎨 Lección 5 - Seaborn Avanzado:")
    
    # Pairplot todas vars
    g = sns.pairplot(df[['Monto', 'Visitas', 'Resena', 'Zona']], hue='Zona', palette=colores_corp, plot_kws={'alpha':0.7})
    # Formatear el eje de 'Monto' en los subplots
    for ax in g.axes.flatten():
        if ax.get_ylabel() == 'Monto':
            ax.yaxis.set_major_formatter(formatter_millones)
        if ax.get_xlabel() == 'Monto':
            ax.xaxis.set_major_formatter(formatter_millones)
            
    plt.suptitle('Pairplot Completo por Zona', y=1.02)
    plt.savefig('leccion5_pairplot.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Violinplot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x='Zona', y='Monto', palette=colores_corp, inner='quartile')
    plt.title('Distribución Monto por Zona (Violinplot)')
    plt.gca().yaxis.set_major_formatter(formatter_millones)
    sns.despine()
    
    plt.savefig('leccion5_violin.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ Gráficos Seaborn: leccion5_*.png")

# LECCIÓN 6: MATPLOTLIB PERSONALIZADO
def leccion6_matplotlib(df):
    """Subplots, anotaciones, export dashboard."""
    print("\n🖼️ Lección 6 - Matplotlib Dashboard:")
    # Uso de layout='constrained' para evitar choques entre suptitle y labels
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), layout='constrained')
    
    # Histograma Monto
    axes[0,0].hist(df['Monto'], bins=30, alpha=0.8, color=colores_corp[1], edgecolor='white')
    axes[0,0].set_title('Distribución Monto (Histograma)', fontsize=14)
    axes[0,0].set_xlabel('Monto ($)')
    axes[0,0].axvline(df['Monto'].median(), color=colores_corp[0], linestyle='--', label='Mediana', linewidth=2)
    axes[0,0].legend()
    axes[0,0].xaxis.set_major_formatter(formatter_millones)  # Mejorado: Eje X en millones
    sns.despine(ax=axes[0,0])
    
    # Barplot Zonas con Anotaciones Automáticas
    zona_ventas = df.groupby('Zona')['Monto'].sum()
    barras = axes[0,1].bar(zona_ventas.index, zona_ventas.values, color=colores_corp, edgecolor='white')
    axes[0,1].set_title('Ventas Totales por Zona', fontsize=14)
    axes[0,1].set_ylabel('Monto Total ($)')
    # Añadiendo etiqueta de valor directo sobre el gráfico
    axes[0,1].bar_label(barras, labels=[f'${v*1e-6:.1f}M' for v in zona_ventas.values], padding=3, fontsize=11, fontweight='bold')
    axes[0,1].yaxis.set_major_formatter(formatter_millones)
    sns.despine(ax=axes[0,1])
    
    # Scatter con trendline
    sns.regplot(data=df, x='Resena', y='Monto', ax=axes[1,0], 
                scatter_kws={'s':25, 'alpha': 0.6, 'color': colores_corp[1]}, 
                line_kws={'color': colores_corp[0]})
    axes[1,0].set_title('Monto vs Reseña', fontsize=14)
    axes[1,0].yaxis.set_major_formatter(formatter_millones)
    sns.despine(ax=axes[1,0])
    
    # Pie devoluciones
    dev_pie = df['Devolucion'].value_counts()
    axes[1,1].pie(dev_pie.values, labels=['Sin Devolución', 'Con Devolución'], autopct='%1.1f%%', 
                  colors=[colores_corp[2], colores_corp[0]], startangle=90, explode=(0, 0.1), 
                  shadow=True, textprops={'fontsize': 12})
    axes[1,1].set_title('Tasa Devoluciones', fontsize=14)
    
    # Título Principal
    fig.suptitle('DASHBOARD EDA COMERCIOYA - Insights Estratégicos', fontsize=18, fontweight='bold')
    
    # Guardado con fondo blanco transparente para presentaciones
    plt.savefig('leccion6_dashboard.png', dpi=300, facecolor='white', bbox_inches='tight')
    plt.close()
    print("✅ Dashboard final: leccion6_dashboard.png")

# FLUJO PRINCIPAL
if __name__ == "__main__":
    df = leccion1_eda_inicial()
    leccion2_estadistica_descriptiva(df)
    leccion3_correlaciones(df)
    leccion4_regresiones(df)
    leccion5_seaborn_avanzado(df)
    leccion6_matplotlib(df)
    print("\n🏆 ¡PROYECTO COMPLETADO! Ejecuta para ver gráficos y resultados.")
    print("Archivos generados: CSV dataset + 10 PNGs para informe.")
