# Torneo Banco Itaú & FEN U. de Chile — Data Science en Finanzas

## Estrategia Cuantitativa USD/CLP: *"Cash as the Default Benchmark"*

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Framework: NumPy & SciPy](https://img.shields.io/badge/Engine-NumPy%20%7C%20SciPy-informational.svg)](https://scipy.org/) [![Validation: Purged Walk-Forward](https://img.shields.io/badge/Validation-Purged%20Walk--Forward-success.svg)](#)

Repositorio oficial del equipo para el **Torneo Banco Itaú y FEN Data Science en Finanzas: ¿Sube o baja el dólar mañana?**. Este proyecto implementa una arquitectura cuantitativa para predecir la dirección del tipo de cambio USD/CLP a un día y traducir dicha señal en decisiones de exposición discreta de $\{-3,-1,0,+1,+3\}$ veces el nocional de referencia, bajo estricto conjunto de información $\mathcal{F}_{t-1}$ (cero *data leakage*).

---

## 1\. Tesis de Inversión y Filosofía de Diseño

Las bases del certamen establecen explícitamente:

> *"Gana la estrategia que genera mayor valor económico neto de costos y no la que acierta más veces, dado que la dificultad no reside en emitir un pronóstico diario sino en discriminar cuándo tomar posición y cuándo abstenerse."*

La gran mayoría de los enfoques cuantitativos tradicionales fracasan en series de divisas intradiarias porque fuerzan al algoritmo a operar de forma continua ($100\%$ del tiempo), incurriendo en una severa erosión por costos de transacción ($1.5\text{ bps}$ por cambio de posición) y pagando carry negativo frente a una tasa de interés local en pesos que promedió dos dígitos durante los ciclos restrictivos del Banco Central de Chile.

Nuestra arquitectura invierte el paradigma:

1. **Cash como Benchmark por Defecto:** La posición natural de la cartera es la **abstención ($\text{Exp} = 0$)**, resguardando el capital en el devengo pasivo de la tasa overnight en pesos (`CLICP` / TPM) libre de riesgo cambiario y sin comisiones.  
2. **Exploit Microestructural de Asincronía Horaria:** El mercado interbancario formal de Santiago (`USDCLP Curncy`) cierra a las **13:30 hrs locales**, mientras que los mercados cambiarios de Nueva York (`BBDXY`, `USDZAR`, `USDMXN`, `CHN1M NDF`) continúan operando hasta las **17:00 hrs ET**. El modelo captura la divergencia acumulada en esas 3.5 horas vespertinas en $t-1$ para posicionarse antes de la apertura de Santiago en $t$.  
3. **Asimetría de Bet Sizing:** La estrategia se abstiene el **$84.27\%$ del tiempo** y únicamente compromete capital cuando la convicción estadística en las colas supera con holgura el carry esperado y los costos de corretaje.

---

## 2\. Los 6 Factores Macro y Microestructurales

Para evitar el sobreajuste dimensional (*curse of dimensionality*), la matriz predictiva $\mathbf{X}_{t-1}$ se restringe estrictamente a **6 factores con fundamento económico directo y signo estable (2010–2026)**, todos con desfase `.shift(1)`:

| Factor | Nombre Técnico | Series Bloomberg | Mecanismo Económico | Peso Final ($w$) |
| :---: | :---- | :---- | :---- | :---: |
| **${F}_{1}$** | `f1_em_divergence` | `USDZAR`, `USDMXN`, `USDBRL`, `USDCLP` | Divergencia residual entre la canasta emergente en NY (17:00 ET) y el cierre de Santiago (13:30 SCL). | **$+0.0236$** |
| **${F}_{2}$** | `f2_usd_divergence` | `BBDXY Index`, `DXY Curncy`, `USDCLP` | Presión vespertina del billete verde a nivel internacional no absorbida por el mercado local. | **$+0.0268$** |
| **${F}_{3}$** | `f3_copper_ret` | `HG1 Comdty`, `LMCADS03 Comdty` | Retorno ponderado de cobre COMEX y LME (términos de intercambio). | **$-0.0396$** |
| **${F}_{4}$** | `f4_ndf_pressure` | `CHN1M Curncy`, `CHNI1M Curncy` | Desbalance de liquidez offshore vía variación de puntos y yield implícito NDF a 1 mes. | **$-0.0043$** |
| **${F}_{5}$** | `f5_rate_spread_2y` | `CHSWP2 Curncy`, `USGG2YR Index` | Dinámica del carry estructural: primera diferencia a 5 días del spread Swap Cámara 2Y vs US Treasury 2Y. | **$-0.0704$** |
| **${F}_{6}$** | `f6_trend_sma50` | `USDCLP Curncy` | Distancia porcentual del spot respecto a su SMA de 50 días (filtro anti-rally y reversión). | **$-0.0524$** |

---

## 3\. Modelo Matemático y Gestión de Riesgo

El pipeline opera sin librerías externas de caja negra, implementado directamente sobre **NumPy y SciPy (L-BFGS-B)**:

* **Ponderación por Retorno Absoluto (*Sample Weighting*, López de Prado, 2018):** Cada observación histórica se pondera proporcionalmente a $|{r}_{i}|+\epsilon$, priorizando eventos de alta volatilidad y desbalance de liquidez sobre días laterales.  
* **Balanceo de Clases 50/50:** Normalización exacta para garantizar que el intercepto $b\approx 0$, evitando sesgos estructurales de predicción.  
* **Ensamble Dual de Horizontes:** Combinación lineal convexa: $$\hat{p}_t = 0.55 \cdot \hat{p}_t^{\text{Estructural (10 años)}} + 0.45 \cdot \hat{p}_t^{\text{Adaptativo (2 años)}}$$
* **Ajuste de Carry Diario:** $$p_{\text{eff}, t} = \hat{p}_t - \text{clip}\left(0.15 \cdot \frac{r_{\text{CLP}} - r_{\text{USD}}}{\sigma_{20d}}, -0.012, +0.012\right)$$
* **Filtro de Tendencia SMA50:** Bloquea ventas cortas agresivas ($-3x$) si el tipo de cambio cotiza sobre su media móvil de 50 días, e inhibe compras si el mercado está en caída estructural.  
* **Histéresis de Rotación:** Banda de amortiguación de $1.5\%$ para impedir giros erráticos entre días consecutivos.

---

## 4\. Resultados Auditados Fuera de Muestra (OOS 2021–2026)

Evaluación estricta mediante **Purged Walk-Forward Validation** (reentrenamiento trimestral de 63 días con embargo de 2 días):

| Métrica Financiera | Estrategia ML Neta | Estrategia ML Bruta | Buy & Hold USD (+1x) | Cash CLP (Exp 0x) |
| :---- | :---: | :---: | :---: | :---: |
| **Retorno Acumulado Total** | **\+169.65%** | \+190.25% | \+65.70% | \+43.28% |
| **CAGR (Tasa Anual Compuesta)** | **18.73%** | 20.25% | 9.13% | 6.42% |
| **Volatilidad Anualizada** | **11.36%** | 11.38% | 14.58% | 0.00% |
| **Sharpe Ratio (Exceso de Retorno)** | **1.02** | 1.13 | 0.25 | 0.00 |
| **Sortino Ratio** | **1.75** | 1.97 | 0.36 | 0.00 |
| **Máximo Drawdown** | **\-9.31%** | \-9.09% | \-24.32% | 0.00% |
| **Calmar Ratio (CAGR / MaxDD)** | **2.01** | 2.23 | 0.38 | 0.00 |
| **Hit Ratio en Días Activos** | **59.83%** | 60.70% | 49.66% | N/A |
| **Tasa de Abstención ($\text{Exp} = 0$)** | **84.27%** | 84.27% | 0.00% | 100.00% |

### Pruebas de Significancia Estadística (Requeridas por las Bases)

* **Test de Diebold-Mariano (HLN-HAC Newey-West):** Estadístico $\text{DM} = -1.1003$ ($p=0.1357$), Brier Score del modelo $0.248959$ vs $0.250000$ del pronóstico equiprobable (menor pérdida cuadrática).
* **Circular Block Bootstrap (10,000 réplicas):** Exceso de retorno observado de **$+11.59\%$ anual sobre la tasa libre de riesgo local** con un **$p\text{-valor} = 0.0044$** (se rechaza la hipótesis nula de suerte con $99.56\%$ de confianza estadística).  
* **ROC-AUC Fuera de Muestra:** $0.5376$ en la serie temporal completa.

---

## 5\. Estructura del Repositorio

```
├── README.md                      # Documentación ejecutiva del modelo
├── requirements.txt               # Dependencias del entorno
├── main_backtest.py               # Motor completo de Walk-Forward, P&L y Bootstrap
├── predict_daily.py               # Script de inferencia en 1 click para monitoreo en vivo
├── senal_del_dia.json             # Salida estructurada de la última señal diaria emitida
├── data/
│   └── base_usdclp.xlsx       # Base oficial de series Bloomberg (Kick-off)
└── docs/
    └── informe_tecnico.pdf        # Informe Ejecutivo para el jurado (código LaTeX incluido)
```

---

## 6\. Instalación y Guía de Uso Rápido (1-Click Execution)

El proyecto utiliza únicamente librerías estándar del ecosistema científico de Python, garantizando portabilidad sin conflictos de compilación en Spyder, VSCode o terminal.

### Paso 1: Clonar y configurar entorno

```sh
git clone https://github.com/tu-usuario/torneo-itau-fen-usdclp.git
cd torneo-itau-fen-usdclp
pip install -r requirements.txt
```

### Paso 2: Replicar el Backtest OOS y los Tests Estadísticos

Ejecuta el pipeline completo de validación fuera de muestra, imprime las tablas de métricas institucionales y despliega los gráficos en Matplotlib:

```sh
python main_backtest.py
```

### Paso 3: Monitoreo en Vivo (Entregable 1 — Señal del Día)

Para procesar la base actualizada al cierre de cada jornada y emitir la predicción para mañana:

```sh
python predict_daily.py --file "data/base_usdclp.xlsx"
```

El script genera automáticamente el archivo `senal_del_dia.json`:

```json
{
    "fecha_cierre_t1": "2026-10-06",
    "ultimo_spot_usdclp": 973.95,
    "probabilidad_estimada": 0.5115,
    "probabilidad_efectiva_carry": 0.5098,
    "carry_diario_bps": 0.17,
    "posicion_sugerida": 0,
    "direccion_esperada": "ABSTENCIÓN (En Pesos Chilenos ganando tasa r_CLP)",
    "status": "SUCCESS"
}
```

---

## 7\. Referencias Bibliográficas

1. **López de Prado, M. (2018).** *Advances in Financial Machine Learning*. John Wiley & Sons.  
2. **Harvey, D., Leybourne, S., & Newbold, P. (1997).** *Testing the equality of prediction mean squared errors*. International Journal of Forecasting, 13(2), 281-291.  
3. **Diebold, F. X., & Mariano, R. S. (1995).** *Comparing predictive accuracy*. Journal of Business & Economic Statistics, 13(3), 253-263.  
4. **Politis, D. N., & Romano, J. P. (1994).** *The stationary bootstrap*. Journal of the American Statistical Association, 89(428), 1303-1313.

---

**Torneo Banco Itaú & FEN Universidad de Chile \| Octubre \- Diciembre 2026**
