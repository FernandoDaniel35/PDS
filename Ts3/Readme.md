# Ts3
### Tarea semanal 3 para la materia Procesamiento Digital de Señales de 6to año de Ingeniería Electrónica de la UTN FRBA


# Estimación de Amplitud y Frecuencia de una Señal
# Consignas para la generación de la señal

Comenzaremos con la generación de la siguiente señal:

$x(n) = a_0 \cdot \sin(\Omega_1 \cdot n) + n_a(n)$

siendo

$\Omega_1 = \Omega_0 + f_r \cdot \frac{2 \pi}{N}$

$\Omega_0 = \frac{\pi}{2}$

siendo la variable aleatoria definida por la siguiente distribución de probabilidad

$f_r \sim U(-2, 2)$

$n_a \sim N(0, \sigma^2)$

---

# Diseño de estimadores

Diseñe los siguientes estimadores de amplitud $a_1$:

$\hat{a}_1^i = |X_i^w(\Omega_0)| = \left| \mathcal{F} \{ x(n) \cdot w_i(n) \} \right|$

para la i-ésima realización y la w-ésima ventana (ver detalles debajo).

Y de frecuencia  
$\Omega_1$:

$\hat{\Omega}_1^i = \arg \max_{\Omega} \left| X_i^w(\Omega) \right|$

para cada una de las ventanas:


- rectangular (sin ventana)
- flattop
- blackmanharris
- otra que elija de scipy.signal.windows

---

# Consignas para la experimentación

- Considere 200 realizaciones (muestras tomadas de $f_r$) de 1000 muestras para cada experimento.
- Parametrice para SNR's de 3 y 10 dB (Ayuda: calibre $a_0$ para que la potencia de la senoidal sea 1 W).

---

# Se pide

1. Realizar una tabla por cada SNR, que describa el sesgo y la varianza de cada estimador para cada ventana analizada. Recuerde incluir las ventanas rectangular (sin ventana), flattop y blackmanharris y otras que considere.

---

# Estimación de Amplitud

| Ventana      | Sesgo (sa) | Varianza (va) |
|--------------|------------|---------------|
| Rectangular  |            |               |
| Flat-top     |            |               |
| Blackman     |            |               |
| Otras        |            |               |

---

# Estimación de Frecuencia

| Ventana      | Sesgo (sa) | Varianza (va) |
|--------------|------------|---------------|
| Rectangular  |            |               |
| Flat-top     |            |               |
| Blackman     |            |               |
| Otras        |            |               |

---

# Ayuda para cálculo de sesgo y varianza

Puede calcular experimentalmente el sesgo y la varianza de un estimador:

$\hat{a}_0 = |X_i^w(\Omega_0)|$

siendo

$sa = E\{\hat{a}_0\} - a_0$

$va = \mathrm{var}\{\hat{a}_0\} = E \{ (\hat{a}_0 - E\{\hat{a}_0\})^2 \}$

y pueden aproximarse cuando consideramos los valores esperados como las medias muestrales:

$E\{\hat{a}_0\} = \mu_{\hat{a}} = \frac{1}{M} \sum_{j=0}^{M-1} \hat{a}_j$

$sa = \mu_{\hat{a}} - a_0$

$va = \frac{1}{M} \sum_{j=0}^{M-1} (a_j^ - \mu_{\hat{a}})^2$

---

# Bonus

- Analice el efecto del zero-padding para el estimador $\hat{\Omega}_1$.
- Proponga estimadores alternativos para frecuencia y amplitud de la senoidal y repita el experimento.
