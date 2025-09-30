# Ts3
### Tarea semanal 3 para la materia Procesamiento Digital de Señales de 6to año de Ingeniería Electrónica de la UTN FRBA


# Estimación de Amplitud y Frecuencia de una Señal

## Señal a generar

Se define la señal:


$$
x(n) = a_0 \cdot \sin(\Omega_1 \cdot n) + n_a(n)
$$


Donde:

<div align="left">
$$
\Omega_0 = \frac{\pi}{2}
$$
</div>

<div align="left">
$$
\Omega_1 = \Omega_0 + f_r \cdot \frac{2\pi}{N}
$$
</div>

<div align="left">
$$
f_r \sim \mathcal{U}(-2, 2)
$$
</div>

<div align="left">
$$
n_a(n) \sim \mathcal{N}(0, \sigma^2)
$$
</div>

## Estimadores a diseñar

### Estimador de amplitud

<div align="left">
$$
\hat{a}_1^i = \left| X_{iw}(\Omega_0) \right| = \left| \mathcal{F}\{x(n) \cdot w_i(n)\} \right|
$$
</div>

### Estimador de frecuencia

<div align="left">
$$
\hat{\Omega}_1^i = \underset{\Omega}{\mathrm{arg\,max}} \left\{ \left| X_{iw}(\Omega) \right| \right\}
$$
</div>

Donde \( w_i(n) \) es la ventana utilizada para ponderar la señal.

## Ventanas a utilizar

1. Rectangular (sin ventana)
2. Flattop (scipy.signal.windows.flattop)
3. Blackman-Harris (scipy.signal.windows.blackmanharris)
4. Otra a elección desde scipy.signal.windows

## Parámetros del experimento

- Cantidad de realizaciones: 200
- Muestras por señal: 1000
- SNRs a evaluar: 3 dB y 10 dB
- Potencia de la senoidal: 1 W (ajustar \( a_0 \) para que se cumpla)

## Métricas a calcular

### Estimación de amplitud

Sesgo:

<div align="left">
$$
s_a = \mathbb{E}[\hat{a}_0] - a_0 \approx \mu_{\hat{a}} - a_0
$$
</div>

Varianza:

<div align="left">
$$
v_a = \mathrm{Var}[\hat{a}_0] \approx \frac{1}{M} \sum_{j=0}^{M - 1} \left( \hat{a}_j - \mu_{\hat{a}} \right)^2
$$
</div>

### Estimación de frecuencia

Sesgo:

<div align="left">
$$
s_{\Omega} = \mathbb{E}[\hat{\Omega}_1] - \Omega_1 \approx \mu_{\hat{\Omega}} - \Omega_1
$$
</div>

Varianza:

<div align="left">
$$
v_{\Omega} = \mathrm{Var}[\hat{\Omega}_1] \approx \frac{1}{M} \sum_{j=0}^{M - 1} \left( \hat{\Omega}_j - \mu_{\hat{\Omega}} \right)^2
$$
</div>

## Resultados esperados

### Para cada SNR (3 dB y 10 dB)

#### Estimación de amplitud

| Ventana     | Sesgo ($s_a$) | Varianza ($v_a$) |
|-------------|----------------|------------------|
| Rectangular |                |                  |
| Flattop     |                |                  |
| Blackman    |                |                  |
| Otra        |                |                  |

#### Estimación de frecuencia

| Ventana     | Sesgo ($s_\Omega$) | Varianza ($v_\Omega$) |
|-------------|---------------------|------------------------|
| Rectangular |                     |                        |
| Flattop     |                     |                        |
| Blackman    |                     |                        |
| Otra        |                     |                        |

## Bonus

- Analizar el efecto del zero-padding en el estimador de frecuencia \( \hat{\Omega}_1 \).
- Proponer estimadores alternativos para amplitud y frecuencia, y repetir el experimento.
