#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Función que genera una señal senoidal a partir de los parámetros recibidos
y luego calcula la DFT

Created on Mon Aug 26 19:27:10 2025

@author: Fernando Daniel Fiamberti
"""
import numpy as np
import matplotlib.pyplot as plt



"""
Parámetros de la señal senoidal:

      ax:     Amplitud máxima de la señal [V]
      
      dc:     Valor medio de la señal [V]
      
      fx:     Frecuencia de la señal [Hz]
      
      ph:     Fase de la señal [rad]
      
      N :     Número de muestras
      
      fs:     Frecuencia de muestreo del ADC [Hz]
      
      Ts:     Período de muestreo [s]       
      
      T_sim:  Tiempo que dura la simulación [s]


Devuelve:

      tt : Tiempo correspondiente a cada muestra
      
      xx : Señal senoidal generada
""" 





"""
Parámetros de la DFT:

       x: Señal muestreada 

Devuelve:
      
      X : DFT de la señal
""" 




# Función que genera una señal senoidal con los parámetros especificados:
def generador_senoidal(ax, dc, fx, ph, N, fs):
 
    # Genera el vector de tiempo
    tt = np.arange(start = 0, stop = T_sim, step = Ts)

    
    # Crea la señal senoidal
    xx = ax * np.sin(2 * np.pi * fx * tt + ph) + dc
    
    return tt, xx




# Función para calcular la DFT manualmente
def dft(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)  # Array para los coeficientes de frecuencia
    for k in range(N):
        # Cálculo de la DFT en el índice k
        X[k] = np.sum(x * np.exp(-2j * np.pi * k * np.arange(N) / N))
    return X







# Parámetros de la señal
ax = 2                # Amplitud    [v]
dc = 0                # Valor medio [v]
fx = 1                # Frecuencia  [Hz]
ph = 0                # Fase        [rad]


# Parámetros de muestreo
fs = 100              # Frec. de muestreo    [Hz]
N = fs                # Número de muestras
Ts = 1/fs             # Período de muestreo  [s]
T_sim = N * Ts        # Tiempo de simulación [s]




# Llamo a la función para generar la senoidal muestreada
tt, xx = generador_senoidal(ax, dc, fx, ph, N, fs)

# Calcular la DFT de la señal
X = dft(xx)


# Tomar solo la mitad positiva del espectro
half_N = N // 2                  # Solo tomo hasta la mitad de las frecuencias
X_pos = X[:half_N]               # Coeficientes del espectro positivo
magnitudes = np.abs(X_pos)       # Magnitud 

# Eje de frecuencias [muestras] (es un entero)
frequencias = np.arange(half_N)  # Eje de frecuencias



"""
plt.subplot(filas, columnas, ìndice)

    filas:    Número de filas

    columnas: Número de columnas

    ìndice: Índice del gráfico donde se debe colocar el gráfico actual 
            (los índices empiezan desde 1).
"""


# Grafica la señal generada
plt.subplot(1, 2, 1)
plt.plot( tt, xx, 'o--')
plt.title('Senoidal Generada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.grid(True)
plt.show()


# Grafica la magnitud de la DFT (solo espectro positivo)
plt.subplot(1, 2, 2)
# plt.figure(figsize=(10, 6))
plt.scatter(frequencias , magnitudes, color='g')  # Usamos scatter para no conectar los puntos
plt.title("Espectro (DFT)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.show()


"""
%matplotlib qt  +   ENTER     =>

permite graficar en una pantalla emergente
"""

