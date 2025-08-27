#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Función que genera una señal senoidal a partir de los parámetros recibidos

Created on Mon Aug 25 09:58:33 2025

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





# Función que genera una señal senoidal con los parámetros especificados:
def generador_senoidal(ax, dc, fx, ph, N, fs):
 
    # Genera el vector de tiempo
    tt = np.arange(start = 0, stop = T_sim, step = Ts)

    
    # Crea la señal senoidal
    xx = ax * np.sin(2 * np.pi * fx * tt + ph) + dc
    
    return tt, xx




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




# Llamo a la función
tt, xx = generador_senoidal(ax, dc, fx, ph, N, fs)



# Grafica la señal generada
plt.plot( tt, xx, '*--')
plt.title('Senoidal Generada')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud [V]')
plt.grid(True)
plt.show()


