#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    1) Se genera una señal senoidal a partir de los parámetros recibidos.

    2) Se genera ruido uniforme y ruido gaussiano.

    3) Se suma el ruido a la señal senoidal generada.

    4) Se calcula la DFT de la señal senoidal + el ruido.
    

Created on Mon Sep 8 19:17:10 2025

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
      
       X : DFT de la señal, matriz (Nx1) de números complejos.
""" 




# --- Función senoidal ---
def generador_senoidal(ax, dc, fx, ph, N, fs):

    # Genera el vector de tiempo
    tt = np.arange(start = 0, stop = T_sim, step =  1/fs)

    # Crea la señal senoidal
    xx = ax * np.sin(2 * np.pi * fx * tt + ph) + dc

    return tt, xx






# --- Función para generar ruido ---
def generar_ruido(N, sigma_cuadrado=4, tipo="uniforme"):
    if tipo == "uniforme":
        delta = np.sqrt(12 * sigma_cuadrado)  # b - a
        a = -delta/2
        b = delta/2
        ruido = np.random.uniform(a, b, size=N)
    elif tipo == "gaussiano":
        sigma = np.sqrt(sigma_cuadrado)
        ruido = np.random.normal(0, sigma, size=N)
    else:
        raise ValueError("El tipo de ruido debe ser 'uniforme' o 'gaussiano'")
    return ruido








# --- Gráfico de la señal senoidal generada ---
def Senoidal_t(tt,xx):
    plt.figure()
#     plt.vlines(tt, ymin=0, ymax=xx, color='blue', label="Señal senoidal") # lìneas verticales 
    plt.plot(tt, xx, 'o', color='red', markersize=2, label="Señal")  # marcar la punta de cada palito
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.title("Señal senoidal")
    plt.grid(True)
    plt.legend()
    plt.show()




# --- Gráfico del ruido generado ---
def ruido_t(tt,ruido,tipo_ruido):
    plt.figure()
#     plt.vlines(tt, ymin=0, ymax=ruido, color='blue', label="Ruido") # lìneas verticales
    plt.plot(tt, ruido, 'o', color='red', markersize=2, label="Ruido")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.title(f"Ruido {tipo_ruido}")
    plt.grid(True)
    plt.legend()
    plt.show()


# --- Gráfico de la señal senoidal generada con el ruido sumado ---
def Senoidal_y_ruido_t(tt, senal_ruidosa):
    plt.figure()
#     plt.vlines(tt, ymin=0, ymax=senal_ruidosa, color='blue', label="Señal + Ruido") # lìneas verticales
    plt.plot(tt, senal_ruidosa, 'o', color='red', markersize=2,label="Señal + Ruido")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Amplitud")
    plt.title(f"Señal con Ruido {tipo_ruido}")
    plt.grid(True)
    plt.legend()
    plt.show()





# --- Grafico la DFT de la señal senoidal generada (sin ruido) ---
def dft_Señal(f_sen, S_sen):
    plt.figure()
#     plt.vlines(f_sen, ymin=0, ymax=S_sen, color='blue', label="Señal senoidal") # lìneas verticales
    plt.plot(f_sen, S_sen, 'o', color='red', markersize=2, label="Señal")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X(f)|")
    plt.title("DFT Señal senoidal")
    plt.grid(True)
    plt.legend()
    plt.show()





# --- Grafico la DFT y la FFT de la señal senoidal generada en el mismo gráfico ---
def DFT_y_FFT(f_sen, S_sen):
    plt.figure()
    plt.plot(f_sen, S_sen, 'o', color='red', markersize=2, label="DFT")
    plt.title("Comparación DFT vs FFT")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X[k]|")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure()
    plt.plot(freqs_fft[:N//2],np.abs(X_fft[:N//2]), 'o', color='red', markersize=2, label="FFT")
    plt.title("Comparación DFT vs FFT")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X[k]|")
    plt.legend()
    plt.grid()
    plt.show()






# --- Grafico la DFT de la señal senoidal generada [dB] (sin ruido) ---
def dft_Señal_dB(f_sen, S_sen):
    plt.figure()
#    plt.vlines(f_sen, ymin=0, ymax= to_dB(S_sen), color='blue', label="Señal senoidal") # lìneas verticales
    plt.plot(f_sen, to_dB(S_sen), 'o', color='red', markersize=2, label="Señal")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud [dB]")
    plt.title("DFT Señal senoidal [dB]")
    plt.grid(True)
    plt.legend()
    plt.show()




# --- Grafico la DFT de la señal + ruido ---
def dft_Señal_y_ruido(f_sum, S_sum):
    plt.figure()
#    plt.vlines(f_sum, ymin=0, ymax= S_sum, color='blue', label="DFT Señal + Ruido ") # lìneas verticales
    plt.plot(f_sum, S_sum, 'o', color='red', markersize=2, label="Señal + Ruido")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("|X(f)|")
    plt.title("DFT Señal + Ruido")
    plt.grid(True)
    plt.legend()
    plt.show()


# --- Grafico la DFT de la señal + ruido [dB] ---
def dft_Señal_y_ruido_dB(f_sum, S_sum):
    plt.figure()
 #    plt.vlines(f_sum, ymin=0, ymax= to_dB(S_sum), color='blue', label="DFT Señal + Ruido [dB]") # lìneas verticales
    plt.plot(f_sum, to_dB(S_sum), 'o', color='red', markersize=2, label="Señal + Ruido")
    plt.xlabel("Frecuencia [Hz]")
    plt.ylabel("Magnitud [dB]")
    plt.title("DFT Señal + Ruido [dB]")
    plt.grid(True)
    plt.legend()
    plt.show()




# --- DFT ---
def dft(x, fs):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.exp(-2j * np.pi * k * n / N)
    X = np.dot(W, x)
    X = np.abs(X) / N  # magnitud normalizada
    f = np.arange(N) * fs / N
    mitad = N // 2
    return f[:mitad], X[:mitad]



# --- Conversión a dB ---
def to_dB(X, eps=1e-12):
    return 20 * np.log10(X + eps)






# --- Parámetros de la señal ---
ax = 5                # Amplitud    [v]
dc = 0                # Valor medio [v]
fx = 1                # Frecuencia  [Hz]
ph = 0                # Fase        [rad]


# --- Parámetros de muestreo ---
fs = 100              # Frec. de muestreo    [Hz]
N = fs                # Número de muestras
Ts = 1/fs             # Período de muestreo  [s]
T_sim = N * Ts        # Tiempo de simulación [s]



# --- Parámetros del ruido ---
sigma_cuadrado = 4
tipo_ruido = "uniforme"
# tipo_ruido = "gaussiano"




# --- Generar señal y ruido ---
tt, xx = generador_senoidal(ax, dc, fx, ph, N, fs)
ruido = generar_ruido(len(tt), sigma_cuadrado, tipo_ruido)
senal_ruidosa = xx + ruido



# --- Calcular DFTs ---
f_sen, S_sen = dft(xx, fs)
f_rui, S_rui = dft(ruido, fs)
f_sum, S_sum = dft(senal_ruidosa, fs)


#    Calculo la  FFT (numpy) y normalizo por N
freqs = np.fft.fftfreq(N,1/fs)
X_fft = np.fft.fft(xx, N) / N
freqs_fft = np.fft.fftfreq(N, 1/fs)

# Tomo solo la parte positiva
half_N = N//2   # hasta la mitad
freqs_pos = freqs[:half_N]
X_fft_pos = X_fft[:half_N]




# --- Gráfico de la señal senoidal generada ---
Senoidal_t(tt,xx)


# --- Gráfico del ruido generado ---
ruido_t(tt,ruido,tipo_ruido)


# --- Gráfico de la señal senoidal generada con el ruido sumado ---
Senoidal_y_ruido_t(tt, senal_ruidosa)


# --- Grafico la DFT de la señal senoidal generada ---
dft_Señal(f_sen, S_sen)



# --- Grafico la DFT de la señal generada [dB] ---
dft_Señal_dB(f_sen, S_sen)


# --- Grafico la DFT de la señal + ruido ---
dft_Señal_y_ruido(f_sum, S_sum)



# --- Grafico la DFT de la señal + ruido [dB] ---
dft_Señal_y_ruido_dB(f_sum, S_sum)

# --- Grafico la DFT y la FFT de la señal ---
DFT_y_FFT(f_sen, S_sen)



# --- Estadística ---
media_exp = np.mean(ruido)
var_exp = np.var(ruido)

print(" ")
print("Media teórica: 0")
print(f"Media experimental: {media_exp:.3f}")
print(f"Varianza teórica: {sigma_cuadrado}")
print(f"Varianza experimental: {var_exp:.3f}")




