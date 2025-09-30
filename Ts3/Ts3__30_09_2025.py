#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimación de Amplitud y Frecuencia de una Señal

Created on Tue Sep 30 08:00:33 2025

@author: Fernando Daniel Fiamberti
"""


# ----------------------------
# IMPORTACIÓN DE LIBRERÍAS
# ----------------------------
import numpy as np  # Librería para cálculos numéricos y manejo de arrays
from scipy.signal import windows  # Librería para generación de ventanas de señales
import matplotlib.pyplot as plt  # Librería para gráficos
import pandas as pd  # Librería para manejo de datos en tablas (DataFrame)

# ----------------------------
# PARÁMETROS GENERALES DEL EXPERIMENTO
# ----------------------------
N = 1000          # Número de muestras por señal
R = 200           # Número de realizaciones por SNR
Ω0 = np.pi / 2    # Frecuencia central de la señal senoidal en rad/muestra
a0 = np.sqrt(2.0) # Amplitud calibrada para que la potencia sea 1 W


# Ventanas a evaluar en el análisis
W_names = np.array(["rectangular","flattop","blackmanharris","hann"])
W = np.stack([
    np.ones(N),  # Ventana rectangular
    windows.flattop(N, sym=False),  # Ventana flattop
    windows.blackmanharris(N, sym=False),  # Ventana Blackman-Harris
    windows.hann(N, sym=False)  # Ventana Hann
])

# Lista de SNR (Signal-to-Noise Ratio) a evaluar
snr_list = np.array([3, 10])

# Colores para graficar diferentes ventanas
colors = ["steelblue","darkorange","green","purple"]

# ----------------------------
# FUNCIÓN FFT
# ----------------------------
def fft(x):
    Nloc = x.shape[1]  # Número de muestras en cada señal
    X = np.fft.fft(x, axis=1)  # FFT de cada señal
    X_mag = np.abs(X)/Nloc  # Magnitud normalizada
    X_mag[:,1:-1]*=2  # Duplicar amplitud para bins intermedios (excepto DC y Nyquist)
    X_mag = X_mag[:,:Nloc//2+1]  # Tomar solo mitad positiva del espectro
    freqs = 2*np.pi*np.arange(X_mag.shape[1])/Nloc  # Vector de frecuencias en rad/muestra
    return freqs, X_mag

# ----------------------------
# GENERACIÓN DE SEÑALES Y RUIDO
# ----------------------------
var_signal = 1.0  # Varianza de la señal
var_noise = var_signal / (10**(snr_list/10))  # Varianza del ruido según SNR
sigma_noise = np.sqrt(var_noise)  # Desviación estándar del ruido

fr = np.random.uniform(-2.0, 2.0, (len(snr_list), R))  # Desviación aleatoria de frecuencia
Ω1_array = Ω0 + fr*(2*np.pi/N)  # Frecuencias de cada realización
n = np.arange(N)  # Vector de tiempo discreto
ruido = np.random.normal(0.0,1.0,(len(snr_list), R, N)) * sigma_noise[:, None, None]  # Ruido gaussiano
x_clean = a0*np.sin(Ω1_array[:,:,None]*n)  # Señales senoidales limpias
x_noisy = x_clean + ruido  # Señales con ruido agregado

# ----------------------------
# APLICACIÓN DE VENTANAS
# ----------------------------
xw = x_noisy[:, None, :, :] * W[None, :, None, :]  # Aplicar cada ventana a cada señal

# ----------------------------
# CÁLCULO DE FFT
# ----------------------------
freqs, Xmag = fft(xw.reshape(-1,N))  # FFT de todas las señales
Xmag = Xmag.reshape(len(snr_list), len(W), R, -1)  # Reestructurar array para separar SNR y ventana

# ----------------------------
# ESTIMACIÓN DE AMPLITUD Y FRECUENCIA
# ----------------------------
k1_array = np.round(Ω1_array * N / (2*np.pi)).astype(int)  # Índices de frecuencia teóricos
a_hats = np.take_along_axis(Xmag, k1_array[:, None, :, None], axis=3).squeeze(axis=3)  # Estimación de amplitud
k_hats = np.argmax(Xmag, axis=3)  # Índice del máximo de FFT (frecuencia estimada)
Omega_hats = freqs[k_hats]  # Frecuencia estimada en rad/muestra

mean_a_hat = np.mean(a_hats, axis=2)  # Promedio de amplitudes estimadas
bias_a = mean_a_hat - a0  # Sesgo de amplitud
var_a = np.mean((a_hats - mean_a_hat[:,:,None])**2, axis=2)  # Varianza de amplitud

delta_Omega = Omega_hats - Ω1_array[:,None,:]  # Error en frecuencia
mean_delta = np.mean(delta_Omega, axis=2)  # Promedio de error en frecuencia
var_delta = np.mean((delta_Omega - mean_delta[:,:,None])**2, axis=2)  # Varianza del error

# ----------------------------
# CREACIÓN DE TABLAS DE RESULTADOS
# ----------------------------
resultados_amp = pd.DataFrame({
    "SNR_dB": np.repeat(snr_list, len(W)),
    "Ventana": np.tile(W_names, len(snr_list)),
    "Sesgo": bias_a.ravel(),
    "Varianza": var_a.ravel(),
    "Media": mean_a_hat.ravel()
})  # Tabla de amplitud

resultados_freq = pd.DataFrame({
    "SNR_dB": np.repeat(snr_list, len(W)),
    "Ventana": np.tile(W_names, len(snr_list)),
    "Sesgo": mean_delta.ravel(),
    "Varianza": var_delta.ravel()
})  # Tabla de frecuencia

# Guardar resultados en CSV
resultados_amp.to_csv("resultados_amplitud.csv", index=False)
resultados_freq.to_csv("resultados_frecuencia.csv", index=False)

# Mostrar resultados en pantalla
print("\n=== Resultados de Amplitud ===")
print(resultados_amp)
print("\n=== Resultados de Frecuencia ===")
print(resultados_freq)

# ----------------------------
# GRAFICOS BASE
# ----------------------------
plt.figure(figsize=(8,4))
plt.plot(n, x_noisy[0,0])
plt.title(f"Señal con ruido en el tiempo - SNR={snr_list[0]} dB")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(10,6))
[plt.plot(n, W[i], label=f"{W_names[i]}") for i in range(len(W))]
plt.title("Ventanas en el dominio del tiempo")
plt.xlabel("Muestras")
plt.ylabel("Amplitud de ventana")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

fft_mean = np.mean(Xmag, axis=2)
plt.figure(figsize=(8,4))
eps = 1e-12
[plt.plot(freqs, 20*np.log10(fft_mean[idx_snr, idx_w]/np.max(fft_mean[idx_snr, idx_w]) + eps),
         label=f"{W_names[idx_w]} - SNR={snr_list[idx_snr]} dB")
 for idx_snr in range(len(snr_list)) for idx_w in range(len(W_names))]
plt.title("FFT promedio por ventana y SNR")
plt.xlabel("Frecuencia (rad/muestra)")
plt.ylabel("Magnitud (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

freqs_win = 2*np.pi*np.arange(N//2+1)/N
fft_win_mag = np.abs(np.fft.fft(W, n=N, axis=1))/N
fft_win_mag[:,1:-1]*=2
fft_win_mag = fft_win_mag[:,:N//2+1]
plt.figure(figsize=(8,4))
eps = 1e-12
[plt.plot(freqs_win, 20*np.log10(fft_win_mag[i]/np.max(fft_win_mag[i]) + eps), label=W_names[i])
 for i in range(len(W_names))]
plt.title("FFT de todas las ventanas")
plt.xlabel("Frecuencia (rad/muestra)")
plt.ylabel("Magnitud (dB)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

hist_bins_a = np.linspace(a_hats.min(), a_hats.max(), 21)
hist_bins_delta = np.linspace(delta_Omega.min(), delta_Omega.max(), 21)

plt.figure(figsize=(10,6))
[plt.hist(a_hats[idx_snr,i], bins=20, alpha=0.5, color=colors[i],
          label=f"{W_names[i]} - SNR={snr_list[idx_snr]} dB", edgecolor='black')
 for idx_snr in range(len(snr_list)) for i in range(len(W))]
plt.title("Comparación de histogramas de amplitud")
plt.xlabel("Amplitud estimada")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(10,6))
[plt.hist(delta_Omega[idx_snr,i], bins=20, alpha=0.5, color=colors[i],
          label=f"{W_names[i]} - SNR={snr_list[idx_snr]} dB", edgecolor='black')
 for idx_snr in range(len(snr_list)) for i in range(len(W))]
plt.title("Comparación de histogramas de error de frecuencia")
plt.xlabel("ΔΩ (rad/muestra)")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

# ----------------------------
# BONUS: ZERO-PADDING Y ESTIMADORES ALTERNATIVOS
# ----------------------------
N_pad = 8*N
xw_padded = np.zeros((len(snr_list), len(W), R, N_pad))
xw_padded[:,:,:,:N] = xw
freqs_zp, Xmag_zp = fft(xw_padded.reshape(-1, N_pad))
Xmag_zp = Xmag_zp.reshape(len(snr_list), len(W), R, -1)
k_hats_zp = np.argmax(Xmag_zp, axis=3)
Omega_hats_zp = 2*np.pi*np.arange(Xmag_zp.shape[3])/N_pad
Omega_hats_zp = Omega_hats_zp[k_hats_zp]

k_prev = np.clip(k_hats-1, 0, Xmag.shape[3]-1)
k_curr = k_hats
k_next = np.clip(k_hats+1, 0, Xmag.shape[3]-1)
a_bins = np.stack([
    np.take_along_axis(Xmag, k_prev[:,:,:,None], axis=3),
    np.take_along_axis(Xmag, k_curr[:,:,:,None], axis=3),
    np.take_along_axis(Xmag, k_next[:,:,:,None], axis=3)
], axis=4)
a_hats_alt = np.mean(a_bins, axis=4).squeeze(axis=3)

alpha = np.take_along_axis(Xmag, k_prev[:,:,:,None], axis=3)[...,0]
beta = np.take_along_axis(Xmag, k_curr[:,:,:,None], axis=3)[...,0]
gamma = np.take_along_axis(Xmag, k_next[:,:,:,None], axis=3)[...,0]
delta = 0.5*(alpha - gamma)/(alpha - 2*beta + gamma)
delta = np.nan_to_num(delta)
Omega_hats_parab = 2*np.pi*(k_hats + delta)/N

mean_a_hat_alt = np.mean(a_hats_alt, axis=2)
bias_a_alt = mean_a_hat_alt - a0
var_a_alt = np.mean((a_hats_alt - mean_a_hat_alt[:,:,None])**2, axis=2)
resultados_amp_alt = pd.DataFrame({
    "SNR_dB": np.repeat(snr_list, len(W)),
    "Ventana": np.tile(W_names, len(snr_list)),
    "Sesgo": bias_a_alt.ravel(),
    "Varianza": var_a_alt.ravel(),
    "Media": mean_a_hat_alt.ravel()
})
resultados_amp_alt.to_csv("resultados_amplitud_bonus.csv", index=False)
print("\n=== Resultados de Amplitud Bonus ===")
print(resultados_amp_alt)

delta_Omega_zp = Omega_hats_zp - Ω1_array[:,None,:]
mean_delta_zp = np.mean(delta_Omega_zp, axis=2)
var_delta_zp = np.mean((delta_Omega_zp - mean_delta_zp[:,:,None])**2, axis=2)
resultados_freq_zp = pd.DataFrame({
    "SNR_dB": np.repeat(snr_list, len(W)),
    "Ventana": np.tile(W_names, len(snr_list)),
    "Sesgo": mean_delta_zp.ravel(),
    "Varianza": var_delta_zp.ravel()
})
resultados_freq_zp.to_csv("resultados_frecuencia_bonus_zeropad.csv", index=False)
print("\n=== Resultados de Frecuencia Bonus Zero-Padding ===")
print(resultados_freq_zp)

delta_Omega_parab = Omega_hats_parab - Ω1_array[:,None,:]
mean_delta_parab = np.mean(delta_Omega_parab, axis=2)
var_delta_parab = np.mean((delta_Omega_parab - mean_delta_parab[:,:,None])**2, axis=2)
resultados_freq_parab = pd.DataFrame({
    "SNR_dB": np.repeat(snr_list, len(W)),
    "Ventana": np.tile(W_names, len(snr_list)),
    "Sesgo": mean_delta_parab.ravel(),
    "Varianza": var_delta_parab.ravel()
})
resultados_freq_parab.to_csv("resultados_frecuencia_bonus_parab.csv", index=False)
print("\n=== Resultados de Frecuencia Bonus Interpolación Parabólica ===")
print(resultados_freq_parab)

plt.figure(figsize=(8,4))
plt.hist(Omega_hats[0,0], bins=20, alpha=0.5, label="FFT original", color="steelblue", edgecolor='black')
plt.hist(Omega_hats_zp[0,0], bins=20, alpha=0.5, label="Zero-padding", color="darkorange", edgecolor='black')
plt.hist(Omega_hats_parab[0,0], bins=20, alpha=0.5, label="Interpolación parabólica", color="green", edgecolor='black')
plt.title(f"Comparación estimadores de frecuencia - {W_names[0]} - SNR={snr_list[0]} dB")
plt.xlabel("Ω estimada (rad/muestra)")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()

plt.figure(figsize=(8,4))
plt.hist(a_hats[0,0], bins=20, alpha=0.5, label="FFT original", color="steelblue", edgecolor='black')
plt.hist(a_hats_alt[0,0], bins=20, alpha=0.5, label="Promedio 3 bins", color="darkorange", edgecolor='black')
plt.title(f"Comparación estimadores de amplitud - {W_names[0]} - SNR={snr_list[0]} dB")
plt.xlabel("Amplitud estimada")
plt.ylabel("Frecuencia")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.close()
