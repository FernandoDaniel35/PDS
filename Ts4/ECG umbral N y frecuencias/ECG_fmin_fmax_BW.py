#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Densidad Espectral de Potencia del ECG 

Se puede fijar el rango de frecuencias para calcular el ancho de banda y dentro
de ese rango de frecuencias se puede variar el umbral.

También se puede elegir desde que muestra hasta que muestra analizar

Created on Tue Sep 23 12:09:34 2025

@author: Fernando Daniel Fiamberti
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch, windows
import pandas as pd
import os
import time

# ----------------------------
# Cargar archivo .mat
# ----------------------------
mat = sio.loadmat('ECG_TP4.mat')
ecg_full = np.squeeze(mat['ecg_lead'])
fs = 1000  # Hz

# ============================
# PARÁMETROS MANUALES DE MUESTRAS Y BW
# ============================
# ----------------------------
# AJUSTE DE RANGO DE MUESTRAS A ANALIZAR
# ----------------------------
# Muestra inicial: 0 para empezar desde el inicio.
N_sample_start = 0  
# Muestra final: 15000 por defecto, o -1 para analizar todo el archivo.
# ¡IMPORTANTE!: Si N_sample_end es mayor a la longitud del archivo, se limitará a la longitud máxima.
N_sample_end = 600000 

f_min_manual = 0  # Frecuencia mínima manual en Hz
f_max_manual = 200  # Frecuencia máxima manual en Hz
# ----------------------------
# UMBRAL MODIFICABLE DENTRO DEL RANGO MANUAL 
# ¡Se usa un umbral alto para garantizar que el BW cambie!
threshold_manual = 0.005  # Umbral relativo (ej: 0.8 = 80% del máximo dentro del rango)
# ============================

# ----------------------------
# Ajustar cantidad de muestras para cálculo
# ----------------------------
# Determina el punto final real
max_len = len(ecg_full)
if N_sample_end == -1:
    N_samples = max_len
else:
    N_samples = min(N_sample_end, max_len)
    
# Extrae el segmento
ecg = ecg_full[N_sample_start:N_samples]
N = len(ecg)
t_start = N_sample_start / fs
t_end = (N_sample_start + N) / fs
print(f"Usando {N} muestras del ECG (tiempo: {t_start:.3f}s a {t_end:.3f}s) para el análisis")


# ----------------------------
# Crear carpeta de resultados si no existe
# ----------------------------
os.makedirs('resultados', exist_ok=True)

# ----------------------------
# FUNCIÓN: Encontrar la frecuencia de máxima energía dentro del rango
# ----------------------------
def find_max_freq(f, Pxx, f_min_limit, f_max_limit):
    """
    Encuentra la frecuencia f_max que tiene la máxima densidad espectral de potencia Pxx_max
    dentro del rango [f_min_limit, f_max_limit].
    """
    idx_start = np.searchsorted(f, f_min_limit)
    idx_end = np.searchsorted(f, f_max_limit, side='right')
    
    Pxx_in_range = Pxx[idx_start:idx_end]
    f_in_range = f[idx_start:idx_end]
    
    if len(Pxx_in_range) == 0:
        return 0, 0
    
    # Encuentra el índice del máximo dentro del rango filtrado
    max_idx_in_range = np.argmax(Pxx_in_range)
    
    f_max = f_in_range[max_idx_in_range]
    Pxx_max = Pxx_in_range[max_idx_in_range]
    
    return f_max, Pxx_max

# ----------------------------
# Función ancho de banda y rango (por umbral relativo DENTRO DEL RANGO MANUAL)
# ----------------------------
def bandwidth(f, Pxx, f_min_limit, f_max_limit, threshold):
    """
    Calcula el ancho de banda de la señal en función de un umbral relativo,
    pero SOLO dentro de los límites de frecuencia f_min_limit y f_max_limit.
    """
    
    idx_start = np.searchsorted(f, f_min_limit)
    idx_end = np.searchsorted(f, f_max_limit, side='right')
    
    f_filtered = f[idx_start:idx_end]
    Pxx_filtered = Pxx[idx_start:idx_end]
    
    if len(Pxx_filtered) == 0:
        return 0, 0, 0, np.zeros_like(Pxx, dtype=bool)

    Pxx_max_in_range = np.max(Pxx_filtered)
    
    if Pxx_max_in_range == 0:
        Pxx_norm_in_range = np.zeros_like(Pxx_filtered)
    else:
        Pxx_norm_in_range = Pxx_filtered / Pxx_max_in_range
    
    mask_in_range = Pxx_norm_in_range > threshold
    
    mask_full = np.zeros_like(Pxx, dtype=bool)
    
    if np.any(mask_in_range):
        
        first_true_idx = np.where(mask_in_range)[0][0]
        last_true_idx = np.where(mask_in_range)[0][-1]
        
        f_low = f_filtered[first_true_idx]
        f_high = f_filtered[last_true_idx]
        BW = f_high - f_low
        
        mask_full[idx_start:idx_end] = mask_in_range
        
        return f_low, f_high, BW, mask_full
    else:
        return 0, 0, 0, mask_full

# ----------------------------
# Función ancho de banda y rango (manual - se mantiene intacto)
# ----------------------------
def bandwidth_manual(f, Pxx, f_min, f_max):
    """
    Calcula el ancho de banda y selecciona los datos de PSD en un rango de frecuencia manual.
    """
    mask = (f >= f_min) & (f <= f_max)
    BW = f_max - f_min
    return f_min, f_max, BW, mask

# ----------------------------
# Función Blackman-Tukey (Mantenida)
# ----------------------------
def per_smooth(x, win=1, M=1024, fs=1000, nfft=4096):
    x = np.asarray(x).flatten()
    x = x - np.mean(x)
    rxx_full = np.correlate(x, x, mode='full')
    mid = len(rxx_full) // 2
    rxx = rxx_full[mid - 1024 + 1 : mid + 1024]
    window_types = {1: 'boxcar', 2: 'hamming', 3: 'hann', 4: 'bartlett', 5: 'blackman'}
    w = windows.get_window(window_types.get(win, 'boxcar'), len(rxx))
    rxx_win = rxx * w
    Pxx_full = np.abs(np.fft.fft(rxx_win, nfft))
    n_unique = nfft // 2 + 1
    Pxx_uni = Pxx_full[:n_unique]
    Pxx = Pxx_uni.copy()
    Pxx[1:-1] *= 2
    f = np.linspace(0, fs / 2, n_unique)
    return f, Pxx

# ----------------------------
# Gráficos Iniciales (Tiempo y FFT)
# ----------------------------
plt.figure(figsize=(10,4))
plt.plot(np.arange(N)/fs + t_start, ecg, color='blue') 
plt.title(f"ECG original en el tiempo ({t_start:.1f}s - {t_end:.1f}s)")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [V]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/ECG_original_tiempo.png', dpi=300)
plt.show() 
# plt.close() <--- REMOVIDO

fft_ecg = np.fft.fft(ecg, n=N)
f_fft = np.linspace(0, fs/2, N//2)
fft_mag = np.abs(fft_ecg[:N//2])

plt.figure(figsize=(10,4))
plt.semilogy(f_fft, fft_mag)
plt.title("FFT del ECG original")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.xlim(f_min_manual, f_max_manual)
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/FFT_ECG.png', dpi=300)
plt.show() 
# plt.close() <--- REMOVIDO

# -------------------------------------------------------------
# 1) Periodograma ventaneado
# -------------------------------------------------------------
ventana = windows.hamming(N)
f_per, Pxx_per = periodogram(ecg, fs=fs, window=ventana, nfft=N, scaling='density')
f_low_per_th, f_high_per_th, BW_per_th, mask_per_th = bandwidth(f_per, Pxx_per, f_min_manual, f_max_manual, threshold_manual)
f_low_per_man, f_high_per_man, BW_per_man, mask_per_man = bandwidth_manual(f_per, Pxx_per, f_min_manual, f_max_manual)
f_max_per, Pxx_max_per = find_max_freq(f_per, Pxx_per, f_min_manual, f_max_manual)

# GRAFICO 1.1: PSD sin sombreado
plt.figure(figsize=(10,5))
plt.semilogy(f_per, Pxx_per, label='Densidad Espectral de Potencia')
plt.axvline(f_max_per, color='red', linestyle='--', label=f'Frecuencia Máx. Energía ({f_max_per:.3f} Hz)')
plt.title("Densidad Espectral de Potencia - Periodograma Ventaneado (Completa)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [V²/Hz]")
plt.legend()
plt.xlim(f_min_manual, f_max_manual)
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/Densidad_Espectral_de_Potencia_periodograma_completa.png', dpi=300)
plt.show() 
# plt.close() <--- REMOVIDO

# GRAFICO 1.2: PSD con sombreado de Umbral
plt.figure(figsize=(10,5))
plt.semilogy(f_per, Pxx_per, label='Densidad Espectral de Potencia')
plt.fill_between(f_per, Pxx_per, where=mask_per_th, color='orange', alpha=0.3, label=f'Ancho de banda (Umbral {threshold_manual*100:.1f}%)')
plt.axvline(f_max_per, color='red', linestyle='--', label=f'Frecuencia Máx. Energía ({f_max_per:.3f} Hz)')
plt.title("Densidad Espectral de Potencia - Periodograma (Umbral en rango)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [V²/Hz]")
plt.legend()
plt.xlim(f_min_manual, f_max_manual)
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/Densidad_Espectral_de_Potencia_periodograma_umbral.png', dpi=300)
plt.show() 
# plt.close() <--- REMOVIDO

# -------------------------------------------------------------
# 2) Welch
# -------------------------------------------------------------
f_welch, Pxx_welch = welch(ecg, fs=fs, window='hamming', nperseg=2048, noverlap=1024, scaling='density')
f_low_w_th, f_high_w_th, BW_w_th, mask_w_th = bandwidth(f_welch, Pxx_welch, f_min_manual, f_max_manual, threshold_manual)
f_low_w_man, f_high_w_man, BW_w_man, mask_w_man = bandwidth_manual(f_welch, Pxx_welch, f_min_manual, f_max_manual)
f_max_w, Pxx_max_w = find_max_freq(f_welch, Pxx_welch, f_min_manual, f_max_manual)

# GRAFICO 2.1: PSD sin sombreado
plt.figure(figsize=(10,5))
plt.semilogy(f_welch, Pxx_welch, label='Densidad Espectral de Potencia')
plt.axvline(f_max_w, color='red', linestyle='--', label=f'Frecuencia Máx. Energía ({f_max_w:.3f} Hz)')
plt.title("Densidad Espectral de Potencia - Método de Welch (Completa)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [V²/Hz]")
plt.legend()
plt.xlim(f_min_manual, f_max_manual)
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/Densidad_Espectral_de_Potencia_welch_completa.png', dpi=300)
plt.show() 
# plt.close() <--- REMOVIDO

# GRAFICO 2.2: PSD con sombreado de Umbral
plt.figure(figsize=(10,5))
plt.semilogy(f_welch, Pxx_welch, label='Densidad Espectral de Potencia')
plt.fill_between(f_welch, Pxx_welch, where=mask_w_th, color='orange', alpha=0.3, label=f'Ancho de banda (Umbral {threshold_manual*100:.1f}%)')
plt.axvline(f_max_w, color='red', linestyle='--', label=f'Frecuencia Máx. Energía ({f_max_w:.3f} Hz)')
plt.title("Densidad Espectral de Potencia - Método de Welch (Umbral en rango)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [V²/Hz]")
plt.legend()
plt.xlim(f_min_manual, f_max_manual)
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/Densidad_Espectral_de_Potencia_welch_umbral.png', dpi=300)
plt.show() 
# plt.close() <--- REMOVIDO

# -------------------------------------------------------------
# 3) Blackman-Tukey
# -------------------------------------------------------------
start_time = time.time()
f_bt, Pxx_bt = per_smooth(ecg, win=2, M=1024, fs=fs, nfft=4096)
end_time = time.time()
print(f"Blackman-Tukey calculado en {end_time - start_time:.1f} segundos")
f_low_bt_th, f_high_bt_th, BW_bt_th, mask_bt_th = bandwidth(f_bt, Pxx_bt, f_min_manual, f_max_manual, threshold_manual)
f_low_bt_man, f_high_bt_man, BW_bt_man, mask_bt_man = bandwidth_manual(f_bt, Pxx_bt, f_min_manual, f_max_manual)
f_max_bt, Pxx_max_bt = find_max_freq(f_bt, Pxx_bt, f_min_manual, f_max_manual)


# GRAFICO 3.1: PSD sin sombreado
plt.figure(figsize=(10,5))
plt.semilogy(f_bt, Pxx_bt, label='Densidad Espectral de Potencia')
plt.axvline(f_max_bt, color='red', linestyle='--', label=f'Frecuencia Máx. Energía ({f_max_bt:.3f} Hz)')
plt.title("Densidad Espectral de Potencia - Blackman-Tukey (Completa)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [V²/Hz]")
plt.legend()
plt.xlim(f_min_manual, f_max_manual)
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/Densidad_Espectral_de_Potencia_blackman_tukey_completa.png', dpi=300)
plt.show() 
# plt.close() <--- REMOVIDO

# GRAFICO 3.2: PSD con sombreado de Umbral
plt.figure(figsize=(10,5))
plt.semilogy(f_bt, Pxx_bt, label='Densidad Espectral de Potencia')
plt.fill_between(f_bt, Pxx_bt, where=mask_bt_th, color='orange', alpha=0.3, label=f'Ancho de banda (Umbral {threshold_manual*100:.1f}%)')
plt.axvline(f_max_bt, color='red', linestyle='--', label=f'Frecuencia Máx. Energía ({f_max_bt:.3f} Hz)')
plt.title("Densidad Espectral de Potencia - Blackman-Tukey (Umbral en rango)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [V²/Hz]")
plt.legend()
plt.xlim(f_min_manual, f_max_manual)
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/Densidad_Espectral_de_Potencia_blackman_tukey_umbral.png', dpi=300)
plt.show() 
# plt.close() <--- REMOVIDO


# ----------------------------
# Tabla de anchos de banda (SOLO UMBRAL)
# ----------------------------
tabla_bw_umbral = pd.DataFrame({
    "Método": ["Periodograma", "Welch", "Blackman–Tukey"],
    f"Umbral Aplicado": [f"{threshold_manual*100:.1f}%", f"{threshold_manual*100:.1f}%", f"{threshold_manual*100:.1f}%"],
    "Frecuencia baja [Hz]": [f_low_per_th, f_low_w_th, f_low_bt_th],
    "Frecuencia alta [Hz]": [f_high_per_th, f_high_w_th, f_high_bt_th],
    "Ancho de banda [Hz]": [BW_per_th, BW_w_th, BW_bt_th],
    "Frecuencia Max Energía [Hz]": [f_max_per, f_max_w, f_max_bt]
})

print(f"\nResultados del Ancho de Banda (BW) calculado por Umbral (Umbral = {threshold_manual*100:.1f}% del Máximo Local):\n")
print(tabla_bw_umbral.set_index('Método').to_string(float_format="%.3f"))

# IMPRESIÓN DE FRECUENCIAS DE MÁXIMA ENERGÍA AL FINAL
print(f"\n--- Frecuencias de Máxima Energía (en [{f_min_manual}Hz, {f_max_manual}Hz]) ---")
print(f"Periodograma:    {f_max_per:.3f} Hz")
print(f"Welch:           {f_max_w:.3f} Hz")
print(f"Blackman-Tukey:  {f_max_bt:.3f} Hz")
print("----------------------------------------------------------------------")


# Guardar la tabla COMPLETA (incluyendo Manual) en el CSV para referencia
tabla_bw_full = pd.DataFrame({
    "Método": ["Periodograma (Umbral)", "Periodograma (Manual)", 
               "Welch (Umbral)", "Welch (Manual)", 
               "Blackman–Tukey (Umbral)", "Blackman–Tukey (Manual)"],
    "Frecuencia baja [Hz]": [f_low_per_th, f_low_per_man, 
                             f_low_w_th, f_low_w_man, 
                             f_low_bt_th, f_low_bt_man],
    "Frecuencia alta [Hz]": [f_high_per_th, f_high_per_man, 
                             f_high_w_th, f_high_w_man, 
                             f_low_bt_th, f_high_bt_man],
    "Ancho de banda [Hz]": [BW_per_th, BW_per_man, 
                            BW_w_th, BW_w_man, 
                            BW_bt_th, BW_bt_man]
})

tabla_bw_full.to_csv('resultados/ancho_de_banda_Densidad_Espectral_de_Potencia.csv', index=False, float_format='%.3f')
print("\n Tabla completa (Umbral y Manual) guardada en 'resultados/ancho_de_banda_Densidad_Espectral_de_Potencia.csv'")
print(" Todos los 8 gráficos (ECG, FFT, y 6 de PSD) se abrirán en ventanas separadas y se guardarán en 'resultados/'")