#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Densidad Espectral de Potencia del ECG - gráficos separados con ancho de banda resaltado
y cantidad de muestras ajustada para ~2 minutos de cálculo

*** CÓDIGO FINAL ESTABLE: Umbral de BW en -10 dB RELATIVOS AL PICO y clave 'ecg_lead' corregida. ***
"""

import numpy as np                       # Para manejo de arrays y cálculos numéricos
import scipy.io as sio                   # Para leer archivos .mat (MATLAB)
import matplotlib.pyplot as plt          # Para generar gráficos
from scipy.signal import periodogram, welch, windows  # Para cálculo de PSD y ventanas
import pandas as pd                      # Para manejar tablas de resultados
import os
import time
import sys # Importar sys para usar sys.exit()

# ----------------------------
# Cargar archivo .mat y extraer señal ECG
# ----------------------------
mat_file = 'ECG_TP4.mat'
# CLAVE CORREGIDA: La señal completa del ECG es 'ecg_lead'.
SIGNAL_KEY = 'ecg_lead' 

try:
    mat_data = sio.loadmat(mat_file)
except FileNotFoundError:
    print(f"Error: Asegúrate de tener el archivo '{mat_file}' en el mismo directorio.")
    sys.exit()

try:
    # Usamos la clave correcta para cargar la señal
    signal_full = mat_data[SIGNAL_KEY].flatten()
    print(f"Éxito: Señal cargada con la clave: '{SIGNAL_KEY}'.")
except KeyError:
    # Si esta clave falla, el archivo no contiene la señal esperada.
    data_keys = [key for key in mat_data.keys() if not key.startswith('__')]
    print(f"Error: Clave de señal esperada ('{SIGNAL_KEY}') no encontrada.")
    print(f"Claves disponibles en '{mat_file}': {data_keys}")
    sys.exit()

fs = 1000  # Frecuencia de muestreo: 1000 Hz

# ----------------------------
# Ajustar cantidad de muestras para cálculo (~2 minutos)
# ----------------------------
N_samples = 120000  # ~2 minutos de datos (120 * 1000 Hz)
ecg_signal = signal_full[:N_samples]
N = len(ecg_signal)
print(f"Usando {N} muestras de ECG para el análisis (~2 min cálculo)")
print(f"Frecuencia de muestreo: {fs} Hz")

# ----------------------------
# Crear carpeta de resultados si no existe
# ----------------------------
os.makedirs('resultados', exist_ok=True)

# ----------------------------
# Función para calcular ancho de banda (UMBRAL -10 dB RELATIVOS)
# ----------------------------
def bandwidth_dB(f, Pxx_dB, threshold_rel_dB=-10.0): # <-- MODIFICACIÓN A -10.0 dB
    """
    Devuelve la frecuencia mínima, máxima y ancho de banda donde el PSD
    supera el umbral fijo relativo en dB.
    
    Pxx_dB: Vector de Densidad Espectral de Potencia en dB (10*log10).
    threshold_rel_dB: Umbral fijo en dB relativo al pico (e.g., -10.0).
    """
    
    absolute_threshold_dB = np.max(Pxx_dB) + threshold_rel_dB
    
    mask = Pxx_dB > absolute_threshold_dB
    
    if np.any(mask):
        f_bw = f[mask]
        f_low = f_bw[0]
        f_high = f_bw[-1]
        BW = f_high - f_low
        return f_low, f_high, BW, mask, absolute_threshold_dB
    else:
        # Devolver el umbral incluso si no hay datos para fines de trazado
        return 0, 0, 0, mask, absolute_threshold_dB

# ----------------------------
# Función Blackman-Tukey (per_smooth)
# ----------------------------
def per_smooth(x, fs, M=2048, win_type='hamming', nfft=4096):
    """
    Estimador de DEP Blackman-Tukey. Retorna Pxx en escala LINEAL.
    """
    x = np.asarray(x).flatten()
    x = x - np.mean(x)
    
    rxx_full = np.correlate(x, x, mode='full')
    mid = len(rxx_full) // 2
    rxx = rxx_full[mid - M : mid + M + 1]  
    
    w = windows.get_window(win_type, len(rxx))
    rxx_win = rxx * w
    
    Pxx_full = np.abs(np.fft.fft(rxx_win, nfft))
    
    n_unique = nfft // 2 + 1
    Pxx = Pxx_full[:n_unique].copy()
    Pxx[1:-1] *= 2 
    
    f = np.linspace(0, fs / 2, n_unique)
    return f, Pxx


# =========================================================================
# I. GRÁFICOS DE CONTEXTO (Señal y Espectro de Magnitud)
# =========================================================================

# ----------------------------
# I.1. Señal en el tiempo
# ----------------------------
time_vector = np.arange(N) / fs
plt.figure(figsize=(10, 4))
plt.plot(time_vector, ecg_signal, linewidth=0.5)
plt.title("1. Señal ECG en el Dominio del Tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [unidades relativas]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/1_ECG_Tiempo_M10.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# I.2. Espectro de Magnitud de FFT (dB)
# ----------------------------
# Cálculo de la FFT de la señal
Y = np.fft.fft(ecg_signal)
P2 = np.abs(Y / N)
P1 = P2[:N//2 + 1]
P1[1:-1] = 2 * P1[1:-1]
f_fft = fs * np.arange(N//2 + 1) / N

# Conversión a dB (20*log10(Magnitud))
P1_dB = 20 * np.log10(P1 + 1e-12) # Se añade un offset para evitar log(0)

plt.figure(figsize=(10, 4))
plt.plot(f_fft, P1_dB, linewidth=1.5)
plt.title("2. Espectro de Magnitud FFT del ECG (dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.xlim(0, 100) # Limitar a la banda de interés biomédica
plt.tight_layout()
plt.savefig('resultados/2_Espectro_ECG_FFT_dB_M10.png', dpi=300)
plt.show()
plt.close()


# =========================================================================
# II. CÁLCULO DE ESTIMADORES DE DEP Y BW (EN dB)
# =========================================================================
UMBRAL_dB = -10.0 # Umbral fijo: -10 dB

# --- CÁLCULO DE PERIODOGRAMA ---
f_per, Pxx_per_lin = periodogram(ecg_signal, fs=fs, window='hamming', nfft=N, scaling='density')
Pxx_per_dB = 10 * np.log10(Pxx_per_lin + 1e-12) # dB
f_low_per, f_high_per, BW_per, mask_per, abs_th_per = bandwidth_dB(f_per, Pxx_per_dB, threshold_rel_dB=UMBRAL_dB)

# --- CÁLCULO DE WELCH ---
fft_size = 2048 # Tamaño de segmento para Welch
f_w, Pxx_w_lin = welch(ecg_signal, fs=fs, window='hamming', nperseg=fft_size, noverlap=fft_size // 2, scaling='density')
Pxx_w_dB = 10 * np.log10(Pxx_w_lin + 1e-12) # dB
f_low_w, f_high_w, BW_w, mask_w, abs_th_w = bandwidth_dB(f_w, Pxx_w_dB, threshold_rel_dB=UMBRAL_dB)

# --- CÁLCULO DE BLACKMAN-TUKEY ---
M = 2048 # Número de lags para Blackman-Tukey
f_bt, Pxx_bt_lin = per_smooth(ecg_signal, fs=fs, M=M)
Pxx_bt_dB = 10 * np.log10(Pxx_bt_lin + 1e-12) # dB
f_low_bt, f_high_bt, BW_bt, mask_bt, abs_th_bt = bandwidth_dB(f_bt, Pxx_bt_dB, threshold_rel_dB=UMBRAL_dB)


# =========================================================================
# III. GRÁFICOS DE ESTIMADORES DE DEP (con BW resaltado)
# =========================================================================

# ----------------------------
# III.1. Periodograma Ventaneado
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(f_per, Pxx_per_dB, label='Periodograma (dB)', color='blue')
plt.axhline(y=abs_th_per, color='r', linestyle='--', label=f'Umbral (Pico - {abs(UMBRAL_dB)} dB)') # Línea de umbral
plt.fill_between(f_per, Pxx_per_dB, where=mask_per, color='blue', alpha=0.3, label='Ancho de banda')
plt.title("3. Estimador de DEP - Periodograma (dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel(r"Densidad Espectral $[\text{dBV}^2/\text{Hz}]$") # Notación LaTeX para V²/Hz
plt.legend()
plt.grid(True)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig('resultados/3_Estimador_Periodograma_dB_M10.png', dpi=300)
plt.show()
plt.close()


# ----------------------------
# III.2. Método de Welch
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(f_w, Pxx_w_dB, label='Welch (dB)', color='green')
plt.axhline(y=abs_th_w, color='r', linestyle='--', label=f'Umbral (Pico - {abs(UMBRAL_dB)} dB)') # Línea de umbral
plt.fill_between(f_w, Pxx_w_dB, where=mask_w, color='green', alpha=0.3, label='Ancho de banda')
plt.title("4. Estimador de DEP - Welch (dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel(r"Densidad Espectral $[\text{dBV}^2/\text{Hz}]$")
plt.legend()
plt.grid(True)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig('resultados/4_Estimador_Welch_dB_M10.png', dpi=300)
plt.show()
plt.close()


# ----------------------------
# III.3. Estimador Blackman-Tukey
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(f_bt, Pxx_bt_dB, label=f'Blackman-Tukey (M={M}) (dB)', color='orange')
plt.axhline(y=abs_th_bt, color='r', linestyle='--', label=f'Umbral (Pico - {abs(UMBRAL_dB)} dB)') # Línea de umbral
plt.fill_between(f_bt, Pxx_bt_dB, where=mask_bt, color='orange', alpha=0.3, label='Ancho de banda')
plt.title("5. Estimador de DEP - Blackman-Tukey (dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel(r"Densidad Espectral $[\text{dBV}^2/\text{Hz}]$")
plt.legend()
plt.grid(True)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig('resultados/5_Estimador_BlackmanTukey_dB_M10.png', dpi=300)
plt.show()
plt.close()


# =========================================================================
# IV. GRÁFICOS DE ANCHO DE BANDA (SOLO el rango de BW)
# =========================================================================

# ----------------------------
# IV.1. Ancho de Banda - Periodograma
# ----------------------------
plt.figure(figsize=(10, 3))
plt.plot(f_per[mask_per], Pxx_per_dB[mask_per], color='blue')
plt.title(f"6. Ancho de Banda (BW) - Periodograma: {BW_per:.2f} Hz (Umbral -{abs(UMBRAL_dB)} dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel(r"Densidad Espectral $[\text{dBV}^2/\text{Hz}]$")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/6_BW_Periodograma_dB_M10.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# IV.2. Ancho de Banda - Welch
# ----------------------------
plt.figure(figsize=(10, 3))
plt.plot(f_w[mask_w], Pxx_w_dB[mask_w], color='green')
plt.title(f"7. Ancho de Banda (BW) - Welch: {BW_w:.2f} Hz (Umbral -{abs(UMBRAL_dB)} dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel(r"Densidad Espectral $[\text{dBV}^2/\text{Hz}]$")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/7_BW_Welch_dB_M10.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# IV.3. Ancho de Banda - Blackman-Tukey
# ----------------------------
plt.figure(figsize=(10, 3))
plt.plot(f_bt[mask_bt], Pxx_bt_dB[mask_bt], color='orange')
plt.title(f"8. Ancho de Banda (BW) - Blackman-Tukey: {BW_bt:.2f} Hz (Umbral -{abs(UMBRAL_dB)} dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel(r"Densidad Espectral $[\text{dBV}^2/\text{Hz}]$")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/8_BW_BlackmanTukey_dB_M10.png', dpi=300)
plt.show()
plt.close()


# =========================================================================
# V. GRÁFICO COMPARATIVO FINAL
# =========================================================================

# ----------------------------
# V.1. Comparación de los tres métodos
# ----------------------------
plt.figure(figsize=(10,6))
plt.plot(f_per, Pxx_per_dB, label='Periodograma (dB)', alpha=0.6, linewidth=1)
plt.plot(f_w, Pxx_w_dB, label='Welch (dB)', alpha=0.8, linewidth=2)
plt.plot(f_bt, Pxx_bt_dB, label='Blackman-Tukey (dB)', alpha=0.8, linewidth=2)
plt.title("9. Comparación de Estimadores de DEP (dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel(r"Densidad Espectral $[\text{dBV}^2/\text{Hz}]$")
plt.legend()
plt.grid(True)
plt.xlim(0, 100)
plt.tight_layout()
plt.savefig('resultados/9_Comparacion_PSD_dB_M10.png', dpi=300)
plt.show()
plt.close()


# ----------------------------
# Tabla de anchos de banda (Última salida de consola)
# ----------------------------
tabla_bw = pd.DataFrame({
    "Método": ["Periodograma", "Welch", "Blackman–Tukey"],
    "Frecuencia baja [Hz]": [f_low_per, f_low_w, f_low_bt],
    "Frecuencia alta [Hz]": [f_high_per, f_high_w, f_high_bt],
    f"Ancho de banda [Hz] (Umbral -{abs(UMBRAL_dB)} dB)": [BW_per, BW_w, BW_bt]
})
print(f"\nTabla de Anchos de Banda (Umbral: {abs(UMBRAL_dB)} dB por debajo del pico)")
print(tabla_bw)