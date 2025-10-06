#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Densidad Espectral de Potencia de la señal de radio desde CSV de FFT - TODOS LOS GRÁFICOS SEPARADOS E INDIVIDUALES.

*** CÓDIGO FINAL MODIFICADO: CRITERIO DE ANCHO DE BANDA CAMBIADO A -45 dB RELATIVOS AL PICO ***
- Se mantiene el Gráfico 2 (Magnitud en dB).
- El umbral para BW en los estimadores DEP ahora es: (Pico Máximo en dB) - 45 dB.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch, windows
import pandas as pd
import os
import time

# ----------------------------
# Parámetros de la señal de radio
# ----------------------------
csv_file = 'Rock and Pop.csv'  # Archivo CSV con los datos de FFT
fs = 2.048e6                  # Frecuencia de muestreo en Hz (informativa para PSD)
fft_size = 2048               # Tamaño de la FFT inicial (solo referencia)
N_samples = None              # Usaremos toda la columna de magnitudes del CSV

# ----------------------------
# Cargar archivo .csv con encabezado
# ----------------------------
try:
    radio_data = pd.read_csv(csv_file, header=0)  # La primera fila es encabezado
except FileNotFoundError:
    print(f"Error: Asegúrate de tener el archivo '{csv_file}' en el mismo directorio.")
    exit()

# Columnas: Magnitud y Frecuencia (en MHz)
signal_full = radio_data.iloc[:,2].astype(float).values  # Columna Magnitud
freqs_MHz = radio_data.iloc[:,1].astype(float).values    # Columna Frecuencia

if N_samples is not None:
    signal = signal_full[:N_samples]
    freqs_MHz = freqs_MHz[:N_samples]
else:
    signal = signal_full

N = len(signal)
print(f"Usando {N} muestras de la magnitud de FFT para el análisis")

# ----------------------------
# Crear carpeta de resultados si no existe
# ----------------------------
os.makedirs('resultados', exist_ok=True)


# ----------------------------
# Función para calcular ancho de banda (MODIFICADA A -45 dB RELATIVOS)
# ----------------------------
def bandwidth(f, Pxx_dB, threshold_rel_dB=-45.0):
    """
    Devuelve la frecuencia mínima, máxima y ancho de banda donde el PSD
    supera el umbral fijo relativo en dB.
    
    Pxx_dB: Vector de Densidad Espectral de Potencia en dB (10*log10).
    threshold_rel_dB: Umbral fijo en dB relativo al pico (e.g., -45.0).
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
def per_smooth(x, fs, M=1024, win_type='hamming', nfft=4096):
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
# I. GRÁFICOS DE CONTEXTO (Espectro de Magnitud de Entrada)
# =========================================================================

# ----------------------------
# I.1. Espectro de Magnitud de RF de Entrada (Lineal)
# ----------------------------
plt.figure(figsize=(10, 4))
plt.plot(freqs_MHz, signal, linewidth=1.5)
plt.title("1. Espectro de Magnitud de RF (Rock and Pop) de Entrada (Lineal)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("Magnitud (Lineal)")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/1_Espectro_RF_Entrada_Lineal.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# I.2. Espectro de Magnitud de RF de Entrada (dB) - GRAFICO 2 INTACTO
# ----------------------------
# Se añade un pequeño offset para evitar log(0)
signal_dB = 20 * np.log10(signal + 1e-12)

plt.figure(figsize=(10, 4))
plt.plot(freqs_MHz, signal_dB, linewidth=1.5)
plt.title("2. Espectro de Magnitud de RF (Rock and Pop) de Entrada (dB)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/2_Espectro_RF_Entrada_dB.png', dpi=300)
plt.show()
plt.close()


# =========================================================================
# II. CÁLCULO DE ESTIMADORES DE DEP (Aplicados a la Magnitud de entrada)
# =========================================================================
UMBRAL_dB = -45.0 # Nuevo umbral fijo

# --- CÁLCULO DE PERIODOGRAMA ---
f_per_dep, Pxx_per_lin = periodogram(signal, fs=fs, window='hamming', nfft=N, scaling='density')
Pxx_per = 10 * np.log10(Pxx_per_lin + 1e-12) # dB
f_per_MHz = f_per_dep / 1e6 # Convertir a MHz
f_low_per, f_high_per, BW_per, mask_per, abs_th_per = bandwidth(f_per_MHz, Pxx_per, threshold_rel_dB=UMBRAL_dB)
print(f"Periodograma calculado.")

# --- CÁLCULO DE WELCH ---
f_w_dep, Pxx_w_lin = welch(signal, fs=fs, window='hamming', nperseg=fft_size, noverlap=fft_size // 2, scaling='density')
Pxx_w = 10 * np.log10(Pxx_w_lin + 1e-12) # dB
f_w_MHz = f_w_dep / 1e6 # Convertir a MHz
f_low_w, f_high_w, BW_w, mask_w, abs_th_w = bandwidth(f_w_MHz, Pxx_w, threshold_rel_dB=UMBRAL_dB)
print(f"Welch calculado.")

# --- CÁLCULO DE BLACKMAN-TUKEY ---
M = 1024 
f_bt_dep, Pxx_bt_lin = per_smooth(signal, fs=fs, M=M)
Pxx_bt = 10 * np.log10(Pxx_bt_lin + 1e-12) # dB
f_bt_MHz = f_bt_dep / 1e6 # Convertir a MHz
f_low_bt, f_high_bt, BW_bt, mask_bt, abs_th_bt = bandwidth(f_bt_MHz, Pxx_bt, threshold_rel_dB=UMBRAL_dB)
print(f"Blackman-Tukey calculado.")


# =========================================================================
# III. GRÁFICOS DE ESTIMADORES DE DEP (con BW resaltado)
# =========================================================================

# ----------------------------
# III.1. Periodograma Ventaneado
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(f_per_MHz, Pxx_per, label='Periodograma (dB)', color='blue')
plt.axhline(y=abs_th_per, color='r', linestyle='--', label=f'Umbral (Pico - {abs(UMBRAL_dB)} dB)') # Línea de umbral
plt.fill_between(f_per_MHz, Pxx_per, where=mask_per, color='blue', alpha=0.3, label='Ancho de banda')
plt.title("3. Estimador de DEP - Periodograma (dB)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/3_Estimador_Periodograma_dB.png', dpi=300)
plt.show()
plt.close()


# ----------------------------
# III.2. Método de Welch
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(f_w_MHz, Pxx_w, label='Welch (dB)', color='green')
plt.axhline(y=abs_th_w, color='r', linestyle='--', label=f'Umbral (Pico - {abs(UMBRAL_dB)} dB)') # Línea de umbral
plt.fill_between(f_w_MHz, Pxx_w, where=mask_w, color='green', alpha=0.3, label='Ancho de banda')
plt.title("4. Estimador de DEP - Welch (dB)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/4_Estimador_Welch_dB.png', dpi=300)
plt.show()
plt.close()


# ----------------------------
# III.3. Estimador Blackman-Tukey
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(f_bt_MHz, Pxx_bt, label=f'Blackman-Tukey (M={M}) (dB)', color='orange')
plt.axhline(y=abs_th_bt, color='r', linestyle='--', label=f'Umbral (Pico - {abs(UMBRAL_dB)} dB)') # Línea de umbral
plt.fill_between(f_bt_MHz, Pxx_bt, where=mask_bt, color='orange', alpha=0.3, label='Ancho de banda')
plt.title("5. Estimador de DEP - Blackman-Tukey (dB)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/5_Estimador_BlackmanTukey_dB.png', dpi=300)
plt.show()
plt.close()


# =========================================================================
# IV. GRÁFICOS DE ANCHO DE BANDA (SOLO el rango de BW)
# =========================================================================

# ----------------------------
# IV.1. Ancho de Banda - Periodograma
# ----------------------------
plt.figure(figsize=(10, 3))
plt.plot(f_per_MHz[mask_per], Pxx_per[mask_per], color='blue')
plt.title(f"6. Ancho de Banda (BW) - Periodograma: {BW_per:.2f} MHz (Umbral -{abs(UMBRAL_dB)} dB)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/6_BW_Periodograma_dB.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# IV.2. Ancho de Banda - Welch
# ----------------------------
plt.figure(figsize=(10, 3))
plt.plot(f_w_MHz[mask_w], Pxx_w[mask_w], color='green')
plt.title(f"7. Ancho de Banda (BW) - Welch: {BW_w:.2f} MHz (Umbral -{abs(UMBRAL_dB)} dB)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/7_BW_Welch_dB.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# IV.3. Ancho de Banda - Blackman-Tukey
# ----------------------------
plt.figure(figsize=(10, 3))
plt.plot(f_bt_MHz[mask_bt], Pxx_bt[mask_bt], color='orange')
plt.title(f"8. Ancho de Banda (BW) - Blackman-Tukey: {BW_bt:.2f} MHz (Umbral -{abs(UMBRAL_dB)} dB)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/8_BW_BlackmanTukey_dB.png', dpi=300)
plt.show()
plt.close()


# =========================================================================
# V. GRÁFICO COMPARATIVO FINAL
# =========================================================================

# ----------------------------
# V.1. Comparación de los tres métodos
# ----------------------------
plt.figure(figsize=(10,6))
plt.plot(f_per_MHz, Pxx_per, label='Periodograma (dB)', alpha=0.6, linewidth=1)
plt.plot(f_w_MHz, Pxx_w, label='Welch (dB)', alpha=0.8, linewidth=2)
plt.plot(f_bt_MHz, Pxx_bt, label='Blackman-Tukey (dB)', alpha=0.8, linewidth=2)
plt.title("9. Comparación de Estimadores de DEP (dB)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/9_Comparacion_PSD_dB.png', dpi=300)
plt.show()
plt.close()


# ----------------------------
# Tabla de anchos de banda (Última salida de consola)
# ----------------------------
tabla_bw = pd.DataFrame({
    "Método": ["Periodograma", "Welch", "Blackman–Tukey"],
    "Frecuencia baja [MHz]": [f_low_per, f_low_w, f_low_bt],
    "Frecuencia alta [MHz]": [f_high_per, f_high_w, f_high_bt],
    f"Ancho de banda [MHz] (Umbral -{abs(UMBRAL_dB)} dB)": [BW_per, BW_w, BW_bt]
})
print(f"\nTabla de Anchos de Banda (Umbral: {abs(UMBRAL_dB)} dB por debajo del pico)")
print(tabla_bw)