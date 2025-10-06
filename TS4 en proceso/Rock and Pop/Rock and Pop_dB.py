#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Densidad Espectral de Potencia de la señal de radio desde CSV de FFT - TODOS LOS GRÁFICOS SEPARADOS E INDIVIDUALES.

*** CÓDIGO FINAL MODIFICADO: AÑADIDO GRÁFICO 2 CON EL ESPECTRO DE MAGNITUD EN dB ***
- Ahora hay 9 gráficos individuales.
- El Gráfico 2 muestra la Magnitud de Entrada en dB (20 * log10(Magnitud)).
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
# Función para calcular ancho de banda (ADAPTADA para trabajar con dB de potencia)
# ----------------------------
def bandwidth(f, Pxx_dB, threshold_rel_power=0.05):
    """
    Devuelve la frecuencia mínima, máxima y ancho de banda donde el PSD
    supera el umbral relativo, TRABAJANDO CON LA ESCALA EN dB.
    
    Pxx_dB: Vector de Densidad Espectral de Potencia en dB (10*log10).
    """
    threshold_dB_offset = 10 * np.log10(threshold_rel_power)
    absolute_threshold_dB = np.max(Pxx_dB) + threshold_dB_offset
    
    mask = Pxx_dB > absolute_threshold_dB
    
    if np.any(mask):
        f_bw = f[mask]
        f_low = f_bw[0]
        f_high = f_bw[-1]
        BW = f_high - f_low
        return f_low, f_high, BW, mask
    else:
        return 0, 0, 0, mask

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
# I.2. Espectro de Magnitud de RF de Entrada (dB)
# ----------------------------
# *** MODIFICACIÓN SOLICITADA: Magnitud de entrada en dB (20 * log10(Magnitud)) ***
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

# --- CÁLCULO DE PERIODOGRAMA ---
f_per_dep, Pxx_per_lin = periodogram(signal, fs=fs, window='hamming', nfft=N, scaling='density')
Pxx_per = 10 * np.log10(Pxx_per_lin) # dB
f_per_MHz = f_per_dep / 1e6 # Convertir a MHz
f_low_per, f_high_per, BW_per, mask_per = bandwidth(f_per_MHz, Pxx_per)
print(f"Periodograma calculado.")

# --- CÁLCULO DE WELCH ---
f_w_dep, Pxx_w_lin = welch(signal, fs=fs, window='hamming', nperseg=fft_size, noverlap=fft_size // 2, scaling='density')
Pxx_w = 10 * np.log10(Pxx_w_lin) # dB
f_w_MHz = f_w_dep / 1e6 # Convertir a MHz
f_low_w, f_high_w, BW_w, mask_w = bandwidth(f_w_MHz, Pxx_w)
print(f"Welch calculado.")

# --- CÁLCULO DE BLACKMAN-TUKEY ---
M = 1024 
f_bt_dep, Pxx_bt_lin = per_smooth(signal, fs=fs, M=M)
Pxx_bt = 10 * np.log10(Pxx_bt_lin) # dB
f_bt_MHz = f_bt_dep / 1e6 # Convertir a MHz
f_low_bt, f_high_bt, BW_bt, mask_bt = bandwidth(f_bt_MHz, Pxx_bt)
print(f"Blackman-Tukey calculado.")


# =========================================================================
# III. GRÁFICOS DE ESTIMADORES DE DEP (con BW resaltado)
# Se reordenan los números de los gráficos para que sigan del 2 (el nuevo)
# =========================================================================

# ----------------------------
# III.1. Periodograma Ventaneado
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(f_per_MHz, Pxx_per, label='Periodograma (dB)', color='blue')
plt.fill_between(f_per_MHz, Pxx_per, where=mask_per, color='blue', alpha=0.3, label='Ancho de banda (5% de Potencia)')
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
plt.fill_between(f_w_MHz, Pxx_w, where=mask_w, color='green', alpha=0.3, label='Ancho de banda (5% de Potencia)')
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
plt.fill_between(f_bt_MHz, Pxx_bt, where=mask_bt, color='orange', alpha=0.3, label='Ancho de banda (5% de Potencia)')
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
plt.title(f"6. Ancho de Banda (BW) - Periodograma: {BW_per:.2f} MHz")
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
plt.title(f"7. Ancho de Banda (BW) - Welch: {BW_w:.2f} MHz")
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
plt.title(f"8. Ancho de Banda (BW) - Blackman-Tukey: {BW_bt:.2f} MHz")
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
    "Ancho de banda [MHz]": [BW_per, BW_w, BW_bt]
})
print("\nTabla de Anchos de Banda (Umbral 5% de Potencia)")
print(tabla_bw)