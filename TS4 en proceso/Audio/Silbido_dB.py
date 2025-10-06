#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Densidad Espectral de Potencia del archivo de audio 'silbido.wav' completo - TODOS LOS GRÁFICOS SEPARADOS E INDIVIDUALES.

*** CÓDIGO FINAL MODIFICADO A dBV²/Hz, CON 9 GRÁFICOS INDIVIDUALES ***
- Todos los gráficos son separados.
- DEP en Decibelios de Potencia (10 * log10(Pxx)).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch, windows
from scipy.io import wavfile
import os
import time
import pandas as pd

# ----------------------------
# Cargar archivo WAV
# ----------------------------
wav_file = 'silbido.wav'  # archivo de audio base
try:
    fs, signal_full = wavfile.read(wav_file)
except FileNotFoundError:
    print(f"Error: Asegúrate de tener el archivo '{wav_file}' en el mismo directorio.")
    exit()

# Si el audio tiene más de un canal, tomar solo el primer canal
if signal_full.ndim > 1:
    signal_full = signal_full[:, 0]

audio_signal = signal_full
N = len(audio_signal)
t = np.arange(N) / fs # Vector de tiempo
print(f"Se usarán todas las {N} muestras del silbido para el análisis")
print(f"Frecuencia de muestreo: {fs} Hz")

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
# I. GRÁFICOS DE CONTEXTO (Señal y FFT)
# =========================================================================

# ----------------------------
# I.1. Señal en el Tiempo
# ----------------------------
plt.figure(figsize=(10, 3))
plt.plot(t, audio_signal, linewidth=0.5)
plt.title("1. Señal de Audio ('silbido.wav') en el Dominio del Tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud (Bins de cuantificación)")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/1_Señal_Silbido_Tiempo.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# I.2. FFT de la Señal Original (Magnitud)
# ----------------------------
Y_fft = np.fft.fft(audio_signal)
P_fft = np.abs(Y_fft)
f_fft = np.fft.fftfreq(N, 1/fs)

half_N = N // 2
plt.figure(figsize=(10, 4))
plt.plot(f_fft[:half_N], P_fft[:half_N], linewidth=0.5)
plt.title("2. Espectro de Magnitud (FFT) de 'silbido.wav'")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud (Lineal)")
plt.xlim(0, fs/2) 
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/2_FFT_Silbido_Magnitud.png', dpi=300)
plt.show()
plt.close()


# =========================================================================
# II. CÁLCULO DE ESTIMADORES DE DEP
# =========================================================================

# --- CÁLCULO DE PERIODOGRAMA ---
start_time = time.time()
ventana = windows.hamming(N)
f_per, Pxx_per_lin = periodogram(audio_signal, fs=fs, window=ventana, nfft=N, scaling='density')
Pxx_per = 10 * np.log10(Pxx_per_lin) # dB
f_low_per, f_high_per, BW_per, mask_per = bandwidth(f_per, Pxx_per)
print(f"Periodograma calculado en {time.time() - start_time:.1f} segundos")

# --- CÁLCULO DE WELCH ---
start_time = time.time()
f_w, Pxx_w_lin = welch(audio_signal, fs=fs, window='hamming', nperseg=2048, noverlap=1024, scaling='density')
Pxx_w = 10 * np.log10(Pxx_w_lin) # dB
f_low_w, f_high_w, BW_w, mask_w = bandwidth(f_w, Pxx_w)
print(f"Welch calculado en {time.time() - start_time:.1f} segundos")

# --- CÁLCULO DE BLACKMAN-TUKEY ---
start_time = time.time()
M = 1024 
f_bt, Pxx_bt_lin = per_smooth(audio_signal, fs=fs, M=M)
Pxx_bt = 10 * np.log10(Pxx_bt_lin) # dB
f_low_bt, f_high_bt, BW_bt, mask_bt = bandwidth(f_bt, Pxx_bt)
print(f"Blackman-Tukey calculado en {time.time() - start_time:.1f} segundos")


# =========================================================================
# III. GRÁFICOS DE ESTIMADORES DE DEP (con BW resaltado)
# =========================================================================

# ----------------------------
# III.1. Periodograma Ventaneado
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(f_per, Pxx_per, label='Periodograma (dB)', color='blue')
plt.fill_between(f_per, Pxx_per, where=mask_per, color='blue', alpha=0.3, label='Ancho de banda (5% de Potencia)')
plt.title("3. Estimador de DEP - Periodograma (dB) - 'silbido.wav'")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/3_Estimador_Periodograma_Silbido_dB.png', dpi=300)
plt.show()
plt.close()


# ----------------------------
# III.2. Método de Welch
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(f_w, Pxx_w, label='Welch (dB)', color='green')
plt.fill_between(f_w, Pxx_w, where=mask_w, color='green', alpha=0.3, label='Ancho de banda (5% de Potencia)')
plt.title("4. Estimador de DEP - Welch (dB) - 'silbido.wav'")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/4_Estimador_Welch_Silbido_dB.png', dpi=300)
plt.show()
plt.close()


# ----------------------------
# III.3. Estimador Blackman-Tukey
# ----------------------------
plt.figure(figsize=(10,5))
plt.plot(f_bt, Pxx_bt, label=f'Blackman-Tukey (M={M}) (dB)', color='orange')
plt.fill_between(f_bt, Pxx_bt, where=mask_bt, color='orange', alpha=0.3, label='Ancho de banda (5% de Potencia)')
plt.title("5. Estimador de DEP - Blackman-Tukey (dB) - 'silbido.wav'")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/5_Estimador_BlackmanTukey_Silbido_dB.png', dpi=300)
plt.show()
plt.close()


# =========================================================================
# IV. GRÁFICOS DE ANCHO DE BANDA (SOLO el rango de BW)
# =========================================================================

# ----------------------------
# IV.1. Ancho de Banda - Periodograma
# ----------------------------
plt.figure(figsize=(10, 3))
plt.plot(f_per[mask_per], Pxx_per[mask_per], color='blue')
plt.title(f"6. Ancho de Banda (BW) - Periodograma: {BW_per:.2f} Hz")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/6_BW_Periodograma_Silbido_dB.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# IV.2. Ancho de Banda - Welch
# ----------------------------
plt.figure(figsize=(10, 3))
plt.plot(f_w[mask_w], Pxx_w[mask_w], color='green')
plt.title(f"7. Ancho de Banda (BW) - Welch: {BW_w:.2f} Hz")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/7_BW_Welch_Silbido_dB.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# IV.3. Ancho de Banda - Blackman-Tukey
# ----------------------------
plt.figure(figsize=(10, 3))
plt.plot(f_bt[mask_bt], Pxx_bt[mask_bt], color='orange')
plt.title(f"8. Ancho de Banda (BW) - Blackman-Tukey: {BW_bt:.2f} Hz")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/8_BW_BlackmanTukey_Silbido_dB.png', dpi=300)
plt.show()
plt.close()


# =========================================================================
# V. GRÁFICO COMPARATIVO FINAL
# =========================================================================

# ----------------------------
# V.1. Comparación de los tres métodos
# ----------------------------
plt.figure(figsize=(10,6))
plt.plot(f_per, Pxx_per, label='Periodograma (dB)', alpha=0.6, linewidth=1)
plt.plot(f_w, Pxx_w, label='Welch (dB)', alpha=0.8, linewidth=2)
plt.plot(f_bt, Pxx_bt, label='Blackman-Tukey (dB)', alpha=0.8, linewidth=2)
plt.title("9. Comparación de Estimadores de DEP (dB) - 'silbido.wav'")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/9_Comparacion_PSD_Silbido_dB.png', dpi=300)
plt.show()
plt.close()


# ----------------------------
# Tabla de anchos de banda (Última salida de consola)
# ----------------------------
tabla_bw = pd.DataFrame({
    "Método": ["Periodograma", "Welch", "Blackman–Tukey"],
    "Frecuencia baja [Hz]": [f_low_per, f_low_w, f_low_bt],
    "Frecuencia alta [Hz]": [f_high_per, f_high_w, f_high_bt],
    "Ancho de banda [Hz]": [BW_per, BW_w, BW_bt]
})
print("\nTabla de Anchos de Banda (Umbral 5% de Potencia)")
print(tabla_bw)