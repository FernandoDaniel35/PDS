#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Densidad Espectral de Potencia del archivo de audio '10kHz.wav' completo.
Incluye análisis mediante:
- Periodograma ventaneado
- Welch
- Blackman-Tukey
Todos los comentarios y resultados están adaptados al análisis de un 10kHz grabado.
*** CÓDIGO FINAL MODIFICADO: FFT Y DEP A ESCALA dB (20*log10 y 10*log10) ***
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch, windows
from scipy.io import wavfile
import os
import time
import pandas as pd

# ----------------------------
# Cargar archivo WAV (10kHz)
# ----------------------------
wav_file = '10kHz_44100Hz_16bit_05sec.wav'  # archivo de audio base
fs, signal_full = wavfile.read(wav_file)

# Si el audio tiene más de un canal, tomar solo el primer canal
if signal_full.ndim > 1:
    signal_full = signal_full[:, 0]

audio_signal = signal_full
N = len(audio_signal)
print(f"Se usarán todas las {N} muestras del 10kHz para el análisis")
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
    
    Pxx_dB: Vector de Densidad Espectral de Potencia en dB (10*log10).
    """
    # Convertir el umbral relativo (e.g., 0.05) a un offset en dB
    threshold_dB_offset = 10 * np.log10(threshold_rel_power)
    # Calcular el umbral absoluto en dB
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
# Función Blackman-Tukey modular
# ----------------------------
def per_smooth(x, win=1, M=1024, fs=1000, nfft=4096):
    """
    Implementación del método Blackman-Tukey para estimar la PSD.
    Devuelve el espectro unilateral (solo frecuencias positivas) en escala LINEAL.
    """
    x = np.asarray(x).flatten()
    N = len(x)
    x = x - np.mean(x)  # Centrar la señal

    # ----------------------------
    # Autocorrelación
    # ----------------------------
    rxx_full = np.correlate(x, x, mode='full')
    mid = len(rxx_full) // 2
    rxx = rxx_full[mid - M + 1 : mid + M]  # Simétrico alrededor de 0

    # ----------------------------
    # Ventana
    # ----------------------------
    window_types = {1: 'boxcar', 2: 'hamming', 3: 'hann', 4: 'bartlett', 5: 'blackman'}
    w = windows.get_window(window_types.get(win, 'boxcar'), len(rxx))
    rxx_win = rxx * w

    # ----------------------------
    # FFT de la autocorrelación suavizada
    # ----------------------------
    Pxx_full = np.abs(np.fft.fft(rxx_win, nfft))
    
    # --- Espectro Unilateral ---
    n_unique = nfft // 2 + 1
    Pxx_uni = Pxx_full[:n_unique]
    Pxx = Pxx_uni.copy()
    Pxx[1:-1] *= 2 # Multiplicar por 2 los puntos intermedios
    
    # Vector de frecuencias para el espectro unilateral
    f = np.linspace(0, fs / 2, n_unique)

    return f, Pxx


# ----------------------------
# 0) Señal de 10kHz en el tiempo
# ----------------------------
plt.figure(figsize=(10, 4))
plt.plot(np.arange(N) / fs, audio_signal, color='blue')
plt.title("0. Señal de 10kHz en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/0_10kHz_tiempo.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# 1) FFT del 10kHz (en dB)
# ----------------------------
fft_signal = np.fft.fft(audio_signal, n=N)
f_fft = np.linspace(0, fs / 2, N // 2)

# MODIFICACIÓN CLAVE: Magnitud en dB (20 * log10(Magnitud))
fft_mag = np.abs(fft_signal[:N // 2])
# Se añade un pequeño offset para evitar log(0)
fft_mag_dB = 20 * np.log10(fft_mag + 1e-12) 

plt.figure(figsize=(10, 4))
plt.plot(f_fft, fft_mag_dB) # Usar plt.plot en lugar de plt.semilogy
plt.title("1. FFT del 10kHz (dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud [dB]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/1_FFT_10kHz_dB.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# 2) Periodograma ventaneado (en dB)
# ----------------------------
ventana = windows.hamming(N)
f_per, Pxx_per_lin = periodogram(audio_signal, fs=fs, window=ventana, nfft=N, scaling='density')
Pxx_per_dB = 10 * np.log10(Pxx_per_lin + 1e-12) # Convertir a dB (10*log10)
f_low_per, f_high_per, BW_per, mask_per = bandwidth(f_per, Pxx_per_dB) # Usar Pxx_per_dB

plt.figure(figsize=(10, 5))
plt.plot(f_per, Pxx_per_dB, label='Densidad Espectral de Potencia') # Usar plt.plot
plt.fill_between(f_per, Pxx_per_dB, where=mask_per, color='orange', alpha=0.3, label='Ancho de banda (5% de Potencia)')
plt.title("2. Densidad Espectral de Potencia - Periodograma Ventaneado (Hamming) - 10kHz (dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/2_DEP_Periodograma_10kHz_dB.png', dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(f_per[mask_per], Pxx_per_dB[mask_per], color='orange')
plt.title(f"3. Ancho de banda - Periodograma (10kHz): {BW_per:.2f} Hz")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/3_AnchoBanda_Periodograma_10kHz_dB.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# 4) Método de Welch (en dB)
# ----------------------------
f_welch, Pxx_welch_lin = welch(audio_signal, fs=fs, window='hamming', nperseg=2048, noverlap=1024, scaling='density')
Pxx_welch_dB = 10 * np.log10(Pxx_welch_lin + 1e-12) # Convertir a dB (10*log10)
f_low_w, f_high_w, BW_w, mask_w = bandwidth(f_welch, Pxx_welch_dB) # Usar Pxx_welch_dB

plt.figure(figsize=(10, 5))
plt.plot(f_welch, Pxx_welch_dB, label='Densidad Espectral de Potencia') # Usar plt.plot
plt.fill_between(f_welch, Pxx_welch_dB, where=mask_w, color='orange', alpha=0.3, label='Ancho de banda (5% de Potencia)')
plt.title("4. Densidad Espectral de Potencia - Welch - 10kHz (dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/4_DEP_Welch_10kHz_dB.png', dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(f_welch[mask_w], Pxx_welch_dB[mask_w], color='orange')
plt.title(f"5. Ancho de banda - Welch (10kHz): {BW_w:.2f} Hz")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/5_AnchoBanda_Welch_10kHz_dB.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# 6) Blackman-Tukey (en dB)
# ----------------------------
start_time = time.time()
f_bt, Pxx_bt_lin = per_smooth(audio_signal, win=2, M=1024, fs=fs, nfft=4096)
Pxx_bt_dB = 10 * np.log10(Pxx_bt_lin + 1e-12) # Convertir a dB (10*log10)
f_low_bt, f_high_bt, BW_bt, mask_bt = bandwidth(f_bt, Pxx_bt_dB) # Usar Pxx_bt_dB

plt.figure(figsize=(10, 5))
plt.plot(f_bt, Pxx_bt_dB, label='Densidad Espectral de Potencia') # Usar plt.plot
plt.fill_between(f_bt, Pxx_bt_dB, where=mask_bt, color='orange', alpha=0.3, label='Ancho de banda (5% de Potencia)')
plt.title("6. Densidad Espectral de Potencia - Blackman-Tukey - 10kHz (dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/6_DEP_BlackmanTukey_10kHz_dB.png', dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(f_bt[mask_bt], Pxx_bt_dB[mask_bt], color='orange')
plt.title(f"7. Ancho de banda - Blackman-Tukey (10kHz): {BW_bt:.2f} Hz")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/7_AnchoBanda_BlackmanTukey_10kHz_dB.png', dpi=300)
plt.show()
plt.close()

end_time = time.time()
print(f"Blackman-Tukey calculado en {end_time - start_time:.1f} segundos")

# ----------------------------
# 8) Gráfico Comparativo de DEP (en dB)
# ----------------------------
plt.figure(figsize=(10,6))
plt.plot(f_per, Pxx_per_dB, label='Periodograma (dB)', alpha=0.6, linewidth=1)
plt.plot(f_welch, Pxx_welch_dB, label='Welch (dB)', alpha=0.8, linewidth=2)
plt.plot(f_bt, Pxx_bt_dB, label='Blackman-Tukey (dB)', alpha=0.8, linewidth=2)
plt.title("8. Comparación de Estimadores de DEP (dB)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral [dBV²/Hz]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/8_Comparacion_PSD_dB.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# Tabla de anchos de banda
# ----------------------------
tabla_bw = pd.DataFrame({
    "Método": ["Periodograma", "Welch", "Blackman–Tukey"],
    "Frecuencia baja [Hz]": [f_low_per, f_low_w, f_low_bt],
    "Frecuencia alta [Hz]": [f_high_per, f_high_w, f_high_bt],
    "Ancho de banda [Hz]": [BW_per, BW_w, BW_bt]
})

print("\nResultados del ancho de banda estimado para el 10kHz:\n")
print(tabla_bw)
tabla_bw.to_csv('resultados/AnchoBanda_10kHz.csv', index=False, float_format='%.3f')
print("\n✅ Tabla guardada en 'resultados/AnchoBanda_10kHz.csv'")
print("✅ Todos los gráficos guardados en 'resultados/' y mostrados en pantalla")