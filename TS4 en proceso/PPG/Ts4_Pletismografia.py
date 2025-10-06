#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Densidad Espectral de Potencia de la señal de Pletismografía - gráficos separados con ancho de banda resaltado
y cantidad de muestras ajustada para ~2 minutos de cálculo

Incluye gráficos adicionales:
- Señal original en el tiempo
- FFT de la señal original
- Gráficos de ancho de banda solo, separados por método
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, welch, windows
import pandas as pd
import os
import time

# ----------------------------
# Cargar archivo CSV
# ----------------------------
csv_file = 'PPG.csv'  # archivo de muestra
df = pd.read_csv(csv_file)

# Supongamos que la columna de datos se llama 'Pletismografía' o 'Signal', si no, tomamos la primera columna
if 'Pletismografía' in df.columns:
    signal_full = df['Pletismografía'].values
elif 'Signal' in df.columns:
    signal_full = df['Signal'].values
else:
    signal_full = df.iloc[:,0].values  # primera columna

fs = 1000  # Hz, mantener mismo valor por compatibilidad con código original

# ----------------------------
# Ajustar cantidad de muestras para cálculo (~2 minutos)
# ----------------------------
N_samples = 15000
pletismografia = signal_full[:N_samples]
N = len(pletismografia)
print(f"Usando {N} muestras de Pletismografía para el análisis (~2 min cálculo)")

# ----------------------------
# Crear carpeta resultados si no existe
# ----------------------------
os.makedirs('resultados', exist_ok=True)

# ----------------------------
# Función ancho de banda y rango (sin cambios)
# ----------------------------
def bandwidth(f, Pxx, threshold=0.05):
    Pxx_norm = Pxx / np.max(Pxx)
    mask = Pxx_norm > threshold
    if np.any(mask):
        f_bw = f[mask]
        return f_bw[0], f_bw[-1], f_bw[-1] - f_bw[0], mask
    else:
        return 0, 0, 0, mask

# ----------------------------
# Función Blackman-Tukey modular (MODIFICADA)
# ----------------------------
def per_smooth(x, win=1, M=1024, fs=1000, nfft=4096):
    """
    Implementación del método Blackman-Tukey para estimar la PSD.
    MODIFICADO: Devuelve el espectro unilateral (solo frecuencias positivas)
    
    Parámetros:
    - x: señal de entrada
    - win: tipo de ventana (1=rectangular, 2=Hamming, 3=Hanning, 4=Bartlett, 5=Blackman)
    - M: número de retardos para autocorrelación
    - fs: frecuencia de muestreo
    - nfft: número de puntos FFT
    Retorna:
    - f: vector de frecuencias (0 a fs/2)
    - Pxx: PSD estimada unilateral
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
    
    # --- MODIFICACIÓN CLAVE: Espectro Unilateral ---
    
    # 1. Calcular el número de puntos no redundantes (hasta Nyquist)
    n_unique = nfft // 2 + 1
    
    # 2. Tomar la primera mitad y escalar para PSD Unilateral
    Pxx_uni = Pxx_full[:n_unique]
    
    # 3. Aplicar escalado para "densidad de potencia" (multiplicar por 2, excepto 0 y Nyquist)
    # Nota: La implementación Blackman-Tukey no siempre usa el factor 1/(fs*N) de forma directa
    # como en periodogram. Para que sea comparable, se escala la parte unilateral.
    Pxx = Pxx_uni.copy()
    Pxx[1:-1] *= 2 # Multiplicar por 2 los puntos intermedios
    
    # 4. Vector de frecuencias para el espectro unilateral
    f = np.linspace(0, fs / 2, n_unique)

    return f, Pxx


# ----------------------------
# 0) Señal original en el tiempo
# ----------------------------
plt.figure(figsize=(10,4))
plt.plot(np.arange(N)/fs, pletismografia, color='blue')
plt.title("Señal de Pletismografía en el tiempo")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/Pletismografia_original_tiempo.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# FFT de la señal original
# ----------------------------
fft_signal = np.fft.fft(pletismografia, n=N)
f_fft = np.linspace(0, fs/2, N//2)
fft_mag = np.abs(fft_signal[:N//2])

plt.figure(figsize=(10,4))
plt.semilogy(f_fft, fft_mag)
plt.title("FFT de la señal de Pletismografía")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/FFT_Pletismografia.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# 1) Periodograma ventaneado
# ----------------------------
ventana = windows.hamming(N)
f_per, Pxx_per = periodogram(pletismografia, fs=fs, window=ventana, nfft=N, scaling='density')
f_low_per, f_high_per, BW_per, mask_per = bandwidth(f_per, Pxx_per)

plt.figure(figsize=(10,5))
plt.semilogy(f_per, Pxx_per, label='Densidad Espectral de Potencia')
plt.fill_between(f_per, Pxx_per, where=mask_per, color='orange', alpha=0.3, label='Ancho de banda')
plt.title("Densidad Espectral de Potencia - Periodograma Ventaneado (Hamming)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/Densidad_Espectral_de_Potencia_periodograma.png', dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(10,5))
plt.plot(f_per[mask_per], Pxx_per[mask_per], color='orange')
plt.title("Ancho de banda - Periodograma")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/AnchoBanda_Densidad_Espectral_de_Potencia_periodograma.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# 2) Welch
# ----------------------------
f_welch, Pxx_welch = welch(pletismografia, fs=fs, window='hamming', nperseg=2048, noverlap=1024, scaling='density')
f_low_w, f_high_w, BW_w, mask_w = bandwidth(f_welch, Pxx_welch)

plt.figure(figsize=(10,5))
plt.semilogy(f_welch, Pxx_welch, label='Densidad Espectral de Potencia')
plt.fill_between(f_welch, Pxx_welch, where=mask_w, color='orange', alpha=0.3, label='Ancho de banda')
plt.title("Densidad Espectral de Potencia - Método de Welch")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/Densidad_Espectral_de_Potencia_welch.png', dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(10,5))
plt.plot(f_welch[mask_w], Pxx_welch[mask_w], color='orange')
plt.title("Ancho de banda - Welch")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/AnchoBanda_Densidad_Espectral_de_Potencia_welch.png', dpi=300)
plt.show()
plt.close()

# ----------------------------
# 3) Blackman-Tukey usando función modular
# ----------------------------
start_time = time.time()
f_bt, Pxx_bt = per_smooth(pletismografia, win=2, M=1024, fs=fs, nfft=4096)
f_low_bt, f_high_bt, BW_bt, mask_bt = bandwidth(f_bt, Pxx_bt)

plt.figure(figsize=(10,5))
plt.semilogy(f_bt, Pxx_bt, label='Densidad Espectral de Potencia')
plt.fill_between(f_bt, Pxx_bt, where=mask_bt, color='orange', alpha=0.3, label='Ancho de banda')
plt.title("Densidad Espectral de Potencia - Blackman-Tukey")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/Densidad_Espectral_de_Potencia_blackman_tukey.png', dpi=300)
plt.show()
plt.close()

plt.figure(figsize=(10,5))
plt.plot(f_bt[mask_bt], Pxx_bt[mask_bt], color='orange')
plt.title("Ancho de banda - Blackman-Tukey")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Densidad espectral")
plt.grid(True)
plt.tight_layout()
plt.savefig('resultados/AnchoBanda_Densidad_Espectral_de_Potencia_blackman_tukey.png', dpi=300)
plt.show()
plt.close()

end_time = time.time()
print(f"Blackman-Tukey calculado en {end_time - start_time:.1f} segundos")

# ----------------------------
# Tabla de anchos de banda
# ----------------------------
tabla_bw = pd.DataFrame({
    "Método": ["Periodograma", "Welch", "Blackman–Tukey"],
    "Frecuencia baja [Hz]": [f_low_per, f_low_w, f_low_bt],
    "Frecuencia alta [Hz]": [f_high_per, f_high_w, f_high_bt],
    "Ancho de banda [Hz]": [BW_per, BW_w, BW_bt]
})

print("\nResultados del ancho de banda estimado:\n")
print(tabla_bw)
tabla_bw.to_csv('resultados/ancho_de_banda_Densidad_Espectral_de_Potencia.csv', index=False, float_format='%.3f')
print("\n✅ Tabla guardada en 'resultados/ancho_de_banda_Densidad_Espectral_de_Potencia.csv'")
print("✅ Todos los gráficos guardados en 'resultados/' y mostrados en pantalla")
