import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# --- 1. Parámetros y Carga de Archivo ---
wav_file = 'CW.wav'
try:
    # Cargar el archivo WAV. fs = frecuencia de muestreo.
    fs, signal_full = wavfile.read(wav_file)

    # Si el audio es estéreo (más de 1 canal), tomamos solo el primer canal.
    if signal_full.ndim > 1:
        signal = signal_full[:, 0]
    else:
        signal = signal_full

    N = len(signal)
    
    print(f"Archivo cargado: {wav_file}")
    print(f"Frecuencia de muestreo (fs): {fs} Hz")
    print(f"Número de muestras (N): {N}")

except FileNotFoundError:
    print(f"¡Error! El archivo '{wav_file}' no se encontró.")
    exit()

# --- 2. Cálculo de la FFT ---

# Aplicar la Transformada Rápida de Fourier
fft_signal = np.fft.fft(signal)

# Calcular el espectro de magnitud (sólo la parte unilateral, hasta la Frecuencia de Nyquist)
fft_mag = np.abs(fft_signal[:N // 2])

# Crear el vector de frecuencias correspondiente
f_fft = np.linspace(0, fs / 2, N // 2)

# --- 3. Identificación del Pico Más Alto ---

# Encontrar el índice del valor de magnitud más alto dentro del espectro completo (0 a fs/2)
peak_index = np.argmax(fft_mag)

# Obtener la frecuencia y la magnitud en ese pico
peak_frequency = f_fft[peak_index]
peak_magnitude = fft_mag[peak_index]

# --- 4. Visualización de la FFT ---
plt.figure(figsize=(12, 6))

# Graficar la magnitud de la FFT (eje Y en escala logarítmica para resaltar el tono)
plt.semilogy(f_fft, fft_mag, label='Espectro de Magnitud')

# Marcar el pico más alto
plt.plot(peak_frequency, peak_magnitude, 'ro', 
         label=f'Pico: {peak_frequency:.2f} Hz', markersize=8)

# Añadir etiquetas al gráfico
plt.annotate(
    f'Frecuencia del Tono: {peak_frequency:.2f} Hz',
    xy=(peak_frequency, peak_magnitude),
    # Ajustar la posición del texto para que sea visible (a 800 Hz)
    xytext=(peak_frequency + 100, peak_magnitude * 0.9),  
    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
    fontsize=12,
    color='red'
)

plt.title("FFT del Archivo CW.wav - Detección de Frecuencia del Tono (500 Hz a 1300 Hz)")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Magnitud (Escala Logarítmica)")

# --- MODIFICACIÓN CLAVE: Limitar el eje X al rango 500 Hz a 1300 Hz ---
plt.xlim(500, 1300) 
# ----------------------------------------------------------------------

plt.grid(True, which="both", ls="--")
plt.legend()
plt.tight_layout()
plt.show()

# Imprimir el resultado en la consola
print("-" * 40)
print(f"La frecuencia del tono con el pico más alto es: {peak_frequency:.2f} Hz")
print("-" * 40)