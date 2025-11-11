# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:25:31 2025

@author: Logo
"""

# -*- coding: utf-8 -*-
"""
Filtro_FIR_Remez_CW.py - FIR (Pasa Banda)
Diseño: Parks-McClellan / Remez (Equirrizado)
VISUALIZACIÓN: Se grafica la señal completa (5.0 segundos), compensando el retardo.
"""
import numpy as np; from scipy import signal as sig; import matplotlib.pyplot as plt
from matplotlib import patches; from scipy.signal import lfilter; import scipy.io.wavfile as wav

# 1. PLANTILLA Y CARGA DE DATOS (Frecuencia de Tono: 700 Hz)
fs = 44100
N_fir = 3000; numtaps = N_fir + 1; demora = N_fir // 2 # τg = 1500 muestras
wp = (650, 750); ws = (550, 850)

# --- Carga de Datos (Cargando CW.wav - SEÑAL COMPLETA) ---
try:
    fs_read, audio_data = wav.read('CW.wav') 
    if audio_data.ndim > 1: audio_one_lead = audio_data[:, 0]
    else: audio_one_lead = audio_data
    if fs_read != fs: fs = fs_read
except Exception as e:
    audio_one_lead = np.random.randn(int(fs * 5)).astype(np.float32) 
cant_muestras = len(audio_one_lead)
# ----------------------

def zplane(z, p, title):
    fig, ax = plt.subplots(figsize=(6, 6)); unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='black', alpha=0.3); ax.add_artist(unit_circle)
    ax.plot(np.real(z), np.imag(z), 'o', markersize=9, label='Ceros'); ax.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
    ax.set_title(title); ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5]); ax.grid(True, which='both', ls=':'); ax.legend(); plt.tight_layout(); plt.show()

# 2. DISEÑO FIR REMEZ (Band-Pass)
bands = [0, ws[0], wp[0], wp[1], ws[1], fs/2]
desired = [0, 1, 0] # 0 (Stop), 1 (Pass), 0 (Stop)
b = sig.remez(numtaps=numtaps, bands=bands, desired=desired, fs=fs)
Audio_f = lfilter(b, 1, audio_one_lead) # Aplicación a la señal completa

z, p, k = sig.tf2zpk(b, 1)
zplane(z, p, title=f'Polos y Ceros - FIR Remez Pasa Banda (N={N_fir})')

# 3. EVALUACIÓN CONSOLIDADA (Visualización de la señal COMPLETA - Retardo Compensado)
# La región de graficación abarca de 0 hasta cant_muestras - demora
zoom_region = np.arange(0, cant_muestras - demora, dtype='uint')
plt.figure(figsize=(12, 6))
plt.plot(zoom_region / fs, audio_one_lead[zoom_region], label='1. Audio Original (CW + Ruido)', color='red', alpha=0.6)
plt.plot(zoom_region / fs, Audio_f[zoom_region + demora], label=f'2. Audio Filtrado (Remez - τg={demora} mues)', linewidth=2, color='blue')
plt.title(f'Señal de Telegrafía Filtrada (Fase Lineal - Duración Total)'); 
plt.xlabel('Tiempo [s]'); plt.ylabel('Amplitud'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()