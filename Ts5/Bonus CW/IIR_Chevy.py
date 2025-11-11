# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:20:31 2025

@author: Logo
"""

"""
Filtro_IIR_Chebyshev1_CW.py - IIR (Pasa Banda)
Diseño: Chebyshev Tipo I (Rizado en banda de paso)
VISUALIZACIÓN: Se grafica la señal completa (5.0 segundos).
"""
import numpy as np; 
from scipy import signal as sig; 
import matplotlib.pyplot as plt
from matplotlib import patches; 
from scipy.signal import sosfiltfilt; 
import scipy.io.wavfile as wav

# 1. PLANTILLA Y CARGA DE DATOS (Frecuencia de Tono: 700 Hz)
fs = 44100
wp = (650, 750)
ws = (550, 850)
alpha_p = 1; alpha_s = 40

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

# 2. DISEÑO IIR CHEBYSHEV I (Band-Pass)
sos = sig.iirdesign(wp=wp, ws=ws, gpass=alpha_p, gstop=alpha_s, ftype='cheby1', 
                    output='sos', fs=fs)
Audio_f = sosfiltfilt(sos, audio_one_lead) # Aplicación con Fase Cero

z, p, k = sig.sos2zpk(sos)
zplane(z, p, title=f'Polos y Ceros - Chebyshev I IIR Pasa Banda (N={len(sos) * 2})')

# 3. EVALUACIÓN CONSOLIDADA (Visualización de la señal COMPLETA)
zoom_region = np.arange(0, cant_muestras, dtype='uint')
plt.figure(figsize=(12, 6))
plt.plot(zoom_region / fs, audio_one_lead[zoom_region], label='1. Audio Original (CW + Ruido)', color='red', alpha=0.6)
plt.plot(zoom_region / fs, Audio_f[zoom_region], label='2. Audio Filtrado (Pasa Banda Chebyshev I)', linewidth=2, color='blue')
plt.title(f'Señal de Telegrafía Filtrada (Fase Cero - Duración Total)'); 
plt.xlabel('Tiempo [s]'); 
plt.ylabel('Amplitud'); 
plt.legend(); plt.grid(True)
plt.tight_layout(); 
plt.show()