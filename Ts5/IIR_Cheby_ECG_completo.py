#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 20:19:05 2025

@author: fer
"""

# -*- coding: utf-8 -*-
"""
Filtro_IIR_Chebyshev1.py - IIR (Chebyshev Tipo I)
Diseño, verificación y aplicación de filtro Chebyshev I IIR Pasa-Banda.
Aplicado a la señal ECG completa.
"""

import numpy as np
import scipy.io as sio
from scipy import signal as sig
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.signal import sosfiltfilt

# =========================================================================
# 1. PLANTILLA DE DISEÑO Y CARGA DE DATOS (fs=1000 Hz)
# =========================================================================
fs = 1000 # Hz
wp = (0.8, 35)  # Banda de Paso (Hz)
ws = (0.1, 40) # Bandas de Detención (Hz)
alpha_p = 1  # Rizado máx. Banda de Paso (dB)
alpha_s = 40  # Atenuación mín. Banda de Detención (dB)

try:
    mat = sio.loadmat('ECG_TP4.mat')
    ecg_one_lead = np.squeeze(mat['ecg_lead']) 
except FileNotFoundError:
    print("Advertencia: Archivo ECG_TP4.mat no encontrado. Usando datos simulados.")
    ecg_one_lead = np.random.randn(15000)
cant_muestras = len(ecg_one_lead)

# Función auxiliar para Polos y Ceros
def zplane(z, p, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='black', alpha=0.3)
    ax.add_artist(unit_circle)
    ax.plot(np.real(z), np.imag(z), 'o', markersize=9, label='Ceros')
    ax.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
    ax.set_title(title); ax.set_xlabel('Real'); ax.set_ylabel('Imaginario')
    ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5]); ax.grid(True, which='both', ls=':'); ax.legend()
    plt.tight_layout(); plt.show()

# =========================================================================
# 2. DISEÑO Y VERIFICACIÓN (Consigna c)
# =========================================================================
f_aprox = 'cheby1'
N, wn = sig.cheb1ord(wp, ws, alpha_p, alpha_s, analog=False, fs=fs)
sos = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s, ftype=f_aprox, output='sos', fs=fs)
w, h = sig.sosfreqz(sos, worN=8000, fs=fs)
w_rad = w / (fs/2) * np.pi
phase = np.unwrap(np.angle(h))
gd = -np.diff(phase) / np.diff(w_rad)

# Polos y Ceros
z, p, k = sig.sos2zpk(sos)
# LÍNEA CORREGIDA: Se usa sos.shape[0] * 2 para el orden N
zplane(z, p, title=f'Polos y Ceros - {f_aprox.capitalize()} I (N={sos.shape[0] * 2})')

# Gráficas de Respuesta en Frecuencia
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1); plt.plot(w, 20 * np.log10(abs(h))); plt.title(f'Respuesta en Magnitud - {f_aprox.capitalize()} I'); plt.ylabel('|H(jω)| [dB]'); plt.ylim([-50, 5]); plt.grid(True, which='both', ls=':')
plt.subplot(3, 1, 2); plt.plot(w, phase); plt.title('Fase'); plt.ylabel('Fase [rad]'); plt.grid(True, which='both', ls=':')
plt.subplot(3, 1, 3); plt.plot(w[:-1], gd); plt.title('Retardo de Grupo'); plt.xlabel('Frecuencia [Hz]'); plt.ylabel('τg [# muestras]'); plt.grid(True, which='both', ls=':')
plt.tight_layout(); plt.show()

# Aplicación de Ejemplo (Señal completa)
ECG_f = sosfiltfilt(sos, ecg_one_lead) 
tiempo = np.arange(cant_muestras) / fs
plt.figure(figsize=(10, 4))
plt.plot(tiempo, ecg_one_lead, label='ECG Original (Completo)', alpha=0.5)
plt.plot(tiempo, ECG_f, label='ECG Filtrado (Chebyshev I - Fase Cero)', linewidth=2)
plt.title('ECG Filtrado (Chebyshev I) - Señal Completa'); plt.xlabel('Tiempo [s]'); plt.legend(); plt.tight_layout(); plt.show()