#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 20:24:40 2025

@author: fer
"""

# -*- coding: utf-8 -*-
"""
Filtro_FIR_Remez.py - FIR (Parks-McClellan / Remez)
Diseño, verificación y aplicación de filtro FIR Pasa-Banda usando Remez.
Aplicado a la señal ECG completa.
"""

import numpy as np
import scipy.io as sio
from scipy import signal as sig
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.signal import lfilter

# =========================================================================
# 1. PLANTILLA DE DISEÑO Y CARGA DE DATOS (fs=1000 Hz)
# =========================================================================
fs = 1000 # Hz
wp = (0.8, 35)  # Banda de Paso (Hz)
ws0 = 0.1  # Banda de Detención Baja (Hz)
ws1 = 40   # Banda de Detención Alta (Hz)
N_fir = 500  # Orden del filtro FIR
numtaps = N_fir + 1 
demora = N_fir // 2 # Retardo de grupo teórico

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
bands = [0, ws0, wp[0], wp[1], ws1, fs/2] # Frecuencias de banda
desired = [0, 1, 0]                       # Magnitud deseada (Detención, Paso, Detención)
b = sig.remez(numtaps=numtaps, bands=bands, desired=desired, fs=fs)

w, h = sig.freqz(b, worN=8000, fs=fs)
phase = np.unwrap(np.angle(h))

# Polos y Ceros
z, p, k = sig.tf2zpk(b, 1)
zplane(z, p, title=f'Polos y Ceros - FIR Remez (N={N_fir})')

# Gráficas de Respuesta en Frecuencia
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1); plt.plot(w, 20 * np.log10(abs(h))); plt.title(f'Respuesta en Magnitud - FIR Remez'); plt.ylabel('|H(jω)| [dB]'); plt.ylim([-50, 5]); plt.grid(True, which='both', ls=':')
plt.subplot(3, 1, 2); plt.plot(w, phase); plt.title('Fase'); plt.ylabel('Fase [rad]'); plt.grid(True, which='both', ls=':')
plt.subplot(3, 1, 3); plt.axhline(demora, color='orange', linestyle='--'); plt.title('Retardo de Grupo'); plt.xlabel('Frecuencia [Hz]'); plt.ylabel('τg [# muestras]'); plt.grid(True, which='both', ls=':')
plt.tight_layout(); plt.show()

# Aplicación de Ejemplo (Señal completa - Retardo Compensado)
ECG_f = lfilter(b, 1, ecg_one_lead) 
tiempo_original = np.arange(cant_muestras) / fs
tiempo_filtrado = np.arange(cant_muestras - demora) / fs
plt.figure(figsize=(10, 4))
plt.plot(tiempo_original, ecg_one_lead, label='ECG Original (Completo)', alpha=0.5)
plt.plot(tiempo_filtrado, ECG_f[demora:], 
        label=f'ECG Filtrado (Remez - Demora={demora})', linewidth=2)
plt.title('ECG Filtrado (FIR Remez) - Señal Completa'); plt.xlabel('Tiempo [s]'); plt.legend(); plt.tight_layout(); plt.show()