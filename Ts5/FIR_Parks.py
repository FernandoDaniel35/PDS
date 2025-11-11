#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 02:03:05 2025

@author: fer
"""
# -*- coding: utf-8 -*-
"""
Filtro_FIR_Remez_Final.py - FIR (Parks-McClellan / Remez)
Diseño, verificación (Polos/Ceros) y EVALUACIÓN CONSOLIDADA (Ruido, Limpio, Filtrado) sin subplot.
Tiene Fase Lineal (τg=250), compensado para asegurar inocuidad.
"""
import numpy as np; import scipy.io as sio; from scipy import signal as sig; import matplotlib.pyplot as plt
from matplotlib import patches; from scipy.signal import lfilter

# 1. PLANTILLA Y CARGA DE DATOS
fs = 1000; wp = (0.8, 35); ws0 = 0.1; ws1 = 40; N_fir = 500; numtaps = N_fir + 1; demora = N_fir // 2 # τg = 250 muestras
try:
    mat = sio.loadmat('ECG_TP4.mat'); ecg_one_lead = np.squeeze(mat['ecg_lead'])
except FileNotFoundError: ecg_one_lead = np.random.randn(20000)
try:
    ecg_clean_lead = np.load('ecg_sin_ruido.npy')
except FileNotFoundError: ecg_clean_lead = ecg_one_lead
cant_muestras = len(ecg_one_lead)

def zplane(z, p, title):
    fig, ax = plt.subplots(figsize=(6, 6)); unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='black', alpha=0.3); ax.add_artist(unit_circle)
    ax.plot(np.real(z), np.imag(z), 'o', markersize=9, label='Ceros'); ax.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos')
    ax.set_title(title); ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5]); ax.grid(True, which='both', ls=':'); ax.legend(); plt.tight_layout(); plt.show()

# 2. DISEÑO FIR REMEZ (Consigna c)
bands = [0, ws0, wp[0], wp[1], ws1, fs/2]; desired = [0, 1, 0]
b = sig.remez(numtaps=numtaps, bands=bands, desired=desired, fs=fs)
ECG_f = lfilter(b, 1, ecg_one_lead) 
z, p, k = sig.tf2zpk(b, 1)
zplane(z, p, title=f'Polos y Ceros - FIR Remez (N={N_fir})')

# 3. EVALUACIÓN CONSOLIDADA (Consigna d) - Se compensa la demora FIR

# A. Evaluación de Atenuación 
regs_ruido = ([4000, 5500], [10000, 11000])
for ii in regs_ruido:
    zoom_region = np.arange(ii[0], ii[1], dtype='uint')
    plt.figure(figsize=(10, 5))
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='1. ECG con Ruido', color='red', alpha=0.6)
    plt.plot(zoom_region, ecg_clean_lead[zoom_region], label='2. ECG Limpio (Referencia NPY)', color='green', alpha=0.5)
    plt.plot(zoom_region, ECG_f[zoom_region + demora], label=f'3. ECG Filtrado (Remez - τg={demora})', linewidth=2, color='blue')
    plt.title(f'Evaluación de Atenuación: Región con Ruido {ii[0]}-{ii[1]}'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

# B. Evaluación de Inocuidad 
regs_inocuidad = ([1000, 2500], [15000, 16500]) # Muestras válidas
for ii in regs_inocuidad:
    start = int(np.max([0, ii[0]])); end = int(np.min([cant_muestras, ii[1]]))
    zoom_region = np.arange(start, end, dtype='uint')
    plt.figure(figsize=(10, 5))
    plt.plot(zoom_region, ecg_clean_lead[zoom_region], label='1. ECG Limpio (Referencia NPY)', color='green', alpha=0.7)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='2. ECG con Ruido', color='red', alpha=0.3)
    plt.plot(zoom_region, ECG_f[zoom_region + demora], label=f'3. ECG Filtrado (Remez - τg={demora})', linewidth=2, color='blue')
    plt.title(f'Evaluación de Inocuidad: Región sin Ruido {start}-{end} (VS Clean NPY)'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()