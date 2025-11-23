#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diseño de los filtros IIR Chebyshev I y Cauer/Elíptico.

Created on Wed Nov 12 21:48:12 2025

@author: Fernando Daniel Fiamberti
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from matplotlib import patches
import warnings
import sys
import io

# -------------------------------------------------------------
# 1. Parámetros Generales
# -------------------------------------------------------------
fs = 1000 # Hz

# === ESPECIFICACIONES ===
wp = (0.8, 35) # Frecuencia de corte/paso (Hz)
ws = (0.1, 40) # Frecuencia de stop/detenida (Hz)
alpha_p = 1 # Atenuación máxima en banda de paso (dB)
alpha_s = 40 # Atenuación mínima en banda de atenuación (dB)
# =============================================================

filtros = {}

# -------------------------------------------------------------
# 2. Diseño de los Filtros
# -------------------------------------------------------------

falso_stderr = io.StringIO()

old_stderr = sys.stderr

try:
    sys.stderr = falso_stderr
    N_butt, wn_butt = sig.buttord(wp, ws, alpha_p, alpha_s, analog=False, fs=fs)
    mi_sos_butt = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s,analog=False, ftype='butter', output='sos', fs=fs)
finally:
    sys.stderr = old_stderr
filtros['Butterworth (N=76)'] = mi_sos_butt


# --- Diseño Chebyshev Tipo I (N ~23) ---
N_cheb1, wn_cheb1 = sig.cheb1ord(wp, ws, alpha_p, alpha_s, analog=False, fs=fs)
mi_sos_cheb1 = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s,analog=False, ftype='cheby1', output='sos', fs=fs)
filtros['Chebyshev I (N=23)'] = mi_sos_cheb1


# --- Diseño Chebyshev Tipo II (N ~23) ---
N_cheb2, wn_cheb2 = sig.cheb2ord(wp, ws, alpha_p, alpha_s, analog=False, fs=fs)
mi_sos_cheb2 = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s,analog=False, ftype='cheby2', output='sos', fs=fs)
filtros['Chebyshev II (N=23)'] = mi_sos_cheb2


# --- Diseño Cauer/Elíptico (Estable, N=12) ---
N_cauer, wn_cauer = sig.ellipord(wp, ws, alpha_p, alpha_s, analog=False, fs=fs)
mi_sos_cauer = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s,analog=False, ftype='ellip', output='sos', fs=fs)
filtros['Cauer/Elliptic (N=12)'] = mi_sos_cauer

# -------------------------------------------------------------
# 3. FILTRADO: Seleccionar solo Chebyshev I y Cauer/Elliptic
# -------------------------------------------------------------
filtros_a_mostrar = ['Chebyshev I (N=23)', 'Cauer/Elliptic (N=12)']

# -------------------------------------------------------------
# 4. Análisis de Respuesta y Polos/Ceros (Solo filtros seleccionados)
# -------------------------------------------------------------
resultados = {}

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=sig.BadCoefficients) # Ignorar advertencias específicas de SciPy

    for name, mi_sos in filtros.items():
        if name not in filtros_a_mostrar:
            continue # Ignorar los filtros no seleccionados
        
        # --- Respuesta en frecuencia ---
        w, h = sig.sosfreqz(mi_sos, worN=np.logspace(np.log10(0.01), np.log10(fs/2), 10000), fs = fs)

        # --- Cálculo de fase y retardo de grupo ---
        phase = np.unwrap(np.angle(h))
        w_rad = w / (fs/2) * np.pi
        gd = -np.diff(phase) / np.diff(w_rad)

        # --- Polos y ceros ---
        z, p, k = sig.sos2zpk(mi_sos)
        order = mi_sos.shape[0] * 2
        

        resultados[name] = {'w': w, 'h': h, 'phase': phase, 'gd': gd, 'z': z, 'p': p, 'order': order}

        # Diagrama de polos y ceros individual
        plt.figure(figsize=(3,3)) 
        plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos' )
        if len(z) > 0:
            plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
        
        axes_hdl = plt.gca()
        unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
        axes_hdl.add_patch(unit_circle)

        plt.axhline(0, color='k', lw=0.5)
        plt.axvline(0, color='k', lw=0.5)
        plt.axis([-1.1, 1.1, -1.1, 1.1])
        plt.title(f'Polos y Ceros - {name} (N={order})')
        plt.xlabel(r'$\Re(z)$')
        plt.ylabel(r'$\Im(z)$')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()


# ----------------------------------------------------------------------------------------
# 5. Gráficas Comparativas (Magnitud, Fase, Retardo de Grupo - Solo los seleccionados)
# ----------------------------------------------------------------------------------------

fig = plt.figure(figsize=(12, 20)) # Altura ajustada para 8 unidades de cuadrícula
gs = fig.add_gridspec(8, 1) # 8 filas (2+2+2+1+1)

eps = 1e-8 # Piso numérico para evitar log10(0)

# 1. Magnitud General (Doble de altura: filas 0 y 1)
ax1 = fig.add_subplot(gs[0:2, 0])
for name, res in resultados.items():
    ax1.plot(res['w'], 20*np.log10(abs(res['h']) + eps), label = name)
ax1.set_title(f'Respuesta en Magnitud - Comparativa (ws={ws[0]} Hz, alpha_s={alpha_s} dB)')
ax1.set_xlabel('Frecuencia [Hz]')
ax1.set_ylabel('|H(jω)| [dB]')
ax1.set_ylim([-50, 5]) 
ax1.set_xlim([0, 41]) # Limita la vista del eje X de 0 a 41 Hz
ax1.grid(True, which='both', ls=':')
ax1.legend()
 
# 2. Magnitud Zoom 0.65 a 0.9 Hz 
ax2 = fig.add_subplot(gs[2:4, 0])
for name, res in resultados.items():
    ax2.plot(res['w'], 20*np.log10(abs(res['h']) + eps), label = name)
ax2.set_title('Magnitud (Zoom 0.65 a 0.9 Hz)')
ax2.set_xlabel('Frecuencia [Hz]')
ax2.set_ylabel('|H(jω)| [dB]')
ax2.set_ylim([-50, 5])
ax2.set_xlim([0.65, 0.9]) # Limita a 0.65-0.9 Hz
ax2.grid(True, which='both', ls=':')

# 3. Magnitud Zoom 33 a 41 Hz 
ax3 = fig.add_subplot(gs[4:6, 0])
for name, res in resultados.items():
    ax3.plot(res['w'], 20*np.log10(abs(res['h']) + eps), label = name)
ax3.set_title('Magnitud (Zoom 33 a 41 Hz)')
ax3.set_xlabel('Frecuencia [Hz]')
ax3.set_ylabel('|H(jω)| [dB]')
ax3.set_ylim([-50, 5])
ax3.set_xlim([33, 41]) # Limita a 33-41 Hz
ax3.grid(True, which='both', ls=':')


# 4. Fase 
ax4 = fig.add_subplot(gs[6, 0])
for name, res in resultados.items():
    ax4.plot(res['w'], res['phase'], label = name)
ax4.set_title('Fase - Comparativa')
ax4.set_xlabel('Frecuencia [Hz]')
ax4.set_ylabel('Fase [rad]')
ax4.grid(True, which='both', ls=':')
ax4.legend()

# 5. Retardo de grupo 
ax5 = fig.add_subplot(gs[7, 0])
for name, res in resultados.items():
    ax5.plot(res['w'][:-1], res['gd'], label = name) 
ax5.set_title('Retardo de Grupo - Comparativa')
ax5.set_xlabel('Frecuencia [Hz]')
ax5.set_ylabel(r'$\tau_g$ [# muestras]')
ax5.grid(True, which='both', ls=':')
ax5.legend()


plt.tight_layout()
plt.show() # Única llamada a show() para mostrar todas las figuras pendientes.