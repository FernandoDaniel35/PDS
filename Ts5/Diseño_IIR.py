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

# Suprime las advertencias de BadCoefficients en esta sección
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=sig.BadCoefficients) # Ignorar advertencias específicas de SciPy

    for name, mi_sos in filtros.items():
        if name not in filtros_a_mostrar:
            continue # Ignorar los filtros no seleccionados
        
        # --- Respuesta en frecuencia ---
        w, h = sig.sosfreqz(mi_sos, worN=np.logspace(np.log10(0.01), np.log10(fs/2), 1000), fs = fs) 

        # --- Cálculo de fase y retardo de grupo ---
        phase = np.unwrap(np.angle(h))
        w_rad = w / (fs/2) * np.pi
        gd = -np.diff(phase) / np.diff(w_rad)

        # --- Polos y ceros ---
        z, p, k = sig.sos2zpk(mi_sos)
        order = mi_sos.shape[0] * 2
        

        resultados[name] = {'w': w, 'h': h, 'phase': phase, 'gd': gd, 'z': z, 'p': p, 'order': order}

        # Diagrama de polos y ceros individual
        plt.figure(figsize=(6,6))
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
        plt.show()

# ----------------------------------------------------------------------------------------
# 5. Gráficas Comparativas (Magnitud, Fase, Retardo de Grupo - Solo los seleccionados)
# ----------------------------------------------------------------------------------------

plt.figure(figsize=(12, 10))
eps = 1e-8 # Piso numérico para evitar log10(0)

# Magnitud
plt.subplot(3,1,1)
for name, res in resultados.items():
    plt.plot(res['w'], 20*np.log10(abs(res['h']) + eps), label = name)
plt.title(f'Respuesta en Magnitud - Comparativa (ws={ws[0]} Hz, alpha_s={alpha_s} dB)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.ylim([-50, 5]) 
plt.grid(True, which='both', ls=':')
plt.legend()
 
# Fase
plt.subplot(3,1,2)
for name, res in resultados.items():
    plt.plot(res['w'], res['phase'], label = name)
plt.title('Fase - Comparativa')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(3,1,3)
for name, res in resultados.items():
    # El retardo de grupo tiene un elemento menos que 'w'
    plt.plot(res['w'][:-1], res['gd'], label = name) 
plt.title('Retardo de Grupo - Comparativa')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel(r'$\tau_g$ [# muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()

plt.tight_layout()
plt.show()