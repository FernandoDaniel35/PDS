#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 14:57:11 2025

@author: fer
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from matplotlib import patches
import sys



fs = 1000 # Hz

  # -------------------------------------------------------------
  # 2. Diseño del Filtro
  # -------------------------------------------------------------

wp = (0.8  , 35 )  # frecuencia de corte/paso (Hz)
ws = (0.1, 40 ) # frecuencia de stop/detenida (Hz)

# divido por dos porque pasaré dos veces por este filtro.
alpha_p = 1  # atenuación máxima a la wp, alfa_max, pérdidas en banda de paso
alpha_s = 40  # atenuación mínima a la ws, alfa_min, mínima atenuación requerida
             # en banda de paso

# Aprox módulo
f_aprox= 'butter'

# --- Diseño del filtro analógico ---
mi_sos_butt = sig.iirdesign(wp = wp, ws = ws, gpass=alpha_p, gstop=alpha_s, 
                        analog=False, ftype= f_aprox, output='sos', fs = fs )

f_aprox= 'cheby1'

mi_sos_cheb1 = sig.iirdesign(wp = wp, ws = ws, gpass=alpha_p, gstop=alpha_s, 
                        analog=False, ftype= f_aprox, output='sos', fs = fs )

f_aprox= 'cheby2'

mi_sos_cheb2 = sig.iirdesign(wp = wp, ws = ws, gpass=alpha_p, gstop=alpha_s, 
                        analog=False, ftype= f_aprox, output='sos', fs = fs )

f_aprox= 'cauer'

mi_sos_cauer = sig.iirdesign(wp = wp, ws = ws, gpass=alpha_p, gstop=alpha_s, 
                        analog=False, ftype= f_aprox, output='sos', fs = fs )


mi_sos = mi_sos_cauer

# --- Respuesta en frecuencia ---
w, h = sig.freqz_sos(mi_sos, worN=np.logspace(-2, 1.9, 1000), fs = fs)  # 10 Hz a 1 MHz aprox.
# w, h = sig.freqz_sos(mi_sos, fs = fs)  # Calcula la respuesta en frecuencia del filtro

# --- Cálculo de fase y retardo de grupo ---
phase = np.unwrap(np.angle(h))
# Retardo de grupo = -dφ/dω
w_rad = w / (fs/2) * np.pi
gd = -np.diff(phase) / np.diff(w_rad)

# --- Polos y ceros ---
z, p, k = sig.sos2zpk(mi_sos)

# --- Gráficas ---
plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(3,1,1)
plt.plot(w, 20*np.log10(abs(h)), label = f_aprox)
plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Fase
plt.subplot(3,1,2)
plt.plot(w, phase, label = f_aprox)
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(3,1,3)
plt.plot(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()


# Diagrama de polos y ceros
plt.figure(figsize=(10,10))
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos' )
axes_hdl = plt.gca()

if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
axes_hdl.add_patch(unit_circle)

plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel(r'$\Re(z)$')
plt.ylabel(r'$\Im(z)$')
plt.legend()
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()