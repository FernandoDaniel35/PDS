# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 15:57:04 2025

@author: Logo
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from matplotlib import patches
import sys

# --- Parámetros Generales ---
fs = 1000 # Hz (Frecuencia de muestreo)
N = 3000   # Orden del filtro (Número de coeficientes - 1)
numtaps = N + 1 # Número de coeficientes/taps

wp = (0.8  , 35 )  # Frecuencia de corte/paso (Hz)
#-----------------------------
# PREDISTORSION
#-----------------------------
delta0 = 0
delta1 = 0.1

ws0 = 0.1 
ws1 = 35.7 

ws = (ws0, ws1 )    # Frecuencia de stop/detenida (Hz)
#-----------------------------


# -------------------------------------------------------------
# Diseño del Filtro FIR (con ventana Rectangular -> Boxcar)
# -------------------------------------------------------------

# 1. Definir la frecuencia y magnitud de la respuesta ideal deseada (en Hz)
f_deseada = [
    0,            # DC
    ws[0] + delta1,        # Frecuencia de inicio de banda de detención baja
    wp[0],        # Frecuencia de inicio de banda de paso
    wp[1],        # Frecuencia de fin de banda de paso
    ws[1] - delta1,        # Frecuencia de inicio de banda de detención alta
    fs / 2        # Frecuencia de Nyquist
]

# 2. Definir la magnitud deseada para cada frecuencia
m_deseada = [
    0,            # Magnitud en banda de detención baja (0 a 0.1 Hz)
    0,
    1,            # Magnitud en banda de paso (0.8 a 35 Hz)
    1,
    0,            # Magnitud en banda de detención alta (40 Hz a Nyquist)
    0
]

# b es el arreglo de coeficientes (respuesta al impulso)
b = sig.firwin2(numtaps=numtaps, freq=f_deseada, gain=m_deseada, fs=fs, window='boxcar')

# --- Respuesta en frecuencia (y Fase) ---
w, H = sig.freqz(b, a=1, worN=np.logspace(-2,1.9,2000), fs=fs) # a=1 para filtros FIR

# --- Cálculo de fase y retardo de grupo ---
phase = np.unwrap(np.angle(H))

# Retardo de grupo (constante para FIR de fase lineal)
# Retardo de Grupo = (N/2) muestras
gd_esperado = N / 2 

w_rad = w / (fs/2) * np.pi
gd = -np.diff(phase) / np.diff(w_rad)


# --- Polos y ceros ---
z = np.roots(b)
p = np.zeros(N) # N polos en el origen


# -------------------------------------------------------------
# 3. Gráficas
# -------------------------------------------------------------
plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(3,1,1)
plt.plot(w, 20*np.log10(abs(H)), label = f'FIR (N={N})')
plt.title('Respuesta en Magnitud (Filtro FIR)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')

# Dibujar las bandas requeridas
plt.axvspan(0, ws[0], color='red', alpha=0.1)     # Banda de Detención Baja
plt.axvspan(wp[0], wp[1], color='green', alpha=0.1) # Banda de Paso
plt.axvspan(ws[1], fs/2, color='red', alpha=0.1)   # Banda de Detención Alta

plt.grid(True, which='both', ls=':')
plt.legend()


# Fase
plt.subplot(3,1,2)
plt.plot(w, phase, label = f'FIR (N={N})')
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.grid(True, which='both', ls=':')
plt.legend()


# Retardo de grupo
plt.subplot(3,1,3)
# Se grafica el retardo de grupo teórico (constante)
plt.axhline(gd_esperado, color='orange', linestyle='--', label=f'τg teórico={gd_esperado:.1f} muestras')
# También se puede graficar el calculado (generalmente ruidoso)
# plt.plot(w[:-1], gd, label = f'τg calculado') 
plt.title('Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()


# Diagrama de polos y ceros
plt.figure(figsize=(10,10))
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos (en z=0)' )
axes_hdl = plt.gca()

if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label='Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
axes_hdl.add_patch(unit_circle)

plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.title('Diagrama de Polos y Ceros (Filtro FIR)')
plt.xlabel(r'$\Re(z)$')
plt.ylabel(r'$\Im(z)$')
plt.legend()
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()