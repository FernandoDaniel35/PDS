#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diseño de dos filtros FIR: Ventana Rectangular (Boxcar) y Remez/Parks-McClellan.

Created on Wed Nov 12 21:57:34 2025

@author: Fernando Daniel Fiamberti
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from matplotlib import patches
import sys

# --- Parámetros Generales ---
fs = 1000 # Hz (Frecuencia de muestreo)
N = 3000    # Orden del filtro (Número de coeficientes - 1)
numtaps = N + 1 # Número de coeficientes/taps

wp = (0.8 , 35 )    # Frecuencia de corte/paso (Hz)
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
# 1. Definición de la Plantilla y Bandas (Común a ambos filtros)
# -------------------------------------------------------------
# Plantilla de sig.firwin2 (solo se usa para Boxcar)
f_deseada = [
    0,              
    ws[0] + delta1,     
    wp[0],          
    wp[1],          
    ws[1] - delta1,     
    fs / 2          
]

m_deseada = [
    0, 0, 1, 1, 0, 0
]

# Bandas para sig.remez (necesita las frecuencias de transición en pares)
# El diseño remez usa la frecuencia normalizada (0 a fs/2), pero podemos usar fs=fs
f_remez = [0, ws[0], wp[0], wp[1], ws[1], fs/2]

# Magnitudes deseadas para remez (Bandas de detención 0, Banda de paso 1)
a_remez = [0, 1, 0]  


# -------------------------------------------------------------
# 2. Diseño del Filtro FIR 1 (Ventana Rectangular -> Boxcar)
# -------------------------------------------------------------
# b_boxcar es el arreglo de coeficientes
b_boxcar = sig.firwin2(numtaps=numtaps, freq=f_deseada, gain=m_deseada, fs=fs, window='boxcar')

# --- Respuesta en frecuencia (Boxcar) ---
w, H_boxcar = sig.freqz(b_boxcar, a=1, worN=np.logspace(-2,np.log10(fs/2),10000), fs=fs)

# --- Cálculo de fase y retardo de grupo (Boxcar) ---
phase_boxcar = np.unwrap(np.angle(H_boxcar))
gd_esperado = N / 2  
w_rad = w / (fs/2) * np.pi
gd_boxcar = -np.diff(phase_boxcar) / np.diff(w_rad)

# --- Polos y ceros (Boxcar) ---
z_boxcar = np.roots(b_boxcar)
p_boxcar = np.zeros(N)


# -------------------------------------------------------------
# 3. Diseño del Filtro FIR 2 (Remez/Parks-McClellan)
# -------------------------------------------------------------
# El diseño Remez utiliza el algoritmo equirriple, que optimiza el rizado.
# Nota: La función remez espera las bandas en pares [banda_detencion, banda_paso, banda_detencion...]
b_remez = sig.remez(numtaps=numtaps, bands=f_remez, desired=a_remez, fs=fs, type='bandpass')

# --- Respuesta en frecuencia (Remez) ---
w_r, H_remez = sig.freqz(b_remez, a=1, worN=np.logspace(-2,np.log10(fs/2),10000), fs=fs)

# --- Cálculo de fase y retardo de grupo (Remez) ---
phase_remez = np.unwrap(np.angle(H_remez))
gd_remez = -np.diff(phase_remez) / np.diff(w_rad)  

# --- Polos y ceros (Remez) ---
z_remez = np.roots(b_remez)
p_remez = np.zeros(N)


# -------------------------------------------------------------
# 4. Diagramas de Polos y Ceros
# -------------------------------------------------------------

# --- Función auxiliar para graficar Polos/Ceros ---
def plot_pz(z, p, title, order):
    plt.figure(figsize=(2.5,2.5))  # Tamaño de figura reducido a la mitad (2.5, 2.5)
    plt.plot(np.real(p), np.imag(p), 'x', markersize=0.5, label='Polos (en z=0)' )  
    axes_hdl = plt.gca()
    if len(z) > 0:
        plt.plot(np.real(z), np.imag(z), 'o', markersize=0.5, fillstyle='none', label='Ceros')  
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
    axes_hdl.add_patch(unit_circle)
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.title(f'Diagrama de Polos y Ceros ({title}, N={order})')
    plt.xlabel(r'$\Re(z)$')
    plt.ylabel(r'$\Im(z)$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_pz(z_boxcar, p_boxcar, 'FIR Boxcar', N)
plot_pz(z_remez, p_remez, 'FIR Remez', N)


# -------------------------------------------------------------
# 5. Gráficas Comparativas
# -------------------------------------------------------------
plt.figure(figsize=(12,23)) 

eps = 1e-8  

# Magnitud Principal (ahora ocupa 2 filas)
plt.subplot(8,1,(1,2)) 
plt.plot(w, 20*np.log10(abs(H_boxcar) + eps), label = f'FIR Boxcar (N={N})', color='blue')
plt.plot(w_r, 20*np.log10(abs(H_remez) + eps), label = f'FIR Remez (N={N}, Equirriple)', color='red', linestyle='--')
plt.title('Respuesta en Magnitud - Comparativa FIR')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')

# Dibujar las bandas requeridas (Regiones de la plantilla)
plt.axvspan(0, ws[0], color='gray', alpha=0.1)      
plt.axvspan(wp[0], wp[1], color='green', alpha=0.1)
plt.axvspan(ws[1], fs/2, color='gray', alpha=0.1)    

plt.ylim([-80, 5])  
plt.xlim([0, 41])  

plt.grid(True, which='both', ls=':')
plt.legend()

# Detalle de 0.65Hz a 0.9Hz
plt.subplot(8,1,(3,4)) 
plt.plot(w, 20*np.log10(abs(H_boxcar) + eps), label = f'FIR Boxcar (N={N})', color='blue')
plt.plot(w_r, 20*np.log10(abs(H_remez) + eps), label = f'FIR Remez (N={N}, Equirriple)', color='red', linestyle='--')
plt.title('Detalle de Magnitud (0Hz a 0.9Hz)') # Título actualizado
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.xlim([0, 0.9]) # Rango específico de 0Hz a 0.9Hz
plt.ylim([-80, 5]) 
plt.axvspan(ws[0], ws[0] + delta1, color='gray', alpha=0.1) # Resaltar transición
plt.axvspan(wp[0], wp[1], color='green', alpha=0.1) # Resaltar banda de paso
plt.grid(True, which='both', ls=':')
plt.legend()


# Detalle de 33Hz a 41Hz
plt.subplot(8,1,(5,6)) 
plt.plot(w, 20*np.log10(abs(H_boxcar) + eps), label = f'FIR Boxcar (N={N})', color='blue')
plt.plot(w_r, 20*np.log10(abs(H_remez) + eps), label = f'FIR Remez (N={N}, Equirriple)', color='red', linestyle='--')
plt.title('Detalle de Magnitud (33Hz a 41Hz)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.xlim([33, 41]) # Rango específico
plt.ylim([-80, 5]) #
plt.axvspan(wp[1] - delta1, ws[1], color='gray', alpha=0.1) # Resaltar transición
plt.grid(True, which='both', ls=':')
plt.legend()

 
# Fase 
plt.subplot(8,1,7) 
plt.plot(w, phase_boxcar, label = 'FIR Boxcar', color='blue')
plt.plot(w_r, phase_remez, label = 'FIR Remez', color='red', linestyle='--')
plt.title('Fase - Comparativa FIR (Fase Lineal)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.xlim([0, 41]) 
plt.grid(True, which='both', ls=':')
plt.legend()


# Retardo de grupo 
plt.subplot(8,1,8) 
# El retardo de grupo es el mismo N/2 para ambos filtros de fase lineal
plt.axhline(gd_esperado, color='orange', linestyle='-', label=f'τg teórico={gd_esperado:.1f} muestras (Ambos)')
plt.title('Retardo de Grupo (Fase Lineal)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel(r'$\tau_g$ [# muestras]')
plt.xlim([0, 41]) 
plt.grid(True, which='both', ls=':')
plt.legend()


plt.tight_layout()
plt.show()