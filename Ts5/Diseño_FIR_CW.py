#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diseño comparativo de filtros FIR Kaiser y Remez (Parks-McClellan) 
para cuatro anchos de banda (50 Hz, 100 Hz, 250 Hz, 500 Hz), centrado en 939.91 Hz.

Incluye Polos/Ceros y Magnitud individuales para cada uno de los 8 filtros.

Created on Wed Nov 12 21:57:34 2025

@author: Fernando Daniel Fiamberti
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from matplotlib import patches
import warnings

# --- Parámetros Generales ---
fs = 44100 # Hz (Aumentada para permitir gráficos hasta 1500 Hz)
FC_TONE = 939.91 # Frecuencia Central requerida

N = 3000   # Orden del filtro (Número de coeficientes - 1)
numtaps = N + 1 # Número de coeficientes/taps

# === CARACTERÍSTICAS FIR INTACTAS ===
# El ancho de banda de transición (Delta F) del diseño original era: 0.8 - 0.1 = 0.7 Hz.
T_ORIGINAL = 0.7 
# Coeficiente Beta para la ventana Kaiser (Aproximadamente A=150 dB para N=3000, DeltaF=0.7 Hz)
BETA_KAISER = 15.6 
# ===============================================

# --- LÍMITES DE FRECUENCIA PARA GRÁFICOS ---
FREQ_LIMIT_PLOT_LOW = 400 # Hz 
FREQ_LIMIT_PLOT_HIGH = 1500 # Hz 
# -------------------------------------------

# === Definición de parámetros por Ancho de Banda (BW) ===
DESIGN_PARAMS = {
    '50': {'BW_P': 50, 'T': T_ORIGINAL}, 
    '100': {'BW_P': 100, 'T': T_ORIGINAL},
    '250': {'BW_P': 250, 'T': T_ORIGINAL},
    '500': {'BW_P': 500, 'T': T_ORIGINAL}
}

FILTER_TYPES = {
    'Kaiser': {'func': 'firwin2', 'window': ('kaiser', BETA_KAISER)}, 
    'Remez': {'func': 'remez', 'window': None}
}

filtros = {}
resultados = {}
eps = 1e-8 # Piso numérico para evitar log10(0)
alpha_s_ref = 100 # Referencia de atenuación para el eje Y (dB)


# --- Función auxiliar para graficar Polos/Ceros ---
def plot_pz(z, p, title, order):
    plt.figure(figsize=(6,6))
    plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label='Polos (en z=0)' )
    axes_hdl = plt.gca()
    if len(z) > 0:
        # Solo graficar ceros dentro del círculo unidad para claridad
        valid_z = z[np.abs(z) <= 1.05] 
        plt.plot(np.real(valid_z), np.imag(valid_z), 'o', markersize=10, fillstyle='none', label='Ceros')
    
    plt.axhline(0, color='k', lw=0.5)
    plt.axvline(0, color='k', lw=0.5)
    unit_circle = patches.Circle((0, 0), radius=1, fill=False, color='gray', ls='dotted', lw=2)
    axes_hdl.add_patch(unit_circle)
    plt.axis([-1.1, 1.1, -1.1, 1.1])
    plt.title(f'Polos y Ceros - {title} (N={order})')
    plt.xlabel(r'$\Re(z)$')
    plt.ylabel(r'$\Im(z)$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- Función auxiliar para graficar Magnitud ---
def plot_magnitude(w, H, title, order, wp_current):
    plt.figure(figsize=(8, 6))
    plt.plot(w, 20*np.log10(abs(H) + eps))
    plt.title(f'Respuesta en Magnitud - {title} (N={order})')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(jω)| [dB]')
    plt.xlim([FREQ_LIMIT_PLOT_LOW, FREQ_LIMIT_PLOT_HIGH]) 
    plt.ylim([-alpha_s_ref - 10, 5]) 
    
    # Se dibujan las bandas centradas para contexto
    plt.axvspan(wp_current[0], wp_current[1], color='green', alpha=0.1, label='Banda de Paso Deseada') 
    plt.axvline(wp_current[0], color='b', linestyle=':', alpha=0.5)
    plt.axvline(wp_current[1], color='b', linestyle=':', alpha=0.5)
    
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------
# 2. Bucle de Diseño para los 4 BWs y 2 Tipos de FIR
# -------------------------------------------------------------
for BW_KEY, params in DESIGN_PARAMS.items():
    BW = params['BW_P']
    T = params['T']
    
    BW_HALF = BW / 2
    
    # 1. Calcular las bandas de frecuencia
    wp_current = (FC_TONE - BW_HALF, FC_TONE + BW_HALF) # Frecuencias de Paso
    ws_current = (wp_current[0] - T, wp_current[1] + T) # Frecuencias de Detenida
    
    # Puntos de frecuencia para la respuesta (solo en el rango de plot)
    freq_points_response = np.linspace(0, fs/2, 2000)

    for FTYPE, FPARAMS in FILTER_TYPES.items():
        name = f'{BW_KEY}Hz_{FTYPE}'
        b = None

        # --- Definición de Plantilla Remez/Kaiser ---
        f_remez = [0, ws_current[0], wp_current[0], wp_current[1], ws_current[1], fs/2]
        a_remez = [0, 1, 0] # Magnitud deseada: 0, 1, 0

        # --- Diseño Remez (Parks-McClellan) ---
        if FTYPE == 'Remez':
            b = sig.remez(numtaps=numtaps, bands=f_remez, desired=a_remez, fs=fs, type='bandpass')

        # --- Diseño Kaiser (usando firwin2 con ventana Kaiser) ---
        elif FTYPE == 'Kaiser':
            # La función firwin2 requiere un conjunto de puntos de frecuencia y ganancias.
            # Usamos los puntos de la plantilla remez para la forma ideal.
            f_deseada = [
                0,               
                ws_current[0] + 0.1, # Se añade una transición suave para firwin2
                wp_current[0],                 
                wp_current[1],                 
                ws_current[1] - 0.1,
                fs / 2          
            ]
            m_deseada = [0, 0, 1, 1, 0, 0] # Magnitudes correspondientes
            
            b = sig.firwin2(
                numtaps=numtaps, 
                freq=f_deseada, 
                gain=m_deseada, 
                fs=fs, 
                window=FPARAMS['window'] # Usar ('kaiser', BETA_KAISER)
            )

        # -------------------------------------------------------------
        # 3. Análisis y Almacenamiento
        # -------------------------------------------------------------
        if b is not None:
            # Respuesta en frecuencia
            w, H = sig.freqz(b, a=1, worN=freq_points_response, fs=fs)
            phase = np.unwrap(np.angle(H))
            w_rad = w / (fs/2) * np.pi
            gd = -np.diff(phase) / np.diff(w_rad)
            
            # Polos y ceros
            z = np.roots(b)
            p = np.zeros(N) # Polos siempre en z=0 para FIR
            
            resultados[name] = {
                'w': w, 'H': H, 'phase': phase, 'gd': gd, 
                'z': z, 'p': p, 'order': N, 'wp': wp_current
            }

            # -------------------------------------------------------------
            # 4. Gráficos Individuales (Polos/Ceros y Magnitud)
            # -------------------------------------------------------------
            plot_pz(z, p, name, N)
            plot_magnitude(w, H, name, N, wp_current)

# ----------------------------------------------------------------------------------------
# 5. Gráficas Comparativas (Magnitud, Fase, Retardo de Grupo) 📈
# ----------------------------------------------------------------------------------------

for BW_KEY in DESIGN_PARAMS.keys():
    plt.figure(figsize=(12, 10))
    plt.suptitle(f'Diseño FIR - Comparativa de Tipos (BW={BW_KEY} Hz, N={N})', fontsize=14)

    # Filtramos solo los resultados del BW actual
    resultados_bw = {k: v for k, v in resultados.items() if k.startswith(f'{BW_KEY}Hz')}

    # Subplot 1: Magnitud
    plt.subplot(3,1,1)
    wp_current = resultados_bw[f'{BW_KEY}Hz_Remez']['wp'] # Usar las bandas de paso de referencia
    
    for name, res in resultados_bw.items():
        if 'Kaiser' in name:
            plt.plot(res['w'], 20*np.log10(abs(res['H']) + eps), label = 'FIR Kaiser', color='blue')
        elif 'Remez' in name:
            plt.plot(res['w'], 20*np.log10(abs(res['H']) + eps), label = 'FIR Remez', color='red', linestyle='--')
            
    plt.title('Respuesta en Magnitud')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(jω)| [dB]')
    plt.xlim([FREQ_LIMIT_PLOT_LOW, FREQ_LIMIT_PLOT_HIGH]) 
    plt.ylim([-alpha_s_ref - 10, 5]) 
    plt.axvspan(wp_current[0], wp_current[1], color='green', alpha=0.1) 
    plt.grid(True, which='both', ls=':')
    plt.legend()

    # Subplot 2: Fase
    plt.subplot(3,1,2)
    for name, res in resultados_bw.items():
        if 'Kaiser' in name:
            plt.plot(res['w'], res['phase'], label = 'FIR Kaiser', color='blue')
        elif 'Remez' in name:
            plt.plot(res['w'], res['phase'], label = 'FIR Remez', color='red', linestyle='--')

    plt.title('Fase (Fase Lineal)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Fase [rad]')
    plt.xlim([FREQ_LIMIT_PLOT_LOW, FREQ_LIMIT_PLOT_HIGH]) 
    plt.grid(True, which='both', ls=':')
    plt.legend()


    # Subplot 3: Retardo de grupo
    plt.subplot(3,1,3)
    # El retardo de grupo es el mismo N/2 para ambos filtros de fase lineal
    gd_esperado = N / 2
    plt.axhline(gd_esperado, color='orange', linestyle='-', label=f'τg teórico={gd_esperado:.1f} muestras (Ambos)')
    plt.title('Retardo de Grupo (Fase Lineal)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel(r'$\tau_g$ [# muestras]')
    plt.xlim([FREQ_LIMIT_PLOT_LOW, FREQ_LIMIT_PLOT_HIGH]) 
    plt.grid(True, which='both', ls=':')
    plt.legend()


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()