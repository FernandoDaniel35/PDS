#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diseño comparativo de filtros IIR Chebyshev I y Cauer/Elíptico para 
los cuatro anchos de banda (50 Hz, 100 Hz, 250 Hz, 500 Hz), centrado en 939.91 Hz.

**FINAL:** Se incluye la corrección del KeyError y el manejo de advertencias de BadCoefficients 
para suprimir los mensajes en Spyder/consola sin modificar la lógica del diseño.
"""

import numpy as np
from scipy import signal as sig
import matplotlib.pyplot as plt
from matplotlib import patches
import warnings
import sys
import io

# =============================================================
# 0. CONFIGURACIÓN DE ADVERTENCIAS
# =============================================================
# 1. Suprimir el warning "BadCoefficients" para evitar los mensajes de Spyder/consola.
warnings.filterwarnings('ignore', 'Badly conditioned filter coefficients.*', category=sig.BadCoefficients)
# 2. También ignorar la advertencia por número de línea (line 1230 en Spyder 6, o 1125 en Spyder 5)
warnings.filterwarnings('ignore', category=UserWarning, message='The parameter output is only valid for butter, cheby1, cheby2, or ellip')


# =============================================================
# 1. PARÁMETROS GENERALES Y ESPECIFICACIONES (CW)
# =============================================================
# Frecuencia de muestreo (Asumiendo un valor común de audio)
fs = 44100 # Hz 
FC_TONE = 939.91 # Frecuencia central del CW

# === Parámetros de atenuación comunes ===
alpha_p = 0.5 # Atenuación máxima en banda de paso (dB)
alpha_s = 80 # Atenuación mínima en banda de atenuación (dB)

# Límite de frecuencia para los gráficos de respuesta
FREQ_LIMIT_PLOT_LOW = 400 # Hz
FREQ_LIMIT_PLOT_HIGH = 1500 # Hz

# === Definición de parámetros por Ancho de Banda (BW) ===
DESIGN_PARAMS = {
    '50': {'BW_Hz': 50, 'T_Hz': 50}, # T=50Hz -> ws = wp ± 50
    '100': {'BW_Hz': 100, 'T_Hz': 50},
    '250': {'BW_Hz': 250, 'T_Hz': 50},
    '500': {'BW_Hz': 500, 'T_Hz': 50}
}
# =============================================================

filtros = {}
resultados = {}

# -------------------------------------------------------------
# 2. Bucle de Diseño para los 4 BW
# -------------------------------------------------------------

# Usamos catch_warnings aquí para encapsular la sección de diseño donde se genera el BadCoefficients
with warnings.catch_warnings():
    
    for BW_KEY_NUMERIC, params in DESIGN_PARAMS.items(): # BW_KEY_NUMERIC es '50', '100', etc.
        BW = params['BW_Hz']
        T = params['T_Hz']
        
        # Cálculos de Bandas de Frecuencia
        wp = (FC_TONE - BW/2, FC_TONE + BW/2) 
        ws = (wp[0] - T, wp[1] + T) 
        
        name_prefix = f"{BW_KEY_NUMERIC}Hz" # e.g., '50Hz'
        
        # Suprimimos la salida de stderr para buttord
        falso_stderr = io.StringIO()
        old_stderr = sys.stderr

        try:
            sys.stderr = falso_stderr
            # Diseño Butterworth (mantenido por estructura)
            N_butt, wn_butt = sig.buttord(wp, ws, alpha_p, alpha_s, analog=False, fs=fs)
            mi_sos_butt = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s,analog=False, ftype='butter', output='sos', fs=fs)
        finally:
            sys.stderr = old_stderr
        # Omitimos guardar Butterworth para la comparativa final, pero se diseña.


        # --- Diseño Chebyshev Tipo I ---
        N_cheb1, wn_cheb1 = sig.cheb1ord(wp, ws, alpha_p, alpha_s, analog=False, fs=fs)
        mi_sos_cheb1 = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s,analog=False, ftype='cheby1', output='sos', fs=fs)
        filtros[f'{name_prefix}_Chebyshev I'] = mi_sos_cheb1


        # --- Diseño Chebyshev Tipo II (mantenido por estructura) ---
        N_cheb2, wn_cheb2 = sig.cheb2ord(wp, ws, alpha_p, alpha_s, analog=False, fs=fs)
        mi_sos_cheb2 = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s,analog=False, ftype='cheby2', output='sos', fs=fs)
        # Omitimos guardar Chebyshev II para la comparativa final, pero se diseña.


        # --- Diseño Cauer/Elíptico ---
        N_cauer, wn_cauer = sig.ellipord(wp, ws, alpha_p, alpha_s, analog=False, fs=fs)
        mi_sos_cauer = sig.iirdesign(wp, ws, gpass=alpha_p, gstop=alpha_s,analog=False, ftype='ellip', output='sos', fs=fs)
        filtros[f'{name_prefix}_Cauer/Elliptic'] = mi_sos_cauer

# -------------------------------------------------------------
# 3. Análisis de Respuesta y Polos/Ceros (Generación de todos los gráficos individuales)
# -------------------------------------------------------------

# Usamos linspace para asegurar puntos en el rango de interés [400, 1500]
freq_points_plot = np.linspace(FREQ_LIMIT_PLOT_LOW, FREQ_LIMIT_PLOT_HIGH, 1000)

for name, mi_sos in filtros.items():
    
    # --- Respuesta en frecuencia ---
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=sig.BadCoefficients) 
        w, h = sig.sosfreqz(mi_sos, worN=freq_points_plot, fs = fs) 
        
        # Cálculo de fase y retardo de grupo
        phase = np.unwrap(np.angle(h))
        w_rad = w / (fs/2) * np.pi
        
        if len(w_rad) > 1:
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", RuntimeWarning)
                 gd = -np.diff(phase) / np.diff(w_rad)
        else:
            gd = np.array([])
    
    # --- Polos y ceros ---
    z, p, k = sig.sos2zpk(mi_sos)
    order = mi_sos.shape[0] * 2
    
    resultados[name] = {'w': w, 'h': h, 'phase': phase, 'gd': gd, 'z': z, 'p': p, 'order': order}

    # --- Gráfico Individual 1: Polos y Ceros ---
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

    # --- Gráfico Individual 2: Magnitud (Restaurado y Corregido) ---
    eps = 1e-8
    
    # Extraer la clave numérica del BW (ej: '50' de '50Hz')
    BW_KEY_FULL = name.split('_')[0]
    BW_KEY_NUMERIC = BW_KEY_FULL.replace('Hz', '') 
    
    BW_VAL = DESIGN_PARAMS[BW_KEY_NUMERIC]['BW_Hz']
    T_VAL = DESIGN_PARAMS[BW_KEY_NUMERIC]['T_Hz']
    
    # Recalcular límites de banda para el plot
    wp_current = (FC_TONE - BW_VAL/2, FC_TONE + BW_VAL/2)
    ws_current = (wp_current[0] - T_VAL, wp_current[1] + T_VAL)
    
    plt.figure(figsize=(8, 6))
    plt.plot(w, 20*np.log10(abs(h) + eps))
    plt.title(f'Respuesta en Magnitud - {name} (N={order})')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('|H(jω)| [dB]')
    plt.xlim([FREQ_LIMIT_PLOT_LOW, FREQ_LIMIT_PLOT_HIGH]) # Límite en [400, 1500] Hz
    plt.ylim([-alpha_s - 10, alpha_p + 1]) 
    
    plt.axhline(alpha_p, color='r', linestyle='--', alpha=0.5, label=f'Límite Bp ({alpha_p} dB)')
    plt.axhline(-alpha_s, color='g', linestyle='--', alpha=0.5, label=f'Límite Bs (-{alpha_s} dB)') 
    
    plt.axvline(wp_current[0], color='b', linestyle=':', alpha=0.5, label='Banda de Paso')
    plt.axvline(wp_current[1], color='b', linestyle=':', alpha=0.5)
    plt.axvline(ws_current[0], color='c', linestyle=':', alpha=0.5, label='Banda de Detención')
    plt.axvline(ws_current[1], color='c', linestyle=':', alpha=0.5)
    
    plt.grid(True, which='both', ls=':')
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------------------------
# 4. Gráficas Comparativas (Magnitud, Fase, Retardo de Grupo) 📈
# ----------------------------------------------------------------------------------------

eps = 1e-8 # Piso numérico para evitar log10(0)

### 4.1. Magnitud Comparativa: Chebyshev I vs Cauer/Elíptico (4 BW)
plt.figure(figsize=(12, 10))
plt.suptitle(f'Diseño IIR - Respuesta en Magnitud (alpha_p={alpha_p} dB, alpha_s={alpha_s} dB)', fontsize=14)

# Subplot 1: Chebyshev I (4 BW)
plt.subplot(2,1,1)
for name, res in resultados.items():
    if 'Chebyshev I' in name:
        plt.plot(res['w'], 20*np.log10(abs(res['h']) + eps), label = f'{name.split("_")[0]} - Chebyshev I')
plt.title('Chebyshev I - Comparativa de Ancho de Banda')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.xlim([FREQ_LIMIT_PLOT_LOW, FREQ_LIMIT_PLOT_HIGH]) # Límite en [400, 1500] Hz
plt.ylim([-alpha_s - 10, alpha_p + 1]) 
plt.grid(True, which='both', ls=':')
plt.legend(loc='lower left')

# Subplot 2: Cauer/Elíptico (4 BW)
plt.subplot(2,1,2)
for name, res in resultados.items():
    if 'Cauer/Elliptic' in name:
        plt.plot(res['w'], 20*np.log10(abs(res['h']) + eps), label = f'{name.split("_")[0]} - Cauer/Elíptico')
plt.title('Cauer/Elíptico - Comparativa de Ancho de Banda')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.xlim([FREQ_LIMIT_PLOT_LOW, FREQ_LIMIT_PLOT_HIGH]) # Límite en [400, 1500] Hz
plt.ylim([-alpha_s - 10, alpha_p + 1]) 
plt.grid(True, which='both', ls=':')
plt.legend(loc='lower left')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


### 4.2. Fase y Retardo de Grupo (Solo para BW=100Hz como ejemplo)
plt.figure(figsize=(12, 10))
plt.suptitle(f'Fase y Retardo de Grupo - Comparativa (BW=100 Hz)', fontsize=14)

# Filtramos solo los resultados de 100Hz
resultados_100hz = {k: v for k, v in resultados.items() if k.startswith('100Hz')}

# Subplot 1: Fase
plt.subplot(2,1,1)
for name, res in resultados_100hz.items():
    plt.plot(res['w'], res['phase'], label = name.split('_')[1])
plt.title('Fase - Comparativa (BW=100Hz)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [rad]')
plt.xlim([FREQ_LIMIT_PLOT_LOW, FREQ_LIMIT_PLOT_HIGH]) # Límite en [400, 1500] Hz
plt.grid(True, which='both', ls=':')
plt.legend()

# Subplot 2: Retardo de grupo
plt.subplot(2,1,2)
for name, res in resultados_100hz.items():
    # El retardo de grupo tiene un elemento menos que 'w'
    plt.plot(res['w'][:-1], res['gd'], label = name.split('_')[1]) 
plt.title('Retardo de Grupo - Comparativa (BW=100Hz)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel(r'$\tau_g$ [# muestras]')
plt.xlim([FREQ_LIMIT_PLOT_LOW, FREQ_LIMIT_PLOT_HIGH]) # Límite en [400, 1500] Hz
plt.grid(True, which='both', ls=':')
plt.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()