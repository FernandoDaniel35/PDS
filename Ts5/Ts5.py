#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aplica, evalúa y verifica la Respuesta en Frecuencia (Punto c).

Created on Wed Nov 12 22:05:31 2025

@author: Fernando Daniel Fiamberti
"""


import numpy as np
from scipy import signal as sig
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
import sys

# =========================================================================
# 1. Parámetros de Carga de Datos y Diseño
# =========================================================================

# Parámetros de carga
DATA_FILENAME = 'ecg.mat'
DATA_KEY = 'ecg_lead'
FS = 1000 # Frecuencia de muestreo (Hz)
N_SAMPLES_TO_USE = 1120000 

# Parámetros de la Región de Interés 
REGION_START = 50000 
REGION_END = 55000 

# ZONAS FINALES 
ZONAS_DE_INTERES_SOLICITADAS_Y_ORDENADAS = [
    (4000, 5500),         # 1. 4000 a 5500
    (10000, 11000),       # 2. 10000 a 11000
    (101000, 118000),     # 3. 101000 a 118000 (101s a 118s)
    (120000, 132000),     # 4. 120000 a 132000 (2 min a 2.2 min)
    (237000, 242000),     # 5. 237000 a 242000
    (247000, 252000),     # 6. 247000 a 252000
    (300000, 312000),     # 7. 300000 a 312000 (5 min a 5.2 min)
    # Zonas de minutos convertidas a muestras
    (600000, 612000),     # 8. 10 min a 10.2 min (600000 a 612000)
    (700000, 745000),     # 9. 700000 a 745000 (Existente)
    (720000, 744000),     # 10. 12 min a 12.4 min (720000 a 744000)
    (732000, 765000),     # 11. ZONA NUEVA SOLICITADA (732s a 765s)
    (900000, 912000),     # 12. 15 min a 15.2 min (900000 a 912000)
    (1080000, 1092000),   # 13. 18 min a 18.2 min (1080000 a 1092000)
]

# Las listas ZONAS_DE_INTERES y REGS_DE_INTERES_ADICIONALES se mantienen vacías/no utilizadas.
ZONAS_DE_INTERES = []
REGS_DE_INTERES_ADICIONALES = [] 

# Parámetros de Diseño
fs = FS
# IIR
alpha_p = 1 # Atenuación máxima en banda de paso (dB)
alpha_s = 40 # Atenuación mínima en banda de detención (dB)
wp = (0.8, 35) # Frecuencia de corte/paso (Hz)
ws_iir = (0.1, 40) # Frecuencia de stop/detenida IIR (Hz)
# FIR
N_fir = 3000 # Orden del filtro FIR
numtaps_fir = N_fir + 1
delta1_fir = 0.1
ws0_fir = 0.1
ws1_fir = 35.7

# --- PARÁMETRO PARA SUAVIZADO DE GRÁFICAS ---
N_FREQZ_POINTS = 32768 # Número de puntos para calcular la respuesta en frecuencia (más puntos = curvas más suaves)
# --- Puntos para el zoom de alta resolución de los IIR en baja frecuencia ---
N_FREQZ_ZOOM_POINTS = 50000 

# =========================================================================
# 2. Carga y Preparación de la Señal ECG con Ruido
# =========================================================================

try:
    print(f"Cargando {DATA_FILENAME}...")
    mat = sio.loadmat(DATA_FILENAME)
    ecg_ruido_full = np.squeeze(mat.get(DATA_KEY, None))
    
    if ecg_ruido_full is None:
        raise KeyError(f"La clave '{DATA_KEY}' no se encontró en {DATA_FILENAME}.")
        
    if len(ecg_ruido_full) >= N_SAMPLES_TO_USE:
        ecg_ruido_full = ecg_ruido_full[:N_SAMPLES_TO_USE]
        print(f"Cargadas {len(ecg_ruido_full)} muestras de ECG con ruido.")
    else:
        # Si la señal es más corta que 1120000, se usa la longitud total
        N_SAMPLES_TO_USE = len(ecg_ruido_full)
        ecg_ruido_full = ecg_ruido_full
        print(f"Advertencia: El archivo tiene menos de 1120000. Usando {N_SAMPLES_TO_USE} muestras.")

except (FileNotFoundError, KeyError, ValueError) as e:
    print(f"¡Error! No se pudo cargar o procesar '{DATA_FILENAME}': {e}")
    N_SAMPLES_TO_USE = 10000 
    t = np.arange(0, N_SAMPLES_TO_USE) / FS
    ecg_ruido_full = np.sin(2 * np.pi * 1 * t) + 1.5 * np.sin(2 * np.pi * 50 * t) 
    print("Usando ECG simulado de emergencia con ruido.")

N_signal = len(ecg_ruido_full)

# =========================================================================
# 3. Diseño de los 4 Filtros
# =========================================================================

# --- FIR Boxcar
f_deseada = [0, ws0_fir + delta1_fir, wp[0], wp[1], ws1_fir - delta1_fir, fs / 2]
m_deseada = [0, 0, 1, 1, 0, 0]
b_boxcar = sig.firwin2(numtaps=numtaps_fir, freq=f_deseada, gain=m_deseada, fs=fs, window='boxcar')

# --- FIR Parks-McClellan (Remez)
f_remez = [0, ws0_fir, wp[0], wp[1], ws1_fir, fs/2]
a_remez = [0, 1, 0] 
b_remez = sig.remez(numtaps=numtaps_fir, bands=f_remez, desired=a_remez, fs=fs, type='bandpass')

with warnings.catch_warnings():
    warnings.simplefilter("ignore", sig.BadCoefficients)
    # --- IIR Chebyshev I
    N_cheb1, _ = sig.cheb1ord(wp, ws_iir, alpha_p, alpha_s, analog=False, fs=fs)
    mi_sos_cheb1 = sig.iirdesign(wp, ws_iir, gpass=alpha_p, gstop=alpha_s, 
                                 analog=False, ftype='cheby1', output='sos', fs=fs)
    # --- IIR Cauer/Elíptico
    N_cauer, _ = sig.ellipord(wp, ws_iir, alpha_p, alpha_s, analog=False, fs=fs)
    mi_sos_cauer = sig.iirdesign(wp, ws_iir, gpass=alpha_p, gstop=alpha_s, 
                                 analog=False, ftype='ellip', output='sos', fs=fs)

# =========================================================================
# 4. Aplicación de Filtrado
# =========================================================================

FILTROS = {
    'FIR Boxcar': {'b': b_boxcar, 'a': 1, 'type': 'FIR', 'design_info': f'N={N_fir}'},
    'FIR Parks-McClellan': {'b': b_remez, 'a': 1, 'type': 'FIR', 'design_info': f'N={N_fir}'},
    'IIR Chebyshev I': {'sos': mi_sos_cheb1, 'type': 'IIR', 'design_info': f'N={N_cheb1}'},
    'IIR Cauer/Elíptico': {'sos': mi_sos_cauer, 'type': 'IIR', 'design_info': f'N={N_cauer}'}
}

ecg_filtrado = {}

# Bloque de manejo de advertencias: suprime BadCoefficients durante el filtrado IIR
with warnings.catch_warnings():
    warnings.simplefilter("ignore", sig.BadCoefficients)
    
    for name, params in FILTROS.items():
        print(f"Aplicando filtro: {name}...")
        
        if params['type'] == 'FIR':
            ecg_filtrado[name] = sig.filtfilt(params['b'], params['a'], ecg_ruido_full)
            
        elif params['type'] == 'IIR':
            ecg_filtrado[name] = sig.sosfiltfilt(params['sos'], ecg_ruido_full)

print("\n--- Filtrado Completado para los 4 Diseños. ---")


# =========================================================================
# 6. Verificación de la Respuesta en Frecuencia 
# =========================================================================

print("\n--- Generando Gráficos de Verificación de Respuesta en Frecuencia (Punto c) ---")

# --- 6.1 Cálculo de la Respuesta en Frecuencia y Plantilla ---
respuestas = {}
eps = 1e-8

for name, params in FILTROS.items():
    
    is_iir_zoom = (name == 'IIR Cauer/Elíptico' or name == 'IIR Chebyshev I')
    
    if is_iir_zoom:
        # --- Cálculo en Alta Resolución para el Zoom IIR (0 a 1 Hz) ---
        sos = params['sos']
        # 50000 puntos en un rango muy estrecho (0 a 1 Hz).
        w_zoom_res, h_zoom_res = sig.sosfreqz(sos, fs=fs, worN=np.linspace(0, 1, N_FREQZ_ZOOM_POINTS, endpoint=False))

        # --- Cálculo Normal (de 1 Hz al final) y concatenación ---
        # 32768 puntos en todo el rango [0, fs/2], se toma solo la parte superior a 1 Hz
        w_norm, h_norm = sig.sosfreqz(sos, fs=fs, worN=N_FREQZ_POINTS)
        
        # Encontrar el índice más cercano a 1 Hz en el cálculo normal
        idx_start_norm = np.searchsorted(w_norm, 1.0)
        
        # Concatenar: Zoom de alta res (excluye el punto 0 Hz), y el resto normal.
        w = np.concatenate((w_zoom_res[1:], w_norm[idx_start_norm:]))
        h = np.concatenate((h_zoom_res[1:], h_norm[idx_start_norm:]))
        
        # Añadir el punto 0 Hz (DC) para asegurar el inicio del espectro.
        w = np.insert(w, 0, w_norm[0])
        h = np.insert(h, 0, h_norm[0])

    elif params['type'] == 'FIR':
        w, h = sig.freqz(params['b'], params['a'], fs=fs, worN=N_FREQZ_POINTS)
    elif params['type'] == 'IIR':
        # Cálculo normal para otros IIR si existieran, aunque solo hay 2 IIR definidos
        w, h = sig.sosfreqz(params['sos'], fs=fs, worN=N_FREQZ_POINTS)
    
    # Calcular fase y retardo de grupo 
    fase = np.unwrap(np.angle(h))
    group_delay = -np.diff(fase) / np.diff(w * 2 * np.pi) 
    
    respuestas[name] = {'w': w, 'h': h, 'fase': fase, 'group_delay': group_delay}

# Definición de la Plantilla de Diseño (Límites en dB)
limite_p_max = 20*np.log10(1 + eps)
limite_p_min = -alpha_p
limite_s_max = -alpha_s

# --- 6.2 Función de Ploteo Individual de la Plantilla ---
def plot_plantilla(ax):
    # Dibuja las bandas requeridas (PLANTILLA)
    ax.axvspan(0, ws_iir[0], color='red', alpha=0.1, label='Banda de Detención')
    ax.axvspan(wp[0], wp[1], color='green', alpha=0.1, label='Banda de Paso')
    ax.axvspan(ws_iir[1], fs/2, color='red', alpha=0.1) 
    # Líneas de las especificaciones (dB)
    ax.axhline(limite_p_max, color='black', linestyle=':', linewidth=1)
    ax.axhline(limite_p_min, color='black', linestyle=':', linewidth=1, label=f'-{alpha_p} dB')
    ax.axhline(limite_s_max, color='black', linestyle='--', linewidth=1, label=f'-{alpha_s} dB')
    ax.grid(True, which='both', ls=':')
    ax.set_xlim([0, 41]) # Límite superior 41 Hz

# --- 6.3 Gráficas de Verificación Individuales (4 Figuras) ---
for name, res in respuestas.items():
    # Solo 6 filas (2+2+2)
    plt.figure(figsize=(10, 18)) 
    
    # Magnitud Principal [dB]
    ax1 = plt.subplot(6, 1, (1, 2)) 
    w_zoom = res['w'][res['w'] <= 41] 
    h_zoom = res['h'][res['w'] <= 41] 
    ax1.plot(w_zoom, 20*np.log10(abs(h_zoom) + eps), label=name, color='blue', linewidth=2)
    plot_plantilla(ax1)
    ax1.set_title(f'Magnitud Principal y Plantilla de Diseño - {name}')
    ax1.set_ylabel('|H(jω)| [dB]')
    ax1.set_ylim([-60, 5])
    ax1.legend(loc='lower left')
    
    # --- ZOOM 1: Baja Frecuencia (0Hz a 0.9Hz) ---
    ax_zoom_low = plt.subplot(6, 1, (3, 4)) 
    ax_zoom_low.plot(w_zoom, 20*np.log10(abs(h_zoom) + eps), label=name, color='blue', linewidth=2)
    
    ax_zoom_low.axvspan(0, ws_iir[0], color='red', alpha=0.1)
    ax_zoom_low.axvspan(ws_iir[0], wp[0], color='yellow', alpha=0.1)
    ax_zoom_low.axhline(limite_s_max, color='red', linestyle='--', linewidth=1)
    ax_zoom_low.axhline(limite_p_min, color='green', linestyle=':', linewidth=1)
    
    is_iir_zoom = (name == 'IIR Cauer/Elíptico' or name == 'IIR Chebyshev I')
    
    # Título dinámico
    ax_zoom_low.set_title(f'Zoom Magnitud: Banda de Detención Baja y Transición ({0.6 if is_iir_zoom else 0}Hz a 0.9Hz)') 
    ax_zoom_low.set_ylabel('|H(jω)| [dB]')
    
    # Zoom de 0.6 Hz a 0.9 Hz para IIR Chebyshev I y Cauer
    if is_iir_zoom:
        ax_zoom_low.set_xlim([0.6, 0.9])
    else:
        ax_zoom_low.set_xlim([0, 0.9])
        
    ax_zoom_low.set_ylim([-60, 5]) 
    ax_zoom_low.grid(True, which='both', ls=':')
    
    # --- ZOOM 2: Alta Frecuencia (34Hz a 41Hz) ---
    ax_zoom_high = plt.subplot(6, 1, (5, 6)) 
    ax_zoom_high.plot(w_zoom, 20*np.log10(abs(h_zoom) + eps), label=name, color='blue', linewidth=2)
    
    ax_zoom_high.axvspan(wp[1], ws_iir[1], color='yellow', alpha=0.1)
    ax_zoom_high.axvspan(ws_iir[1], 41, color='red', alpha=0.1)
    ax_zoom_high.axhline(limite_s_max, color='red', linestyle='--', linewidth=1)
    ax_zoom_high.axhline(limite_p_min, color='green', linestyle=':', linewidth=1)
    
    ax_zoom_high.set_title('Zoom Magnitud: Banda de Detención Alta y Transición (34Hz a 41Hz)')
    ax_zoom_high.set_ylabel('|H(jω)| [dB]')
    ax_zoom_high.set_xlim([34, 41])
    ax_zoom_high.set_ylim([-60, 5])
    ax_zoom_high.grid(True, which='both', ls=':')
    ax_zoom_high.set_xlabel('Frecuencia [Hz]')
    
    plt.suptitle(f'Verificación de Respuesta en Frecuencia - {name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 6.4 Gráfica de Verificación Comparativa (1 Figura) ---
plt.figure(figsize=(12, 20)) 

# Magnitud Principal [dB] - Comparativa
ax_comp_mag = plt.subplot(6, 1, (1, 2)) 
colores_comp = ['blue', 'green', 'orange', 'purple']
estilos_comp = ['-', '--', ':', '-.']
i = 0
for name, res in respuestas.items():
    w_zoom = res['w'][res['w'] <= 41] 
    h_zoom = res['h'][res['w'] <= 41] 
    ax_comp_mag.plot(w_zoom, 20*np.log10(abs(h_zoom) + eps), label=name, color=colores_comp[i], linestyle=estilos_comp[i], linewidth=2)
    i += 1
plot_plantilla(ax_comp_mag)
ax_comp_mag.set_title('Respuesta en Magnitud Principal - Comparativa de los 4 Filtros vs. Plantilla')
ax_comp_mag.set_ylabel('|H(jω)| [dB]')
ax_comp_mag.set_ylim([-60, 5])
ax_comp_mag.legend(loc='lower left')

# --- ZOOM 1: Baja Frecuencia (0Hz a 0.9Hz) - Comparativa ---
ax_comp_zoom_low = plt.subplot(6, 1, (3, 4)) 
i = 0
for name, res in respuestas.items():
    # Usar el cálculo completo (incluyendo la alta resolución) para los IIR
    is_iir_zoom = (name == 'IIR Cauer/Elíptico' or name == 'IIR Chebyshev I')
    
    if is_iir_zoom:
        # Se grafican todos los puntos calculados (incluyendo la alta resolución)
        ax_comp_zoom_low.plot(res['w'], 20*np.log10(abs(res['h']) + eps), label=name, color=colores_comp[i], linestyle=estilos_comp[i], linewidth=2)
    else:
        w_zoom = res['w'][res['w'] <= 41] 
        h_zoom = res['h'][res['w'] <= 41] 
        ax_comp_zoom_low.plot(w_zoom, 20*np.log10(abs(h_zoom) + eps), label=name, color=colores_comp[i], linestyle=estilos_comp[i], linewidth=2)
    
    i += 1

ax_comp_zoom_low.axvspan(0, ws_iir[0], color='red', alpha=0.1)
ax_comp_zoom_low.axvspan(ws_iir[0], wp[0], color='yellow', alpha=0.1)
ax_comp_zoom_low.axhline(limite_s_max, color='red', linestyle='--', linewidth=1)
ax_comp_zoom_low.axhline(limite_p_min, color='green', linestyle=':', linewidth=1)

ax_comp_zoom_low.set_title('Zoom Magnitud: Banda de Detención Baja y Transición (IIR con alta resolución)')
ax_comp_zoom_low.set_ylabel('|H(jω)| [dB]')
# El límite X se mantiene en [0, 0.9] en el comparativo para ver todos
ax_comp_zoom_low.set_xlim([0, 0.9]) 
ax_comp_zoom_low.set_ylim([-60, 5]) 
ax_comp_zoom_low.grid(True, which='both', ls=':')
ax_comp_zoom_low.legend(loc='lower left', fontsize='small')


# --- ZOOM 2: Alta Frecuencia (34Hz a 41Hz) - Comparativa ---
ax_comp_zoom_high = plt.subplot(6, 1, (5, 6))
i = 0
for name, res in respuestas.items():
    w_zoom = res['w'][res['w'] <= 41] 
    h_zoom = res['h'][res['w'] <= 41] 
    ax_comp_zoom_high.plot(w_zoom, 20*np.log10(abs(h_zoom) + eps), label=name, color=colores_comp[i], linestyle=estilos_comp[i], linewidth=2)
    i += 1
    
ax_comp_zoom_high.axvspan(wp[1], ws_iir[1], color='yellow', alpha=0.1)
ax_comp_zoom_high.axvspan(ws_iir[1], 41, color='red', alpha=0.1)
ax_comp_zoom_high.axhline(limite_s_max, color='red', linestyle='--', linewidth=1)
ax_comp_zoom_high.axhline(limite_p_min, color='green', linestyle=':', linewidth=1)
    
ax_comp_zoom_high.set_title('Zoom Magnitud: Banda de Detención Alta y Transición (34Hz a 41Hz)')
ax_comp_zoom_high.set_ylabel('|H(jω)| [dB]')
ax_comp_zoom_high.set_xlim([34, 41])
ax_comp_zoom_high.set_ylim([-60, 5])
ax_comp_zoom_high.grid(True, which='both', ls=':')
ax_comp_zoom_high.legend(loc='lower left', fontsize='small')
ax_comp_zoom_high.set_xlabel('Frecuencia [Hz]')

plt.suptitle('Comparativa General de Respuesta en Frecuencia (Punto c)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# =========================================================================
# 5. Visualización de Resultados, Zonas de Interés Ordenadas
# =========================================================================

print(f"\n--- Generando {len(ZONAS_DE_INTERES_SOLICITADAS_Y_ORDENADAS)} Gráficos de Zoom ---")

ZONAS_FINALES = []

# Definición de las ubicaciones de leyenda solicitadas:
# Zona 4: 'upper left'
# Zona 5: 'upper right'
# Zona 7: 'upper left'
# Zona 8: 'upper left'
# Zona 11: 'lower right' (del pedido anterior)
# Zona 12: 'upper right'
# Zona 13: 'upper right'

for idx, (start_s, end_s) in enumerate(ZONAS_DE_INTERES_SOLICITADAS_Y_ORDENADAS):
    # Asegurar orden (start < end) y límites (dentro de N_signal)
    start_final = max(0, min(start_s, end_s))
    end_final = min(N_signal - 1, max(start_s, end_s))
    
    # Asegurar que el rango sea válido
    if end_final > start_final:
        ZONAS_FINALES.append((start_final, end_final))

    # Determinar la posición de la leyenda
    legend_loc = 'lower left' # Posición por defecto
    
    # Se utiliza el índice + 1 para identificar la zona (1-based index)
    zona_num = idx + 1 
    
    if zona_num == 4 or zona_num == 7 or zona_num == 8:
        # Zona 4 (120000, 132000), Zona 7 (300000, 312000), Zona 8 (600000, 612000)
        legend_loc = 'upper left'
    elif zona_num == 5 or zona_num == 12 or zona_num == 13:
        # Zona 5 (237000, 242000), Zona 12 (900000, 912000), Zona 13 (1080000, 1092000)
        legend_loc = 'upper right'
    elif zona_num == 11:
        # Zona 11 (732000, 765000) - Mantener 'lower right' del pedido anterior
        legend_loc = 'lower right'

    # ---------------------------------------------------------------------
    # Gráfica Individual de la Zona (Original vs. 4 Filtros)
    # ---------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Intervalo limitado de 0 a N_signal
    idx_zoom = np.arange(start_final, end_final, dtype='uint')
    
    t_zoom = idx_zoom / FS
    
    # Gráfico de la señal original con ruido
    plt.plot(t_zoom, ecg_ruido_full[idx_zoom], label='ECG Original con Ruido', color='black', alpha=0.3, linestyle='-', linewidth=1)
    
    colores_comp = ['blue', 'green', 'orange', 'purple']
    estilos_comp = ['-', '--', ':', '-.']
    i = 0
    for name, ecg_f in ecg_filtrado.items():
        plt.plot(t_zoom, ecg_f[idx_zoom], 
                 label=f'Filtrado: {name}', color=colores_comp[i], linestyle=estilos_comp[i], linewidth=2)
        i += 1
        
    start_time = start_final / FS
    end_time = end_final / FS
    
    plt.title(f'Zona {idx+1}: Comparativa de los 4 Filtros en Región de Interés ')
    plt.suptitle(f'Región: {start_time:.2f}s a {end_time:.2f}s | Muestras: {start_final} a {end_final}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [unidades]')
    plt.grid(True, which='both', ls=':')
    
    # Aplicar la ubicación de la leyenda
    plt.legend(loc=legend_loc, fontsize='small')
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
print(f"--- {len(ZONAS_FINALES)} Gráficos de Zoom Completados. ---")

print("\n--- ¡Ejecución Completa!")