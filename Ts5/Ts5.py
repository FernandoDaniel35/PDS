#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 10:16:37 2025

@author: fer
"""

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
N_SAMPLES_TO_USE = 600000

# Parámetros de la Región de Interés (AHORA SE DEFINE POR MÚLTIPLES ZONAS)
REGION_START = 50000 # [Muestra de inicio - Se mantiene para compatibilidad]
REGION_END = 55000 # [Muestra de fin - Se mantiene para compatibilidad]

# Zonas de interés a graficar (4 zonas)
# --- Zonas basadas en minutos
ZONAS_MINUTOS_RAW = (np.array([5, 5.2]) * 60 * FS,)
# --- Zonas basadas en muestras fijas
ZONAS_MUESTRAS_FIJAS = (
    [101 * FS, 118 * FS],      # 101s a 118s
    [247 * FS, 242 * FS],      # 247s a 242s (orden invertido)
    [247 * FS, 252 * FS],      # 247s a 252s
)

# Convertir todas las zonas a tuplas de enteros de muestra y combinarlas
ZONAS_DE_INTERES = []

for start_end in ZONAS_MINUTOS_RAW:
    # Convertir a entero y agregar
    ZONAS_DE_INTERES.append((int(start_end[0]), int(start_end[1])))

for start_end in ZONAS_MUESTRAS_FIJAS:
    # Asegurar que sean enteros
    ZONAS_DE_INTERES.append((int(start_end[0]), int(start_end[1])))


# --- REORDENAMIENTO DE ZONAS SOLICITADO ---
# Orden original (índices): [0] = 5-5.2min, [1] = 101-118s, [2] = 247-242s, [3] = 247-252s
# Nuevo orden solicitado: 101s, 247-242s, 247-252s, 5-5.2min (300s)
if len(ZONAS_DE_INTERES) >= 4:
    ZONAS_DE_INTERES = [
        ZONAS_DE_INTERES[1],  # 101s a 118s (Ahora es la Zona 1)
        ZONAS_DE_INTERES[2],  # 247s a 242s (Ahora es la Zona 2)
        ZONAS_DE_INTERES[3],  # 247s a 252s (Ahora es la Zona 3)
        ZONAS_DE_INTERES[0]   # 5min a 5.2min (Ahora es la Zona 4 - ÚLTIMA)
    ]


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
        N_SAMPLES_TO_USE = len(ecg_ruido_full)
        ecg_ruido_full = ecg_ruido_full
        print(f"Advertencia: El archivo tiene menos de 600000. Usando {N_SAMPLES_TO_USE} muestras.")

except (FileNotFoundError, KeyError, ValueError) as e:
    print(f"¡Error! No se pudo cargar o procesar '{DATA_FILENAME}': {e}")
    N_SAMPLES_TO_USE = 10000 
    t = np.arange(0, N_SAMPLES_TO_USE) / FS
    ecg_ruido_full = np.sin(2 * np.pi * 1 * t) + 1.5 * np.sin(2 * np.pi * 50 * t) 
    print("Usando ECG simulado de emergencia con ruido.")

N_signal = len(ecg_ruido_full)

# Ajuste de límites si las zonas exceden la longitud de la señal (sin imprimir advertencias)
for i in range(len(ZONAS_DE_INTERES)):
    start, end = ZONAS_DE_INTERES[i]
    if end > N_signal:
        ZONAS_DE_INTERES[i] = (start, N_signal - 1)
    elif start >= end:
        # Asegurar que el inicio sea menor que el fin, ajustando a un rango fijo si es necesario
        ZONAS_DE_INTERES[i] = (max(0, end - 5000), end)

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
# 6. Verificación de la Respuesta en Frecuencia (Punto c) 
# =========================================================================

print("\n--- Generando Gráficos de Verificación de Respuesta en Frecuencia (Punto c) ---")

# --- 6.1 Cálculo de la Respuesta en Frecuencia y Plantilla ---
respuestas = {}
eps = 1e-8
for name, params in FILTROS.items():
    if params['type'] == 'FIR':
        # Uso de N_FREQZ_POINTS para suavizar la curva
        w, h = sig.freqz(params['b'], params['a'], fs=fs, worN=N_FREQZ_POINTS)
    elif params['type'] == 'IIR':
        # Uso de N_FREQZ_POINTS para suavizar la curva
        w, h = sig.sosfreqz(params['sos'], fs=fs, worN=N_FREQZ_POINTS)
    
    # Calcular fase y retardo de grupo
    fase = np.unwrap(np.angle(h))
    # w * 2 * np.pi convierte w de Hz a rad/s
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
    plt.figure(figsize=(10, 10))
    
    # Magnitud [dB]
    ax1 = plt.subplot(3, 1, 1)
    w_zoom = res['w'][res['w'] <= 41] 
    h_zoom = res['h'][res['w'] <= 41] 
    ax1.plot(w_zoom, 20*np.log10(abs(h_zoom) + eps), label=name, color='blue', linewidth=2)
    plot_plantilla(ax1)
    ax1.set_title(f'Magnitud y Plantilla de Diseño - {name}')
    ax1.set_ylabel('|H(jω)| [dB]')
    ax1.set_ylim([-60, 5])
    ax1.legend(loc='lower left')
    
    # Fase [rad]
    ax2 = plt.subplot(3, 1, 2)
    w_zoom = res['w'][res['w'] <= 41] 
    fase_zoom = res['fase'][res['w'] <= 41] 
    ax2.plot(w_zoom, fase_zoom, label=name, color='orange', linewidth=2)
    ax2.set_title(f'Fase - {name}')
    ax2.set_ylabel('Fase [rad]')
    ax2.grid(True, which='both', ls=':')
    ax2.set_xlim([0, 41]) 
    
    # Retardo de Grupo [muestras]
    ax3 = plt.subplot(3, 1, 3)
    # 1. Array de frecuencias del Retardo de Grupo (size N-1)
    w_gd_full = res['w'][:-1] 
    # 2. Máscara de zoom para el array de tamaño N-1
    gd_mask_zoom = w_gd_full <= 41 
    # 3. Aplicar la máscara a los arrays de tamaño N-1
    w_gd_zoom = w_gd_full[gd_mask_zoom]
    gd_zoom = res['group_delay'][gd_mask_zoom]
    # 4. Plotear
    ax3.plot(w_gd_zoom, gd_zoom, label=name, color='green', linewidth=2)
    ax3.set_title(f'Retardo de Grupo (Group Delay) - {name}')
    ax3.set_xlabel('Frecuencia [Hz]')
    ax3.set_ylabel('Retardo [muestras]')
    ax3.grid(True, which='both', ls=':')
    ax3.set_xlim([0, 41]) 
    ax3.set_ylim([0, 1500])
    
    plt.suptitle(f'Verificación de Respuesta en Frecuencia - {name}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# --- 6.4 Gráfica de Verificación Comparativa (1 Figura) ---
plt.figure(figsize=(12, 12))

# Subplot 1: Magnitud [dB] - Comparativa
ax_comp_mag = plt.subplot(3, 1, 1)
colores_comp = ['blue', 'green', 'orange', 'purple']
estilos_comp = ['-', '--', ':', '-.']
i = 0
for name, res in respuestas.items():
    w_zoom = res['w'][res['w'] <= 41] 
    h_zoom = res['h'][res['w'] <= 41] 
    ax_comp_mag.plot(w_zoom, 20*np.log10(abs(h_zoom) + eps), label=name, color=colores_comp[i], linestyle=estilos_comp[i], linewidth=2)
    i += 1
plot_plantilla(ax_comp_mag)
ax_comp_mag.set_title('Respuesta en Magnitud - Comparativa de los 4 Filtros vs. Plantilla')
ax_comp_mag.set_ylabel('|H(jω)| [dB]')
ax_comp_mag.set_ylim([-60, 5])
ax_comp_mag.legend(loc='lower left')

# Subplot 2: Fase [rad] - Comparativa
ax_comp_fase = plt.subplot(3, 1, 2)
i = 0
for name, res in respuestas.items():
    w_zoom = res['w'][res['w'] <= 41] 
    fase_zoom = res['fase'][res['w'] <= 41] 
    ax_comp_fase.plot(w_zoom, fase_zoom, label=name, color=colores_comp[i], linestyle=estilos_comp[i], linewidth=2)
    i += 1
ax_comp_fase.set_title('Fase - Comparativa de los 4 Filtros')
ax_comp_fase.set_ylabel('Fase [rad]')
ax_comp_fase.grid(True, which='both', ls=':')
ax_comp_fase.set_xlim([0, 41]) 
ax_comp_fase.legend(loc='upper right')

# Subplot 3: Retardo de Grupo [muestras] - Comparativa
ax_comp_gd = plt.subplot(3, 1, 3)
i = 0
for name, res in respuestas.items():
    # 1. Array de frecuencias del Retardo de Grupo (size N-1)
    w_gd_full = res['w'][:-1] 
    # 2. Máscara de zoom para el array de tamaño N-1
    gd_mask_zoom = w_gd_full <= 41 
    # 3. Aplicar la máscara
    w_gd_zoom = w_gd_full[gd_mask_zoom]
    gd_zoom = res['group_delay'][gd_mask_zoom]

    ax_comp_gd.plot(w_gd_zoom, gd_zoom, label=name, color=colores_comp[i], linestyle=estilos_comp[i], linewidth=2)
    i += 1
ax_comp_gd.set_title('Retardo de Grupo (Group Delay) - Comparativa de los 4 Filtros')
ax_comp_gd.set_xlabel('Frecuencia [Hz]')
ax_comp_gd.set_ylabel('Retardo [muestras]')
ax_comp_gd.grid(True, which='both', ls=':')
ax_comp_gd.set_xlim([0, 41]) 
ax_comp_gd.set_ylim([0, 1500])
ax_comp_gd.legend(loc='upper right')

plt.suptitle('Comparativa General de Respuesta en Frecuencia (Punto c)', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# =========================================================================
# 7. Gráficos Adicionales en el Tiempo (Solicitud del Usuario)
# =========================================================================

# Nuevas zonas de interés solicitadas
regs_interes_raw = ( 
    np.array([2, 2.2]) *60*fs, # minutos a muestras
    np.array([5, 5.2]) *60*fs, # minutos a muestras (Ya estaba, pero se incluye para completar la lista)
    np.array([10, 10.2]) *60*fs, # minutos a muestras
    np.array([12, 12.4]) *60*fs, # minutos a muestras
    np.array([15, 15.2]) *60*fs, # minutos a muestras
    np.array([18, 18.2]) *60*fs, # minutos a muestras
    [4000, 5500], # muestras
    [10e3, 11e3], # muestras
    # ZONAS AGREGADAS POR SOLICITUD DEL USUARIO:
    [4000, 5500], # muestras (Duplicado para cumplir con la solicitud explícita)
    [10000, 11000], # muestras (Duplicado para cumplir con la solicitud explícita)
)

# Convertir las zonas a tuplas de enteros de muestra
REGS_DE_INTERES_ADICIONALES = []

for start_end in regs_interes_raw:
    start_s = int(start_end[0])
    end_s = int(start_end[1])
    # Asegurar orden y límites
    start_final = max(0, min(start_s, end_s))
    end_final = min(N_signal - 1, max(start_s, end_s))
    
    if end_final > start_final:
        REGS_DE_INTERES_ADICIONALES.append((start_final, end_final))

print(f"\n--- Generando {len(REGS_DE_INTERES_ADICIONALES)} Gráficos de Zoom Adicionales para las Nuevas Zonas de Interés ---")

for idx, (REGION_START_ZONA, REGION_END_ZONA) in enumerate(REGS_DE_INTERES_ADICIONALES):
    
    # ---------------------------------------------------------------------
    # Gráfica Individual de la Zona (Original vs. 4 Filtros)
    # ---------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Intervalo limitado de 0 a N_signal
    idx_zoom = np.arange(REGION_START_ZONA, REGION_END_ZONA, dtype='uint')
    
    t_zoom = idx_zoom / FS
    
    # Gráfico de la señal original con ruido
    plt.plot(t_zoom, ecg_ruido_full[idx_zoom], label='ECG Original con Ruido', color='black', alpha=0.3, linestyle='-', linewidth=1)
    
    colores_comp = ['blue', 'green', 'orange', 'purple']
    estilos_comp = ['-', '--', ':', '-.']
    i = 0
    for name, ecg_f in ecg_filtrado.items():
        # Asegurar que los datos filtrados coincidan con el índice
        plt.plot(t_zoom, ecg_f[idx_zoom], 
                  label=f'Filtrado: {name}', color=colores_comp[i], linestyle=estilos_comp[i], linewidth=2)
        i += 1
        
    start_s = REGION_START_ZONA / FS
    end_s = REGION_END_ZONA / FS
    
    plt.title(f'Zona Adicional {idx+1}: Comparativa de los 4 Filtros en Región de Interés')
    plt.suptitle(f'Región: {start_s:.2f}s a {end_s:.2f}s | Muestras: {REGION_START_ZONA} a {REGION_END_ZONA}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [unidades]')
    plt.grid(True, which='both', ls=':')
    plt.legend(loc='lower left', fontsize='small')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
print("--- Gráficos de Zoom Adicionales Completados. ---")

# =========================================================================
# 5. Visualización de Resultados (Evaluación - Punto d) 
# =========================================================================

# --- 5.1 Gráficas de Señal por Zona de Interés ---
print(f"\n--- Generando {len(ZONAS_DE_INTERES)} Gráficos de Zoom para las Zonas de Interés Originales ---")

for idx, (REGION_START_ZONA, REGION_END_ZONA) in enumerate(ZONAS_DE_INTERES):
    
    # ---------------------------------------------------------------------
    # Gráfica Individual de la Zona (Original vs. 4 Filtros)
    # ---------------------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Intervalo limitado de 0 a N_signal
    idx_zoom = np.arange(np.max([0, REGION_START_ZONA]), np.min([N_signal, REGION_END_ZONA]), dtype='uint')
    
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
        
    start_s = REGION_START_ZONA / FS
    end_s = REGION_END_ZONA / FS
    
    plt.title(f'Zona {idx+1}: Comparativa de los 4 Filtros en Región de Interés')
    plt.suptitle(f'Región: {start_s:.2f}s a {end_s:.2f}s | Muestras: {REGION_START_ZONA} a {REGION_END_ZONA}')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud [unidades]')
    plt.grid(True, which='both', ls=':')
    plt.legend(loc='lower left', fontsize='small')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
print("--- Gráficos de Zoom Originales Completados. ---")

print("\n--- ¡Ejecución Completa!")