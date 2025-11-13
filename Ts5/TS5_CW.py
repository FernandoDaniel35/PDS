#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CW_Comparativa_Filtrado_FINAL_CORREGIDO.py

Análisis completo de 4 filtros para dos anchos de banda (250 Hz y 500 Hz).
Correcciones aplicadas:
1. Ajuste de beta para Kaiser para cumplir 80dB.
2. Ajuste del vector Remez para garantizar la atenuación.
3. Ajuste de la métrica de rizo en BP.
"""

import numpy as np
from scipy import signal as sig
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
import sys

# =========================================================================
# 1. Parámetros Globales y Carga
# =========================================================================

DATA_FILENAME = 'CW.wav'
N_FIR = 100  # Orden bajo para FIR
alpha_p = 1   # Atenuación máxima en banda de paso (dB)
alpha_s = 80  # Atenuación mínima en banda de detención (dB)
alpha_s_remez = 40.0 # Restricción relajada para Parks-McClellan a N=100

os.makedirs('resultados_CW_final', exist_ok=True)

try:
    print(f"Cargando {DATA_FILENAME}...")
    fs, signal_full = wavfile.read(DATA_FILENAME)
    if signal_full.ndim > 1:
        signal_full = signal_full[:, 0]
        
    audio_signal = signal_full.astype(np.float64) / np.max(np.abs(signal_full))
    
    FS = fs
    N_signal = len(audio_signal)
    print(f"Cargadas {N_signal} muestras. Frecuencia de muestreo: {FS} Hz")

except FileNotFoundError:
    print(f"¡Error! No se encontró el archivo '{DATA_FILENAME}'.")
    sys.exit()

# =========================================================================
# 2. Definición de Diseños por Ancho de Banda
# =========================================================================

# Definiciones de frecuencia para cada BW
DESIGN_PARAMS = {
    '250': {
        'BW_Hz': 250,
        'wp': (575, 825),       # Banda de Paso (Centro 700 Hz)
        'ws_iir': (525, 875),   # Estándar: T=50 Hz
        'ws_remez': (475, 925) # Optimizado: T=100 Hz
    },
    '500': {
        'BW_Hz': 500,
        'wp': (450, 950),       # Banda de Paso (Centro 700 Hz)
        'ws_iir': (400, 1000),  # Estándar: T=50 Hz
        'ws_remez': (350, 1050) # Optimizado: T=100 Hz
    }
}

# =========================================================================
# 3. Bucle Principal de Análisis
# =========================================================================

for BW_KEY, params in DESIGN_PARAMS.items():
    
    BW = params['BW_Hz']
    wp = params['wp']
    ws_iir = params['ws_iir']
    ws0_iir, ws1_iir = ws_iir
    ws0_remez, ws1_remez = params['ws_remez']
    wp0, wp1 = wp
    
    print(f"\n#################################################")
    print(f"## INICIANDO ANÁLISIS PARA BW = {BW} Hz ##")
    print(f"#################################################")

    # --- 3.1. Diseño de los 4 Filtros para el BW actual ---

    # 1. FIR Kaiser (Usa N=100, alpha_s=80dB, banda de transición Estándar)
    # CÁLCULO CORREGIDO de beta para alpha_s = 80dB:
    if alpha_s > 50:
        beta_kaiser = 0.1102 * (alpha_s - 8.7)
    elif alpha_s >= 21:
        beta_kaiser = 0.5842 * (alpha_s - 21)**0.4 + 0.07886 * (alpha_s - 21)
    else:
        beta_kaiser = 0.0
        
    f_corte_low = (wp0 + ws0_iir) / 2
    f_corte_high = (wp1 + ws1_iir) / 2
    b_boxcar = sig.firwin(numtaps=N_FIR, cutoff=[f_corte_low, f_corte_high], fs=FS, 
                              window=('kaiser', beta_kaiser), pass_zero=False)

    # 2. FIR Parks-McClellan (Remez) - as=40dB y banda de transición Optimizado
    # VECTOR REMEZ CORREGIDO: Se usa el peso para asegurar la atenuación
    f_remez = [0, ws0_remez, wp0, wp1, ws1_remez, FS/2]
    a_remez = [0, 1, 0] 
    # El ripple en la banda de paso (Rp) es alpha_p. El peso Ws/Wp = Rp/Rs.
    # Dado que Parks-McClellan no usa dB en sus pesos, estimamos el peso de detención (Ws)
    # usando la relación de atenuaciones deseadas (Rs/Rp).
    # Error en BP = 10^(alpha_p/20) - 1. Error en BS = 10^(-alpha_s_remez/20).
    # Peso = Error_BP / Error_BS
    remez_weight = (10**(alpha_p/20) - 1) / (10**(-alpha_s_remez/20))
    b_remez = sig.remez(numtaps=N_FIR, bands=f_remez, desired=a_remez, fs=FS, type='bandpass', 
                        weight=[remez_weight, 1, remez_weight]) # Se aplica el peso a ambas bandas de detención

    # Suprimir advertencias de diseño IIR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sig.BadCoefficients)
        
        # 3. IIR Chebyshev I (Usa alpha_s=80dB y banda de transición Estándar)
        N_cheb1, _ = sig.cheb1ord(wp, ws_iir, alpha_p, alpha_s, analog=False, fs=FS)
        mi_sos_cheb1 = sig.iirdesign(wp, ws_iir, gpass=alpha_p, gstop=alpha_s, 
                                     analog=False, ftype='cheby1', output='sos', fs=FS)
        
        # 4. IIR Cauer/Elíptico (Usa alpha_s=80dB y banda de transición Estándar)
        N_cauer, _ = sig.ellipord(wp, ws_iir, alpha_p, alpha_s, analog=False, fs=FS)
        mi_sos_cauer = sig.iirdesign(wp, ws_iir, gpass=alpha_p, gstop=alpha_s, 
                                     analog=False, ftype='ellip', output='sos', fs=FS)

    # --- 3.2. Estructura de Filtros y Aplicación ---
    
    FILTROS = {
        'FIR Kaiser (as=80dB, T=50Hz)': {'b': b_boxcar, 'a': 1, 'type': 'FIR', 'design_info': f'N={N_FIR}, as=80dB, T=50Hz'},
        'FIR Parks-McClellan (as=40dB, T=100Hz)': {'b': b_remez, 'a': 1, 'type': 'FIR', 'design_info': f'N={N_FIR}, as=40dB, T=100Hz'},
        'IIR Chebyshev I (as=80dB, T=50Hz)': {'sos': mi_sos_cheb1, 'type': 'IIR', 'design_info': f'N={N_cheb1}, L={len(mi_sos_cheb1)}'},
        'IIR Cauer/Elíptico (as=80dB, T=50Hz)': {'sos': mi_sos_cauer, 'type': 'IIR', 'design_info': f'N={N_cauer}, L={len(mi_sos_cauer)}'}
    }

    cw_filtrado = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sig.BadCoefficients) 
        
        for name, filter_params in FILTROS.items():
            print(f"Aplicando filtro: {name} (BW={BW} Hz)...")
            
            if filter_params['type'] == 'FIR':
                cw_filtrado[name] = sig.lfilter(filter_params['b'], filter_params['a'], audio_signal)
                
            elif filter_params['type'] == 'IIR':
                cw_filtrado[name] = sig.sosfilt(filter_params['sos'], audio_signal)

    print("\n--- Filtrado Completado. ---")
    
    # --- 3.3. Cálculo de Respuestas y Métricas ---
    
    freq_range_plot = np.linspace(wp0 - 150, wp1 + 150, 10000) # Rango de ploteo dinámico
    all_responses = {}
    performance_data = []

    for name, filter_params in FILTROS.items():
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning) 
            
            if filter_params['type'] == 'FIR':
                w, h = sig.freqz(filter_params['b'], filter_params['a'], worN=freq_range_plot, fs=FS)
                w_gd, gd = sig.group_delay((filter_params['b'], filter_params['a']), w=freq_range_plot, fs=FS)
            elif filter_params['type'] == 'IIR':
                w, h = sig.sosfreqz(filter_params['sos'], worN=freq_range_plot, fs=FS)
                b_tf, a_tf = sig.sos2tf(filter_params['sos'])
                w_gd, gd = sig.group_delay((b_tf, a_tf), w=freq_range_plot, fs=FS)
        
        mag_db = 20 * np.log10(np.abs(h) + 1e-8)
        phase = np.unwrap(np.angle(h))
        
        all_responses[name] = {'w': w, 'mag_db': mag_db, 'phase': phase, 'gd': gd, 
                               'design_info': filter_params['design_info']}
        
        # Extracción de Métricas (usando las bandas de detención propias de cada diseño)
        bp_mask = (w >= wp0) & (w <= wp1)
        
        # CÁLCULO CORREGIDO: Rizo máximo en BP (desviación absoluta de 0 dB)
        max_ripple = np.max(np.abs(mag_db[bp_mask])) if np.any(bp_mask) else np.nan
        
        if 'Parks-McClellan' in name:
            bs_low_mask = (w >= 0) & (w <= ws0_remez)
            bs_high_mask = (w >= ws1_remez) & (w <= FS/2)
        else:
            bs_low_mask = (w >= 0) & (w <= ws0_iir)
            bs_high_mask = (w >= ws1_iir) & (w <= FS/2)
            
        min_atten_low = -np.min(mag_db[bs_low_mask]) if np.any(bs_low_mask) else np.nan
        min_atten_high = -np.min(mag_db[bs_high_mask]) if np.any(bs_high_mask) else np.nan
        min_attenuation = np.min([min_atten_low, min_atten_high])
        
        orden_str = filter_params['design_info'].split(',')[0].strip()
        avg_gd = np.mean(gd[bp_mask]) if np.any(bp_mask) else np.nan
        
        performance_data.append({
            'Filtro': name,
            'Orden (N)': orden_str,
            'Rizo Máximo en BP (dB)': f"{max_ripple:.4f}",
            'Atenuación Mínima en BS (dB)': f"{min_attenuation:.1f}" if min_attenuation != np.nan else "N/A",
            'Retardo de Grupo Promedio (muestras)': f"{avg_gd:.1f}"
        })
        
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("/", "_")
        
        # --- 3.4. Gráficos Individuales (Magnitud, Fase, Retardo de Grupo) ---
        
        # Magnitud Individual (Gráfico 1-4)
        plt.figure(figsize=(8, 6))
        plt.plot(w, mag_db)
        plt.title(f'Magnitud: {name} (BW={BW} Hz)', fontsize=12)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud (dB)')
        # Se ajusta el límite inferior a 90 dB, ya que Kaiser y IIR deben cumplir 80 dB
        plt.ylim(-alpha_s - 10, 5) 
        
        plt.axhline(alpha_p, color='r', linestyle='--', alpha=0.5, label='Límite Bp (1 dB)')
        plt.axhline(-alpha_s, color='g', linestyle='--', alpha=0.5, label='Límite Bs (-80 dB)') 
        
        if 'Parks-McClellan' in name:
             plt.axhline(-alpha_s_remez, color='orange', linestyle='--', alpha=0.8, label=f'Límite Remez ({alpha_s_remez} dB)')
             plt.axvline(ws0_remez, color='m', linestyle=':', alpha=0.8, label=f'Bs Remez ({ws0_remez}/{ws1_remez}Hz)')
             plt.axvline(ws1_remez, color='m', linestyle=':', alpha=0.8)
             plt.legend(loc='upper right', fontsize=8)
        else:
             plt.axvline(ws0_iir, color='c', linestyle=':', alpha=0.5, label=f'Bs Estandar ({ws0_iir}/{ws1_iir}Hz)')
             plt.axvline(ws1_iir, color='c', linestyle=':', alpha=0.5)
             plt.legend(loc='upper right', fontsize=8)
        
        plt.axvline(wp0, color='b', linestyle=':', alpha=0.5)
        plt.axvline(wp1, color='b', linestyle=':', alpha=0.5, label='Banda de Paso')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(f'resultados_CW_final/BW{BW}_Magnitud_{safe_name}_INDIVIDUAL.png', dpi=300)
        plt.close() # Cierre de figura para evitar la advertencia de límite

        # Fase Individual (Gráfico 5-8)
        plt.figure(figsize=(8, 6))
        plt.plot(w, phase)
        plt.title(f'Fase: {name} (BW={BW} Hz)', fontsize=12)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Fase (radianes)')
        plt.axvline(wp0, color='r', linestyle=':', alpha=0.5)
        plt.axvline(wp1, color='r', linestyle=':', alpha=0.5, label='Banda de Paso')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'resultados_CW_final/BW{BW}_Fase_{safe_name}_INDIVIDUAL.png', dpi=300)
        plt.close() # Cierre de figura para evitar la advertencia de límite
        
        # Retardo de Grupo Individual (Gráfico 9-12)
        plt.figure(figsize=(8, 6))
        plt.plot(w, gd)
        plt.title(f'Retardo de Grupo: {name} (BW={BW} Hz)', fontsize=12)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Retardo (Muestras)')
        plt.axvline(wp0, color='r', linestyle=':', alpha=0.5)
        plt.axvline(wp1, color='r', linestyle=':', alpha=0.5, label='Banda de Paso')
        plt.ylim(0, 100)
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'resultados_CW_final/BW{BW}_RetardoGrupo_{safe_name}_INDIVIDUAL.png', dpi=300)
        plt.close() # Cierre de figura para evitar la advertencia de límite


    # --- 3.5. Gráficos Comparativos (Magnitud, Fase, Retardo de Grupo) ---
    
    # Magnitud Comparativa (Gráfico 13)
    plt.figure(figsize=(10, 8))
    plt.suptitle(f'Comparación de Magnitud de Filtros (BW={BW} Hz)', fontsize=14)
    for name, res in all_responses.items():
        plt.plot(res['w'], res['mag_db'], label=f'{name}')
    plt.title(f'Plantilla: Bp=[{wp0}-{wp1}]Hz. Remez: as={alpha_s_remez}dB, T=100Hz. Otros: as={alpha_s}dB, T=50Hz.') 
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.ylim(-alpha_s - 10, 5) 
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'resultados_CW_final/BW{BW}_Magnitud_COMPARATIVA.png', dpi=300)
    # No se cierra aquí, ya que los gráficos comparativos son menos frecuentes.


    # Fase Comparativa (Gráfico 14)
    plt.figure(figsize=(10, 8))
    plt.suptitle(f'Comparación de Fase de Filtros (BW={BW} Hz)', fontsize=14)
    for name, res in all_responses.items():
        plt.plot(res['w'], res['phase'], label=f'{name}')
    plt.title('Fase desenrollada $\\phi(\\omega)$') 
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Fase (radianes)')
    plt.axvline(wp0, color='r', linestyle=':', alpha=0.5)
    plt.axvline(wp1, color='r', linestyle=':', alpha=0.5, label='Banda de Paso')
    plt.legend(loc='lower left')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'resultados_CW_final/BW{BW}_Fase_COMPARATIVA.png', dpi=300)


    # Retardo de Grupo Comparativo (Gráfico 15)
    plt.figure(figsize=(10, 8))
    plt.suptitle(f'Comparación de Retardo de Grupo (BW={BW} Hz)', fontsize=14)
    for name, res in all_responses.items():
        plt.plot(res['w'], res['gd'], label=f'{name}')
    plt.title('Retardo de Grupo $\\tau_g(\\omega)$ (Muestras)') 
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Retardo (Muestras)')
    plt.axvline(wp0, color='r', linestyle=':', alpha=0.5)
    plt.axvline(wp1, color='r', linestyle=':', alpha=0.5, label='Banda de Paso')
    plt.ylim(0, 100) 
    plt.legend(loc='upper right')
    plt.grid(True, which="both", ls="--")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'resultados_CW_final/BW{BW}_RetardoGrupo_COMPARATIVO.png', dpi=300)
    
    
    # --- 3.6. Gráficos de Señal Filtrada (Tiempo) ---
    t = np.arange(N_signal) / FS
    y_original = audio_signal
    
    # Cuatro Gráficos Individuales (Gráfico 16-19)
    for i, (name, signal) in enumerate(cw_filtrado.items()):
        
        plt.figure(figsize=(10, 5))
        plt.suptitle(f'Señal Original vs. Filtrada con {name} (BW={BW} Hz)', fontsize=14)
        plt.plot(t, y_original, label='Señal Original (CW)', color='gray', alpha=0.7)
        
        linewidth = 2.5 if 'Parks-McClellan' in name else 1
        
        color = 'red' if 'Parks-McClellan' in name else ('blue' if 'Kaiser' in name else ('green' if 'Chebyshev' in name else 'purple'))
            
        plt.plot(t, signal, label=f'Filtrada con {name}', color=color, linewidth=linewidth)
        plt.title(f'Detalle Temporal de la Señal Filtrada ({name})', fontsize=12)
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Amplitud')
        plt.grid(True)
        plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("/", "_")
        plt.savefig(f'resultados_CW_final/BW{BW}_Senial_{safe_name}_INDIVIDUAL_TIEMPO.png', dpi=300)


    # Gráfico Comparativo (Gráfico 20)
    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Comparación de Señal Filtrada (BW={BW} Hz)', fontsize=14)

    plt.plot(t, y_original, label='Señal Original (CW)', color='black', alpha=0.5, linewidth=1)

    for name, signal in cw_filtrado.items():
        linewidth = 2.5 if 'Parks-McClellan' in name else 1
        color = 'red' if 'Parks-McClellan' in name else ('blue' if 'Kaiser' in name else ('green' if 'Chebyshev' in name else 'purple'))
        plt.plot(t, signal, label=f'Filtrada con {name}', color=color, linewidth=linewidth, linestyle='--')

    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=9)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'resultados_CW_final/BW{BW}_Senial_Filtrada_COMPARATIVA_TIEMPO.png', dpi=300)


    # --- 3.7. Imprimir Tabla de Rendimiento ---
    tabla_performance = pd.DataFrame(performance_data)
    print(f"\n--- TABLA DE RENDIMIENTO NUMÉRICO (BW={BW} Hz) ---\n")
    print(tabla_performance.to_string(index=False)) 
    tabla_performance.to_csv(f'resultados_CW_final/BW{BW}_Performance_Tabla.csv', index=False)


# =========================================================================
# 4. Mostrar todos los gráficos creados
# =========================================================================
plt.show()