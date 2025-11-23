#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CW_Comparativa_Filtrado_FINAL_COMPLETO_V9.py

Análisis para 50 Hz, 100 Hz, 250 Hz y 500 Hz, centrado en 939.91 Hz.
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
# 0. CONFIGURACIÓN DE GRÁFICOS Y ENTORNO
# =========================================================================
# Configuramos Matplotlib para permitir más de 20 figuras abiertas sin advertencia.
plt.rcParams['figure.max_open_warning'] = 100

# Importación específica para reproductores de audio en Jupyter
try:
    from IPython.display import Audio, display, HTML
    JUPYTER_ENV = True
except ImportError:
    JUPYTER_ENV = False
    print("Advertencia: No se detectó IPython.display. Los reproductores de audio no se mostrarán.")


# =========================================================================
# 1. Parámetros Globales y Carga
# =========================================================================

DATA_FILENAME = 'CW.wav'
N_FIR = 4000 	 # Orden alto para forzar 80 dB de atenuación
alpha_p = 0.5 	 # Atenuación máxima en banda de paso (dB) para IIR
alpha_p_fir = 1.0	# Atenuación máxima en banda de paso (dB) para FIR
alpha_s = 80 	 # Atenuación mínima en banda de detención (dB) para IIR
alpha_s_remez = 80.0 # Restricción más estricta para Parks-McClellan

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
# 2. Definición de Diseños por Ancho de Banda (50 Hz, 100 Hz, 250 Hz, 500 Hz)
# =========================================================================

FC_TONE = 939.91 

# ---Anchos de Banda de Transición ---
T_FIR_KAISER = 30 	
T_FIR_REMEZ = 50 	


DESIGN_PARAMS = {
    '50': { 
        'BW_Hz': 50,
        # wp = [FC - 25, FC + 25]
        'wp': (FC_TONE - 25, FC_TONE + 25), 	
        # ws_iir (Mínimo requerido para IIR, T=50Hz)
        'ws_iir': (FC_TONE - 75, FC_TONE + 75), 	
        # ws_fir (Nueva, T=30Hz)
        'ws_fir': (FC_TONE - (25 + T_FIR_KAISER), FC_TONE + (25 + T_FIR_KAISER)), 
        # ws_remez (Nueva, T=50Hz)
        'ws_remez': (FC_TONE - (25 + T_FIR_REMEZ), FC_TONE + (25 + T_FIR_REMEZ)) 
    },
    '100': {
        'BW_Hz': 100,
        # wp = [FC - 50, FC + 50]
        'wp': (FC_TONE - 50, FC_TONE + 50), 	
        # ws_iir (Mínimo requerido para IIR, T=50Hz)
        'ws_iir': (FC_TONE - 100, FC_TONE + 100), 	
        # ws_fir (Nueva, T=30Hz)
        'ws_fir': (FC_TONE - (50 + T_FIR_KAISER), FC_TONE + (50 + T_FIR_KAISER)), 
        # ws_remez (Nueva, T=50Hz)
        'ws_remez': (FC_TONE - (50 + T_FIR_REMEZ), FC_TONE + (50 + T_FIR_REMEZ)) 
    },
    '250': {
        'BW_Hz': 250,
        # wp = [FC - 125, FC + 125]
        'wp': (FC_TONE - 125, FC_TONE + 125), 	
        # ws_iir (Mínimo requerido para IIR, T=50Hz)
        'ws_iir': (FC_TONE - 175, FC_TONE + 175), 	
        # ws_fir (Nueva, T=30Hz)
        'ws_fir': (FC_TONE - (125 + T_FIR_KAISER), FC_TONE + (125 + T_FIR_KAISER)), 
        # ws_remez (Nueva, T=50Hz)
        'ws_remez': (FC_TONE - (125 + T_FIR_REMEZ), FC_TONE + (125 + T_FIR_REMEZ)) 
    },
    '500': {
        'BW_Hz': 500,
        # wp = [FC - 250, FC + 250]
        'wp': (FC_TONE - 250, FC_TONE + 250), 	
        # ws_iir (Mínimo requerido para IIR, T=50Hz)
        'ws_iir': (FC_TONE - 300, FC_TONE + 300), 
        # ws_fir (Nueva, T=30Hz)
        'ws_fir': (FC_TONE - (250 + T_FIR_KAISER), FC_TONE + (250 + T_FIR_KAISER)), 
        # ws_remez (Nueva, T=50Hz)
        'ws_remez': (FC_TONE - (250 + T_FIR_REMEZ), FC_TONE + (250 + T_FIR_REMEZ)) 
    }
}

# Guardo todas las señales filtradas (para la sección de audio final)
all_filtered_signals = {}
all_filtered_signals['Original'] = audio_signal 

# =========================================================================
# 3. Bucle Principal de Análisis
# =========================================================================

# MODIFICACIÓN APLICADA: Cambiar la iteración sobre DESIGN_PARAMS para ordenar las claves
# de mayor a menor ancho de banda ('500', '250', '100', '50').
for BW_KEY in ['500', '250', '100', '50']:
    params = DESIGN_PARAMS[BW_KEY]
    
    BW = params['BW_Hz']
    wp = params['wp']
    ws_iir = params['ws_iir']
    ws_fir = params['ws_fir'] 
    ws0_iir, ws1_iir = ws_iir
    ws0_fir, ws1_fir = ws_fir 
    ws0_remez, ws1_remez = params['ws_remez']
    wp0, wp1 = wp
    
    print(f"\n#################################################")
    print(f"## INICIANDO ANÁLISIS PARA BW = {BW} Hz (FC={FC_TONE:.2f} Hz) ##")
    print(f"#################################################")

    # --- 3.1. Diseño de los 4 Filtros para el BW actual ---

    # 1. FIR Kaiser
    # Usa alpha_s = 80 dB
    if alpha_s > 50:
        beta_kaiser = 0.1102 * (alpha_s - 8.7)
    elif alpha_s >= 21:
        beta_kaiser = 0.5842 * (alpha_s - 21)**0.4 + 0.07886 * (alpha_s - 21)
    else:
        beta_kaiser = 0.0
        
    f_corte_low = (wp0 + ws0_fir) / 2
    f_corte_high = (wp1 + ws1_fir) / 2
    b_boxcar = sig.firwin(numtaps=N_FIR, cutoff=[f_corte_low, f_corte_high], fs=FS, 
                              window=('kaiser', beta_kaiser), pass_zero=False)
    

    # 2. FIR Parks-McClellan (Remez)
    f_remez = [0, ws0_remez, wp0, wp1, ws1_remez, FS/2]
    a_remez = [0, 1, 0] 
    # Peso Remez se calcula en base a alpha_p_fir (1.0 dB) y alpha_s_remez (80.0 dB)
    remez_weight = (10**(alpha_p_fir/20) - 1) / (10**(-alpha_s_remez/20)) 
    b_remez_raw = sig.remez(numtaps=N_FIR, bands=f_remez, desired=a_remez, fs=FS, type='bandpass', 
                            weight=[remez_weight, 1, remez_weight]) 
    
    # --- Normalización de Remez (Mantenida sin cambios) ---
    w_remez_test, h_remez_test = sig.freqz(b_remez_raw, 1, worN=10000, fs=FS)
    bp_mask_test = (w_remez_test >= wp0) & (w_remez_test <= wp1)
    if np.any(bp_mask_test):
        max_gain_bp = np.max(np.abs(h_remez_test[bp_mask_test]))
        if max_gain_bp > 0:
            b_remez = b_remez_raw / max_gain_bp # Normalizamos para que la ganancia máxima en BP sea 1 (0 dB)
        else:
            b_remez = b_remez_raw
            print(f"Advertencia: Ganancia cero detectada en Remez para BW={BW}Hz.")
    else:
        b_remez = b_remez_raw
        print(f"Advertencia: Máscara BP vacía para Remez en BW={BW}Hz. No se normalizó.")
    # ----------------------------------------------------

    # Suprimir advertencias de diseño IIR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sig.BadCoefficients)
        
        # 3. IIR Chebyshev I (Usa alpha_p = 0.5 dB y alpha_s = 80 dB)
        N_cheb1, _ = sig.cheb1ord(wp, ws_iir, alpha_p, alpha_s, analog=False, fs=FS)
        mi_sos_cheb1 = sig.iirdesign(wp, ws_iir, gpass=alpha_p, gstop=alpha_s, 
                                     analog=False, ftype='cheby1', output='sos', fs=FS)
        
        # 4. IIR Cauer/Elíptico (Usa alpha_p = 0.5 dB y alpha_s = 80 dB)
        N_cauer, _ = sig.ellipord(wp, ws_iir, alpha_p, alpha_s, analog=False, fs=FS)
        mi_sos_cauer = sig.iirdesign(wp, ws_iir, gpass=alpha_p, gstop=alpha_s, 
                                     analog=False, ftype='ellip', output='sos', fs=FS)

    # --- 3.2. Estructura de Filtros y Aplicación ---
    
    FILTROS_DESIGN = {
        'FIR Kaiser': {'b': b_boxcar, 'a': 1, 'type': 'FIR', 'design_info': f'N={N_FIR}, ap={alpha_p_fir}dB, as={alpha_s}dB, T={T_FIR_KAISER}Hz'}, 
        'FIR Parks-McClellan': {'b': b_remez, 'a': 1, 'type': 'FIR', 'design_info': f'N={N_FIR}, ap={alpha_p_fir}dB, as={alpha_s_remez}dB, T={T_FIR_REMEZ}Hz'}, 
        'IIR Chebyshev I': {'sos': mi_sos_cheb1, 'type': 'IIR', 'design_info': f'N={N_cheb1}, ap={alpha_p}dB, as={alpha_s}dB, L={len(mi_sos_cheb1)}'}, 
        'IIR Cauer/Elíptico': {'sos': mi_sos_cauer, 'type': 'IIR', 'design_info': f'N={N_cauer}, ap={alpha_p}dB, as={alpha_s}dB, L={len(mi_sos_cauer)}'} 
    }
    
    cw_filtrado = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sig.BadCoefficients) 
        
        for name, filter_params in FILTROS_DESIGN.items():
            print(f"Aplicando filtro: {name} (BW={BW} Hz)...")
            
            if filter_params['type'] == 'FIR':
                signal_out = sig.lfilter(filter_params['b'], filter_params['a'], audio_signal)
                
            elif filter_params['type'] == 'IIR':
                signal_out = sig.sosfilt(filter_params['sos'], audio_signal)
            
            # Normalizar la salida y guardar en el diccionario general
            max_amp = np.max(np.abs(signal_out))
            normalized_signal = signal_out / max_amp if max_amp > 0 else signal_out
            
            # Usar un nombre único para el almacenamiento general
            full_name = f"{BW_KEY}Hz_{name}"
            all_filtered_signals[full_name] = normalized_signal
            cw_filtrado[name] = normalized_signal 

    print("\n--- Filtrado Completado. ---")
    
    # --- 3.3. Cálculo de Respuestas y Métricas y Generación de Gráficos ---
    
    # Rango de frecuencia forzado para TODOS los plots de respuesta de frecuencia: 500 Hz a 1400 Hz
    freq_range_plot_all = np.linspace(500, 1400, 10000)

    all_responses = {}
    performance_data = []

    for name, filter_params in FILTROS_DESIGN.items():
        
        # Determinar el rango de plot y los límites X para la figura
        # *** MODIFICADO PARA FORZAR RANGO DE 500Hz a 1400Hz ***
        freq_range_plot = freq_range_plot_all
        xlim_low, xlim_high = 500, 1400
        # ******************************************************
        
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
                               'design_info': filter_params['design_info'], 
                               'xlim_low': xlim_low, 'xlim_high': xlim_high} 
        
        # Extracción de Métricas
        bp_mask = (w >= wp0) & (w <= wp1)
        max_ripple = np.max(np.abs(mag_db[bp_mask])) if np.any(bp_mask) else np.nan
        
        # Definir la atenuación de paso (alpha_p) y los límites de banda de rechazo (ws)
        if filter_params['type'] == 'FIR':
            alpha_p_current = alpha_p_fir # 1.0 dB
            alpha_s_current = alpha_s_remez # 80dB
            if 'Parks-McClellan' in name:
                bs_low_mask = (w >= 0) & (w <= ws0_remez) 
                bs_high_mask = (w >= ws1_remez) & (w <= FS/2) 
                ws_low_line, ws_high_line = ws0_remez, ws1_remez
                ws_label = f'Bs Remez ({ws0_remez:.2f}/{ws1_remez:.2f}Hz)'
            else: # Kaiser
                alpha_s_current = alpha_s # 80dB
                bs_low_mask = (w >= 0) & (w <= ws0_fir) 
                bs_high_mask = (w >= ws1_fir) & (w <= FS/2) 
                ws_low_line, ws_high_line = ws0_fir, ws1_fir
                ws_label = f'Bs Kaiser ({ws0_fir:.2f}/{ws1_fir:.2f}Hz)'
        else: # IIR
            alpha_p_current = alpha_p # 0.5 dB
            alpha_s_current = alpha_s # 80dB
            bs_low_mask = (w >= 0) & (w <= ws0_iir)
            bs_high_mask = (w >= ws1_iir) & (w <= FS/2)
            ws_low_line, ws_high_line = ws0_iir, ws1_iir
            ws_label = f'Bs IIR Estandar ({ws0_iir:.2f}/{ws1_iir:.2f}Hz)'
            
        min_atten_low = -np.min(mag_db[bs_low_mask]) if np.any(bs_low_mask) else np.nan
        min_atten_high = -np.min(mag_db[bs_high_mask]) if np.any(bs_high_mask) else np.nan
        min_attenuation = np.min([min_atten_low, min_atten_high])
        
        orden_str = filter_params['design_info'].split(',')[0].strip()
        avg_gd = np.mean(gd[bp_mask]) if np.any(bp_mask) else np.nan
        
        performance_data.append({
            'Filtro': name,
            'Orden (N)': orden_str,
            'Rizo Máx en BP (dB)': f"{max_ripple:.4f}",
            'Atenuación Mín en BS (dB)': f"{min_attenuation:.1f}" if min_attenuation != np.nan else "N/A",
            'Retardo de Grupo Prom (muestras)': f"{avg_gd:.1f}"
        })
        
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("/", "_")
        
        # --- Gráfico de Magnitud Individual para CADA FILTRO ---
        fig_height_individual = 8 
        plt.figure(figsize=(8, fig_height_individual))
        
        plt.plot(w, mag_db)
        plt.title(f'Magnitud: {name} (BW={BW} Hz, FC={FC_TONE:.2f} Hz)', fontsize=12)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud (dB)')
        plt.xlim(xlim_low, xlim_high) 
        plt.ylim(-alpha_s - 10, alpha_p_fir + 1) 
        
        # Línea de límite de Banda de Paso (usa alpha_p_current)
        plt.axhline(alpha_p_current, color='r', linestyle='--', alpha=0.5, label=f'Límite Bp ({alpha_p_current} dB)') 
        
        # Línea de límite de Banda de Rechazo
        plt.axhline(-alpha_s_current, color='g', linestyle='--', alpha=0.5, label=f'Límite Bs (-{alpha_s_current} dB)') 
        
        # Líneas de Banda de Rechazo (ws)
        plt.axvline(ws_low_line, color='m' if 'Parks-McClellan' in name else ('c' if 'Kaiser' in name else 'k'), 
                        linestyle=':', alpha=0.8, label=ws_label)
        plt.axvline(ws_high_line, color='m' if 'Parks-McClellan' in name else ('c' if 'Kaiser' in name else 'k'), 
                        linestyle=':', alpha=0.8)

        # Líneas de Banda de Paso (wp)
        plt.axvline(wp0, color='b', linestyle=':', alpha=0.5)
        plt.axvline(wp1, color='b', linestyle=':', alpha=0.5, label='Banda de Paso')
        plt.legend(loc='upper right', fontsize=8) 
        
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(f'resultados_CW_final/BW{BW}_Magnitud_{safe_name}_INDIVIDUAL.png', dpi=300)


    # --- Gráficos Comparativos (SÓLO Magnitud para el BW actual) ---
    
    plt.figure(figsize=(12, 8)) # Altura de referencia: 8
    plt.suptitle(f'Comparación de Magnitud de Filtros (BW={BW} Hz) - Centrado en {FC_TONE:.2f} Hz', fontsize=14)
    
    xlim_comparative = 1400 
    
    for name, res in all_responses.items():
        plt.plot(res['w'], res['mag_db'], label=f'{name}')
        
    plt.title(f'Plantilla: Bp=[{wp0:.2f}-{wp1:.2f}]Hz. as=80dB. FIR: ap={alpha_p_fir}dB, T={T_FIR_KAISER}/{T_FIR_REMEZ}Hz. IIR: ap={alpha_p}dB.') 
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.xlim(500, xlim_comparative) 
    plt.ylim(-alpha_s - 10, alpha_p_fir + 1) 
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'resultados_CW_final/BW{BW}_Magnitud_COMPARATIVA.png', dpi=300)


    # --- 3.6. Gráficos de Señal Filtrada (Tiempo) ---
    t = np.arange(N_signal) / FS
    y_original = audio_signal
    
    # Cuatro Gráficos Individuales 
    for i, (name, signal) in enumerate(cw_filtrado.items()):
        
        plt.figure(figsize=(12, 8))
        plt.suptitle(f'Señal Original vs. Filtrada con {name} (BW={BW} Hz) - Tono {FC_TONE:.2f} Hz', fontsize=14)
        plt.plot(t, y_original, label='Señal Original (CW)', color='gray', alpha=0.7)
        
        linewidth = 2.5 if 'FIR' in name else 1
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


    # Gráfico Comparativo 
    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Comparación de Señal Filtrada (BW={BW} Hz) - Tono {FC_TONE:.2f} Hz', fontsize=14)

    plt.plot(t, y_original, label='Señal Original (CW)', color='black', alpha=0.5, linewidth=1)

    for name, signal in cw_filtrado.items():
        linewidth = 2.5 if 'FIR' in name else 1
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

# =========================================================================
# 5. Reproductores de Audio para Comparación y Generación de HTML 
# =========================================================================

TEMP_DIR = 'temp_audio_files'
os.makedirs(TEMP_DIR, exist_ok=True) # Aseguramos la creación de la carpeta
HTML_FILENAME = 'audio_players.html'

FILTROS_BASE = ['FIR Kaiser', 'FIR Parks-McClellan', 'IIR Chebyshev I', 'IIR Cauer/Elíptico']

def get_audio_path_and_save(signal, name):
    """Guarda la señal y devuelve la ruta relativa para el HTML."""
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("/", "_")
    temp_filename = os.path.join(TEMP_DIR, f"{safe_name}.wav")
    
    audio_int16 = (signal * 32767).astype(np.int16)
    wavfile.write(temp_filename, FS, audio_int16)
    
    # Devuelve la ruta relativa que el HTML debe usar
    return os.path.join(TEMP_DIR, f"{safe_name}.wav")

def generate_html_players():
    """Genera el contenido HTML con reproductores de audio, con los 4 anchos de banda en una fila."""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>Comparación de Filtros de Audio</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .container {{ border: 2px solid #0056b3; padding: 15px; margin-bottom: 30px; border-radius: 8px; }}
            h2, h4 {{ color: #0056b3; border-bottom: 1px solid #ccc; padding-bottom: 5px; }}
            /* MODIFICACIÓN CLAVE: Usar flexbox para mantener los 4 elementos en una fila, distribuidos equitativamente. */
            .grid {{ display: flex; flex-wrap: nowrap; justify-content: space-between; gap: 10px; }} 
            .col {{ 
                flex: 1 1 25%; /* Cada columna ocupa el 25% del espacio */
                padding: 10px; 
                box-sizing: border-box; 
                text-align: center; /* Centrar el contenido de la columna */
            }}
            .col:not(:last-child) {{ border-right: 1px solid #eee; }}
            p {{ font-weight: bold; margin-bottom: 5px; color: #333; font-size: 0.9em; }}
        </style>
    </head>
    <body>
        <h1> Comparativa de Señales Filtradas ({FC_TONE:.2f} Hz)</h1>
        <p><strong>Rutas:</strong> El archivo <code>{HTML_FILENAME}</code> asume que la carpeta <code>{TEMP_DIR}</code> está en el mismo nivel.</p>
        <p><strong>Nota para Notebooks:</strong> Esta tabla HTML se genera y muestra aquí. Si la ves en GitHub (estáticamente), los reproductores deben funcionar si subiste la carpeta <code>{TEMP_DIR}</code>.</p>

        <div class="container">
            <h2>Señal Original (Arriba de todo, sin filtro)</h2>
            <audio controls src="{get_audio_path_and_save(all_filtered_signals['Original'], 'Original')}">
                Su navegador no soporta el elemento de audio.
            </audio>
        </div>

        <h2>Comparativas por Tipo de Filtro (4 anchos de banda por fila)</h2>
    """
    
    # Definir el orden de los anchos de banda, de mayor a menor (de izquierda a derecha)
    # MODIFICACIÓN: La generación del HTML ya estaba en orden descendente ('500', '250', '100', '50'). Se mantiene.
    BW_ORDER = ['500', '250', '100', '50']
    
    # Recorrer todos los tipos de filtro
    for base_name in FILTROS_BASE:
        
        # Generar el bloque HTML para el filtro actual
        combined_html = f"""
        <div class="container">
            <h4>{base_name}</h4>
            <div class="grid">
        """
        
        # Recorrer los anchos de banda en el orden deseado
        for bw_key in BW_ORDER:
            
            # Obtener el nombre completo y la ruta del archivo
            full_name = f"{bw_key}Hz_{base_name}"
            path = get_audio_path_and_save(all_filtered_signals.get(full_name), full_name)
            label = f"BW = {bw_key}Hz"
            if bw_key == '50':
                 label += " (Máx. Selectividad)"
            elif bw_key == '500':
                 label += " (Mín. Selectividad)"
            
            # Añadir la columna al bloque HTML
            combined_html += f"""
                <div class="col">
                    <p>{label}</p>
                    <audio controls src="{path}">Su navegador no soporta audio.</audio>
                </div>
            """

        combined_html += """
            </div>
        </div>
        """
        html_content += combined_html

    html_content += """
    </body>
    </html>
    """
    return html_content

# --- Ejecución de Generación y Guardado ---

# 1. Ejecutar el guardado de todos los archivos WAV y generar el HTML
final_html_content = generate_html_players()

# 2. Guardar el archivo HTML
try:
    with open(HTML_FILENAME, 'w') as f:
        f.write(final_html_content)
    print(f"\n####################################################")
    print(f"## ÉXITO: Archivo HTML generado para Navegador/GitHub ##")
    print(f"####################################################")
    print(f"1. Archivos WAV guardados en: '{TEMP_DIR}'")
    print(f"2. Archivo HTML generado: '{HTML_FILENAME}'")
    
    if JUPYTER_ENV:
        print("\n3. Visualizando la tabla de reproductores en el Notebook (salida estática para GitHub):")
        # 3. Mostrar la salida interactiva si es Jupyter
        display(HTML(final_html_content))
    else:
        print(f"\nPara ver los reproductores, abra el archivo '{HTML_FILENAME}' en su navegador.")

except IOError as e:
    print(f"\nError al escribir el archivo HTML: {e}")

# =========================================================================
# 6. Limpieza de Archivos Temporales
# =========================================================================
    pass