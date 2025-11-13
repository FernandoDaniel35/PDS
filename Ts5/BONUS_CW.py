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
# Esto genera el RuntimeWarning que observaste, pero permite que se generen todos los gráficos.
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
N_FIR = 1000        # Orden alto para flancos abruptos
alpha_p = 0.5       # Atenuación máxima en banda de paso (dB)
alpha_s = 80        # Atenuación mínima en banda de detención (dB) para IIR
alpha_s_remez = 30.0 # Restricción relajada para Parks-McClellan a N=1000

os.makedirs('resultados_CW_final', exist_ok=True)

try:
    print(f"Cargando {DATA_FILENAME}...")
    fs, signal_full = wavfile.read(DATA_FILENAME)
    if signal_full.ndim > 1:
        signal_full = signal_full[:, 0]
        
    # Normalizar para evitar clipping al escribir el archivo WAV
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

DESIGN_PARAMS = {
    '50': { 
        'BW_Hz': 50,
        # wp = [FC - 25, FC + 25] -> [914.91, 964.91]
        'wp': (FC_TONE - 25, FC_TONE + 25),       
        # ws_iir = [FC - 75, FC + 75] (T=50Hz) -> [864.91, 1014.91]
        'ws_iir': (FC_TONE - 75, FC_TONE + 75),   
        # ws_remez = [FC - 125, FC + 125] (T=100Hz) -> [814.91, 1064.91]
        'ws_remez': (FC_TONE - 125, FC_TONE + 125) 
    },
    '100': {
        'BW_Hz': 100,
        'wp': (FC_TONE - 50, FC_TONE + 50),       
        'ws_iir': (FC_TONE - 100, FC_TONE + 100),   
        'ws_remez': (FC_TONE - 150, FC_TONE + 150) 
    },
    '250': {
        'BW_Hz': 250,
        'wp': (FC_TONE - 125, FC_TONE + 125),       
        'ws_iir': (FC_TONE - 175, FC_TONE + 175),   
        'ws_remez': (FC_TONE - 225, FC_TONE + 225) 
    },
    '500': {
        'BW_Hz': 500,
        'wp': (FC_TONE - 250, FC_TONE + 250),       
        'ws_iir': (FC_TONE - 300, FC_TONE + 300),  
        'ws_remez': (FC_TONE - 350, FC_TONE + 350) 
    }
}

# Almacén de todas las señales filtradas (para la sección de audio final)
all_filtered_signals = {}
all_filtered_signals['Original'] = audio_signal 

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
    print(f"## INICIANDO ANÁLISIS PARA BW = {BW} Hz (FC={FC_TONE:.2f} Hz) ##")
    print(f"#################################################")

    # --- 3.1. Diseño de los 4 Filtros para el BW actual ---

    # 1. FIR Kaiser
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

    # 2. FIR Parks-McClellan (Remez)
    f_remez = [0, ws0_remez, wp0, wp1, ws1_remez, FS/2]
    a_remez = [0, 1, 0] 
    remez_weight = (10**(alpha_p/20) - 1) / (10**(-alpha_s_remez/20))
    b_remez = sig.remez(numtaps=N_FIR, bands=f_remez, desired=a_remez, fs=FS, type='bandpass', 
                        weight=[remez_weight, 1, remez_weight]) 

    # Suprimir advertencias de diseño IIR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sig.BadCoefficients)
        
        # 3. IIR Chebyshev I
        N_cheb1, _ = sig.cheb1ord(wp, ws_iir, alpha_p, alpha_s, analog=False, fs=FS)
        mi_sos_cheb1 = sig.iirdesign(wp, ws_iir, gpass=alpha_p, gstop=alpha_s, 
                                     analog=False, ftype='cheby1', output='sos', fs=FS)
        
        # 4. IIR Cauer/Elíptico
        N_cauer, _ = sig.ellipord(wp, ws_iir, alpha_p, alpha_s, analog=False, fs=FS)
        mi_sos_cauer = sig.iirdesign(wp, ws_iir, gpass=alpha_p, gstop=alpha_s, 
                                     analog=False, ftype='ellip', output='sos', fs=FS)

    # --- 3.2. Estructura de Filtros y Aplicación ---
    
    FILTROS_DESIGN = {
        'FIR Kaiser': {'b': b_boxcar, 'a': 1, 'type': 'FIR', 'design_info': f'N={N_FIR}, as=80dB, T=50Hz'},
        'FIR Parks-McClellan': {'b': b_remez, 'a': 1, 'type': 'FIR', 'design_info': f'N={N_FIR}, as=30dB, T=100Hz'},
        'IIR Chebyshev I': {'sos': mi_sos_cheb1, 'type': 'IIR', 'design_info': f'N={N_cheb1}, L={len(mi_sos_cheb1)}'},
        'IIR Cauer/Elíptico': {'sos': mi_sos_cauer, 'type': 'IIR', 'design_info': f'N={N_cauer}, L={len(mi_sos_cauer)}'}
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
    
    # Rango de frecuencia para IIR: alrededor de la banda de paso
    freq_range_plot_iir = np.linspace(wp0 - 150, wp1 + 150, 10000)
    
    # Rango de frecuencia para FIR Kaiser y Parks-McClellan (0 Hz a 3 kHz)
    freq_range_plot_fir = np.linspace(0, 3000, 10000) 

    all_responses = {}
    performance_data = []

    for name, filter_params in FILTROS_DESIGN.items():
        
        # Determinar el rango de plot y los límites X para la figura
        if 'FIR' in name: # Aplica a Kaiser y Parks-McClellan
            freq_range_plot = freq_range_plot_fir
            xlim_low, xlim_high = 0, 3000 
        else: # Chebyshev I y Cauer/Elíptico
            freq_range_plot = freq_range_plot_iir
            xlim_low, xlim_high = wp0 - 150, wp1 + 150 
            
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
        
        if 'Parks-McClellan' in name:
            bs_low_mask = (w >= 0) & (w <= ws0_remez)
            bs_high_mask = (w >= ws1_remez) & (w <= FS/2)
            alpha_s_current = alpha_s_remez 
        else:
            bs_low_mask = (w >= 0) & (w <= ws0_iir)
            bs_high_mask = (w >= ws1_iir) & (w <= FS/2)
            alpha_s_current = alpha_s 
            
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
        
        # --- Gráfico de Magnitud Individual para CADA FILTRO ---
        plt.figure(figsize=(8, 6))
        plt.plot(w, mag_db)
        plt.title(f'Magnitud: {name} (BW={BW} Hz, FC={FC_TONE:.2f} Hz)', fontsize=12)
        plt.xlabel('Frecuencia (Hz)')
        plt.ylabel('Magnitud (dB)')
        plt.xlim(xlim_low, xlim_high) 
        plt.ylim(-alpha_s - 10, 5) 
        plt.axhline(alpha_p, color='r', linestyle='--', alpha=0.5, label=f'Límite Bp ({alpha_p} dB)')
        plt.axhline(-alpha_s_current, color='g', linestyle='--', alpha=0.5, label=f'Límite Bs (-{alpha_s_current} dB)') 
        
        if 'Parks-McClellan' in name:
             plt.axvline(ws0_remez, color='m', linestyle=':', alpha=0.8, label=f'Bs Remez ({ws0_remez:.2f}/{ws1_remez:.2f}Hz)')
             plt.axvline(ws1_remez, color='m', linestyle=':', alpha=0.8)
             plt.legend(loc='upper right', fontsize=8)
        else:
             plt.axvline(ws0_iir, color='c', linestyle=':', alpha=0.5, label=f'Bs Estandar ({ws0_iir:.2f}/{ws1_iir:.2f}Hz)')
             plt.axvline(ws1_iir, color='c', linestyle=':', alpha=0.5)
             plt.legend(loc='upper right', fontsize=8)
        
        plt.axvline(wp0, color='b', linestyle=':', alpha=0.5)
        plt.axvline(wp1, color='b', linestyle=':', alpha=0.5, label='Banda de Paso')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(f'resultados_CW_final/BW{BW}_Magnitud_{safe_name}_INDIVIDUAL.png', dpi=300)


        # --- Gráficos Individuales Combinados (Magnitud y Fase) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        fig.suptitle(f'Respuesta de Frecuencia: {name} (BW={BW} Hz) - Centrado en {FC_TONE:.2f} Hz', fontsize=14)

        # Subplot 1: Magnitud
        ax1.plot(w, mag_db)
        ax1.set_title('Magnitud (dB)', fontsize=12)
        ax1.set_ylabel('Magnitud (dB)')
        ax1.set_xlim(xlim_low, xlim_high) 
        ax1.set_ylim(-alpha_s - 10, 5) 
        
        ax1.axhline(alpha_p, color='r', linestyle='--', alpha=0.5, label=f'Límite Bp ({alpha_p} dB)')
        ax1.axhline(-alpha_s_current, color='g', linestyle='--', alpha=0.5, label=f'Límite Bs (-{alpha_s_current} dB)') 
        
        if 'Parks-McClellan' in name:
             ax1.axvline(ws0_remez, color='m', linestyle=':', alpha=0.8, label=f'Bs Remez ({ws0_remez:.2f}/{ws1_remez:.2f}Hz)')
             ax1.axvline(ws1_remez, color='m', linestyle=':', alpha=0.8)
             ax1.legend(loc='upper right', fontsize=8)
        else:
             ax1.axvline(ws0_iir, color='c', linestyle=':', alpha=0.5, label=f'Bs Estandar ({ws0_iir:.2f}/{ws1_iir:.2f}Hz)')
             ax1.axvline(ws1_iir, color='c', linestyle=':', alpha=0.5)
             ax1.legend(loc='upper right', fontsize=8)
        
        ax1.axvline(wp0, color='b', linestyle=':', alpha=0.5)
        ax1.axvline(wp1, color='b', linestyle=':', alpha=0.5, label='Banda de Paso')
        ax1.grid(True, which="both", ls="--")

        # Subplot 2: Fase
        ax2.plot(w, phase, color='red')
        ax2.set_title('Fase (rad)', fontsize=12)
        ax2.set_xlabel('Frecuencia (Hz)')
        ax2.set_ylabel('Fase (radianes)')
        ax2.axvline(wp0, color='r', linestyle=':', alpha=0.5)
        ax2.axvline(wp1, color='r', linestyle=':', alpha=0.5)
        ax2.grid(True, which="both", ls="--")
        ax2.set_xlim(xlim_low, xlim_high) 
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'resultados_CW_final/BW{BW}_Magnitud_Fase_{safe_name}_COMBINADO.png', dpi=300)


    # --- Gráficos Comparativos (SÓLO Magnitud para el BW actual) ---
    
    plt.figure(figsize=(10, 8))
    plt.suptitle(f'Comparación de Magnitud de Filtros (BW={BW} Hz) - Centrado en {FC_TONE:.2f} Hz', fontsize=14)
    
    xlim_comparative = 3000
    
    for name, res in all_responses.items():
        plt.plot(res['w'], res['mag_db'], label=f'{name}')
        
    plt.title(f'Plantilla: Bp=[{wp0:.2f}-{wp1:.2f}]Hz. Remez: as={alpha_s_remez}dB, T=100Hz. Otros: as={alpha_s}dB, T=50Hz.') 
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.xlim(0, xlim_comparative) 
    plt.ylim(-alpha_s - 10, 5) 
    plt.legend(loc='lower left', fontsize=9)
    plt.grid(True, which="both", ls="--")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'resultados_CW_final/BW{BW}_Magnitud_COMPARATIVA.png', dpi=300)


    # --- 3.6. Gráficos de Señal Filtrada (Tiempo) ---
    t = np.arange(N_signal) / FS
    y_original = audio_signal
    
    # Cuatro Gráficos Individuales 
    for i, (name, signal) in enumerate(cw_filtrado.items()):
        
        plt.figure(figsize=(10, 5))
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
# 5. Reproductores de Audio para Comparación (ESTRUCTURA CUÁDRUPLE COLUMNA) 🎧
# =========================================================================

if JUPYTER_ENV:
    
    TEMP_DIR = 'temp_audio_files'
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    print("\n\n#############################################")
    print("## REPRODUCTORES DE AUDIO PARA COMPARACIÓN ##")
    print("#############################################")
    
    FILTROS_BASE = ['FIR Kaiser', 'FIR Parks-McClellan', 'IIR Chebyshev I', 'IIR Cauer/Elíptico']

    # --- 5.1. Reproductor Original (Fila 1) ---
    print("\n\n--- Señal Original ---")
    display(Audio(all_filtered_signals['Original'], rate=FS))


    # --- Helper para guardar y obtener el reproductor HTML ---
    def get_audio_html(signal, name):
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("/", "_")
        temp_filename = os.path.join(TEMP_DIR, f"{safe_name}.wav")
        audio_int16 = (signal * 32767).astype(np.int16)
        wavfile.write(temp_filename, FS, audio_int16)
        return Audio(temp_filename, rate=FS)._repr_html_()
    # --------------------------------------------------------

    # --- 5.2. Reproductores Comparativos (Filas 2-5) ---
    print("\n\n--- Comparativas de Ancho de Banda (500Hz vs 250Hz vs 100Hz vs 50Hz) ---")
    
    for base_name in FILTROS_BASE:
        
        # Obtener señales
        signal_500 = all_filtered_signals.get(f"500Hz_{base_name}")
        signal_250 = all_filtered_signals.get(f"250Hz_{base_name}")
        signal_100 = all_filtered_signals.get(f"100Hz_{base_name}")
        signal_50 = all_filtered_signals.get(f"50Hz_{base_name}") 

        # Generar HTML de audio
        html_500 = get_audio_html(signal_500, f"500Hz_{base_name}")
        html_250 = get_audio_html(signal_250, f"250Hz_{base_name}")
        html_100 = get_audio_html(signal_100, f"100Hz_{base_name}")
        html_50 = get_audio_html(signal_50, f"50Hz_{base_name}") 

        # Combinar en una estructura HTML con 4 columnas (aprox. 24% de ancho cada una)
        combined_html = f"""
        <div style="border: 2px solid #0056b3; padding: 10px; margin-top: 20px; border-radius: 5px;">
            <h4 style="text-align: center; color: #0056b3; margin-top: 0; border-bottom: 1px solid #ccc; padding-bottom: 5px;">
                {base_name}
            </h4>
            <div style="display: flex; justify-content: space-between; clear: both;">
                
                <div style="width: 24%; padding: 5px; border-right: 1px solid #eee;">
                    <p style="font-weight: bold; margin-bottom: 5px; color: #333;">BW = 500Hz</p>
                    {html_500}
                </div>
                
                <div style="width: 24%; padding: 5px; border-right: 1px solid #eee;">
                    <p style="font-weight: bold; margin-bottom: 5px; color: #333;">BW = 250Hz</p>
                    {html_250}
                </div>
                
                <div style="width: 24%; padding: 5px; border-right: 1px solid #eee;">
                    <p style="font-weight: bold; margin-bottom: 5px; color: #333;">BW = 100Hz</p>
                    {html_100}
                </div>

                <div style="width: 24%; padding: 5px;">
                    <p style="font-weight: bold; margin-bottom: 5px; color: #333;">BW = 50Hz (Máx. Selectividad)</p>
                    {html_50}
                </div>
            </div>
        </div>
        """
        # Mostrar el bloque HTML completo
        display(HTML(combined_html))
        
    print("\nArchivos temporales de audio generados en la carpeta: " + TEMP_DIR)
    
# =========================================================================
# 6. Limpieza de Archivos Temporales
# =========================================================================
    pass