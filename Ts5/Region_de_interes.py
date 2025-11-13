import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy import signal as sig

# =========================================================================
# 1. PARÁMETROS Y CARGA DE DATOS (Tomado de Ts4.py)
# =========================================================================
fs = 1000  # Hz. Frecuencia de muestreo
N_samples = 15000 # Número de muestras a usar (para el análisis de ejemplo)

try:
    # Asegúrate de tener 'ECG_TP4.mat' en el mismo directorio
    mat = sio.loadmat('ECG_TP4.mat')
    ecg_full = np.squeeze(mat['ecg_lead'])
except FileNotFoundError:
    print("Error: No se encontró el archivo 'ECG_TP4.mat'. Usando datos simulados.")
    t = np.arange(0, N_samples) / fs
    # Señal simulada con ruido (para emular la necesidad de filtrado)
    ecg_full = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*15*t) + 2*np.random.randn(N_samples) 

# Variables de contexto necesarias para que el código no falle, aunque no se usen
ecg_one_lead = ecg_full[:N_samples] # Señal ECG a analizar
cant_muestras = len(ecg_one_lead)    # Longitud de la señal usada
demora = 0 # Valor ficticio para evitar errores en la línea comentada
ECG_f_win = np.zeros_like(ecg_one_lead) # Array ficticio para evitar errores
ECG_f_butt = np.zeros_like(ecg_one_lead) # Array ficticio para evitar errores

print(f"Usando {cant_muestras} muestras del ECG original.")

# =========================================================================
# 2. VISUALIZACIÓN DE REGIONES (Tu código original sin descomentar)
# =========================================================================

print("\nGenerando gráficos de regiones de interés (solo ECG original)...")

###################################
#%% Regiones de interés con ruido #
###################################
 
regs_interes = ( 
        [4000, 5500], # muestras
        [10e3, 11e3], # muestras
        )
 
for ii in regs_interes:
    
    # intervalo limitado de 0 a cant_muestras
    start_idx = int(np.max([0, ii[0]]))
    end_idx = int(np.min([cant_muestras, ii[1]]))
    zoom_region = np.arange(start_idx, end_idx, dtype='uint')
    
    plt.figure(1)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    #plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window')
    
    plt.title('ECG filtering example from ' + str(ii[0]) + ' to ' + str(ii[1]) )
    plt.ylabel('Amplitud [V]')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    # axes_hdl.legend() # Comentado si solo se muestra 1 línea
    axes_hdl.set_yticks(())
    plt.tight_layout()
            
    plt.show()
 
###################################
#%% Regiones de interés sin ruido #
###################################
 
regs_interes = ( 
        np.array([5, 5.2]) *60*fs, # minutos a muestras
        np.array([12, 12.4]) *60*fs, # minutos a muestras
        np.array([15, 15.2]) *60*fs, # minutos a muestras
        )

# Aseguramos que los puntos de inicio/fin estén dentro de cant_muestras (15000)
# Ajusto los minutos de ejemplo para que caigan dentro de las 15 muestras disponibles (15s)
max_t_s = cant_muestras / fs # Tiempo máximo en segundos

regs_interes_ajustados = []
for start_t, end_t in regs_interes:
    start_idx = int(start_t)
    end_idx = int(end_t)
    
    # Solo incluimos si el final está dentro de la señal cargada
    if end_idx < cant_muestras:
        regs_interes_ajustados.append((start_idx, end_idx))
    else:
         print(f"Advertencia: Región ({start_idx}, {end_idx}) fuera del rango de {cant_muestras} muestras y fue omitida.")

for start_idx, end_idx in regs_interes_ajustados:
    
    # intervalo limitado de 0 a cant_muestras
    zoom_region = np.arange(start_idx, end_idx, dtype='uint')
    
    plt.figure(2)
    plt.plot(zoom_region, ecg_one_lead[zoom_region], label='ECG', linewidth=2)
    #plt.plot(zoom_region, ECG_f_butt[zoom_region], label='Butterworth')
    plt.plot(zoom_region, ECG_f_win[zoom_region + demora], label='FIR Window') # Esta línea debe estar comentada si no quieres ver el filtro FIR
    
    plt.title(f'ECG original de {start_idx} a {end_idx} muestras')
    plt.ylabel('Amplitud [V]')
    plt.xlabel('Muestras (#)')
    
    axes_hdl = plt.gca()
    # axes_hdl.legend() # Comentado si solo se muestra 1 línea
    axes_hdl.set_yticks(())
    plt.tight_layout()
            
    plt.show()