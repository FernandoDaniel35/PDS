import matplotlib.pyplot as plt
from matplotlib import  patches
from pytc2.sistemas_lineales import  plot_plantilla
# https://pytc2.readthedocs.io/en/latest/notebooks/principal.html#


# --- Respuesta en frecuencia ---
w, h = signal.freqz_sos(mi_sos, worN=np.logspace(-2, 1.9, 1000), fs = fs)  # 10 Hz a 1 MHz aprox.
# w, h = signal.freqz_sos(mi_sos, fs = fs)  # Calcula la respuesta en frecuencia del filtro

# --- Cálculo de fase y retardo de grupo ---
phase = np.unwrap(np.angle(h))
# Retardo de grupo = -dφ/dω
w_rad = w / (fs/2) * np.pi
gd = -np.diff(phase) / np.diff(w_rad)

# --- Polos y ceros ---
z, p, k = signal.sos2zpk(mi_sos)

# --- Gráficas ---
# plt.figure(figsize=(12,10))

# Magnitud
plt.subplot(3,1,1)
plt.plot(w, 20*np.log10(abs(h)), label = f_aprox)
plot_plantilla(filter_type = 'bandpass', fpass = wp, ripple = alpha_p , fstop = ws, attenuation = alpha_s, fs = fs)

plt.title('Respuesta en Magnitud')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('|H(jω)| [dB]')
plt.grid(True, which='both', ls=':')
# plt.axis([0, 80, -60, 5 ]);

plt.legend()

# Fase
plt.subplot(3,1,2)
plt.plot(w, np.degrees(phase), label = f_aprox)
plt.title('Fase')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Fase [°]')
plt.grid(True, which='both', ls=':')
plt.legend()

# Retardo de grupo
plt.subplot(3,1,3)
plt.plot(w[:-1], gd, label = f_aprox)
plt.title('Retardo de Grupo')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('τg [# muestras]')
plt.grid(True, which='both', ls=':')
plt.legend()


# Diagrama de polos y ceros
plt.figure(figsize=(10,10))
plt.plot(np.real(p), np.imag(p), 'x', markersize=10, label=f'{f_aprox} Polos' )
axes_hdl = plt.gca()

if len(z) > 0:
    plt.plot(np.real(z), np.imag(z), 'o', markersize=10, fillstyle='none', label=f'{f_aprox} Ceros')
plt.axhline(0, color='k', lw=0.5)
plt.axvline(0, color='k', lw=0.5)
unit_circle = patches.Circle((0, 0), radius=1, fill=False,
                             color='gray', ls='dotted', lw=2)
axes_hdl.add_patch(unit_circle)

plt.axis([-1.1, 1.1, -1.1, 1.1])
plt.title('Diagrama de Polos y Ceros (plano s)')
plt.xlabel(r'$\Re(z)$')
plt.ylabel(r'$\Im(z)$')
plt.legend()
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()