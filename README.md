# PDS

### Tareas semanales para la materia Procesamiento Digital de Señales de 6to año de Ingeniería Electrónica de la UTN FRBA del segundo cuatrimestre de 2025

**Ts1:** En esta primera tarea realizamos un generador de señales senoidales utilizando Python en Spyder de Ubuntu.

**Ts2:** Utilizando el generador de señales senoidales de la tarea anterior, calculamos su DFT y la graficamos.
Luego le sumamos ruido uniforme de varianza $\sigma^2 = 4$. y observamos el resultado en forma gráfica.
Reemplazamos el ruido uniforme por ruido Gaussiano y graficamos nuevamente.

**Ts3:** Estimación de Amplitud y Frecuencia de una Señal.

**Ts4:** Primeras nociones de estimación espectral: Ancho de banda de señales

**Ts5:**  Filtrado digital lineal de ECG (1ra parte): Usando el archivo ecg.mat que contiene un registro electrocardiográfico (ECG) registrado durante una prueba de esfuerzo, junto con una serie de variables descriptas a continuación. Diseñe y aplique los filtros digitales necesarios para mitigar las siguientes fuentes de contaminación:

    Ruido causado por el movimiento de los electrodos (Alta frecuencia).
    Ruido muscular (Alta frecuencia).
    Movimiento de la línea de base del ECG, inducido en parte por la respiración (Baja frecuencia).
    
    Bonus:
    💎 Proponga algún tipo de señal, ya sea de la TS anterior u otra que no haya sido analizada y repita el análisis. No 
    olvide explicar su origen y cómo fue digitalizada.

    **Agregué una señal de telegrafía inmersa en ruido, grabada por mi celular de la salida de un equipo   
    Kenwood TS-120s y realicé el diseño de los 4 filtros. Grafiqué el resultado en magnitud y luego grafiqué la señal en 
    el tiempo de cada salida filtrada. Los audios se encuentran en la carpeta temp_audio_files.**
 
