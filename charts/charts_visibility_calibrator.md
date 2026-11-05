# Calibración de visibilities en el nodo: `cudaVisibilityCalibrator`

Estado al 2026-10-27. Tarea de Notion: *Calibración en nodo* (sección *Calibración de correlator*), en el Registro de tareas.

## 1. Objetivo y decisiones de diseño

Aplicar en la GPU las gains instrumentales a las visibilities, justo después del correlador N2 Astron, sin tocar el kernel del correlador ni el baseband (que se calibra offline).

| Decisión | Motivo |
|---|---|
| Calibrar sobre la matriz de visibilities, no sobre los voltajes | Los voltajes son `int4+int4` y el kernel Astron ya es funcional con ese formato. Pasar a float y re-cuantizar sería fuente de errores. La salida del correlador ya es `int32`, que sí se puede operar. |
| Un solo comando nuevo, `cudaVisibilityCalibrator`, entre `cudaCorrelatorAstron` y `cudaSyncOutput` | Mantiene la arquitectura modular. No cambia ningún stage ni config existente, y quien no agregue el comando no cambia nada. |
| Los productos de baseline `1/(g_i g_j*)` se calculan offline en charts-utils | La GPU solo hace una multiplicación compleja por visibilidad. |
| Calibración in-place en `int32`, factor `complex64`, redondeo al más cercano con saturación | `chartsAccumulate` y `rawFileWrite` no cambian (mismo frame size). Decidido con el usuario. |
| Gains promedio en el tiempo (`nanmean`), una tabla estática | Se recalibra cada ~2 días, no en tiempo real. Conviene revisar `phase_scatter_deg` y `amp_scatter` de `cal_pipeline_gpu.py` para confirmar que el promedio es seguro. |
| Gains NaN o cero dan factor 0 | Los datos marcados (`--flag-freqs`) se anulan en vez de propagar NaN. |
| Antenas sin gains quedan con gain 1 | Pasan sin calibrar. Ojo: una baseline entre antena calibrada y no calibrada queda a medias. |

## 2. Layout de datos (lo que hay que respetar)

- Frame del correlador: `[freq_local][baseline][pol_a][pol_b][re, im]`, todo `int32`.
- `baseline = recv_a*(recv_a+1)/2 + recv_b` con `recv_a >= recv_b` (triángulo inferior). Con 64 antenas: 32 receivers × 2 pols = 528 bloques, igual que `num_blocks`.
- Elemento raw `r`: receiver `r//2`, pol `r%2`.
- **Orden de antenas invertido:** la antena CHARTS `a` es el elemento raw `N-1-a` (`charts_utils/core/antenna_indices.py`).
- Convención `V_ab = x_a · conj(x_b)`. `apply_gains` calcula `V/(g_i g_j*)`.
- Dos shards de frecuencia (`corr_freq_0` y `corr_freq_1`), 336 canales cada uno. La tabla es de banda completa (672 canales) y cada comando toma su rebanada con `freq_offset` (0 y 336).

## 3. Qué se implementó

### kotekan (`/home/jpcontreras/kotekan`)
- `lib/cuda/cudaVisibilityCalibrator.{hpp,cpp,cu}` y su registro en `lib/cuda/CMakeLists.txt`.
  - Lee al iniciar un `.npy` `complex64` C-contiguo de forma `(n_freq, num_blocks, 2, 2)`, valida forma y rango de canales, y sube a la GPU solo su rebanada.
  - Kernel: un hilo por visibilidad, `float32` con `fmaf`, redondeo `__float2int_rn` con saturación a `[-2^31, 2147483520]`.
  - Sin `calibration_file` no lanza ningún kernel (pass-through).
  - Exige `block_size: 2` y `num_data_sets: 1`; falla al iniciar si no se cumplen.
  - Parámetros: `gpu_mem_correlation_matrix`, `num_local_freq`, `num_blocks`, `calibration_file` (opcional), `freq_offset` (default 0).
- `charts/config/64antennas_corr_cal.yaml`: copia de `64antennas_corr.yaml` con el comando agregado en ambas listas, con `freq_offset` 0 y 336 y `calibration_file: /data/calibration/vis_cal_64.npy` (ruta de ejemplo, ajustar). El config original no se modificó.

### charts-utils (`/home/jpcontreras/charts-utils`)
- `charts_utils/cal/kotekan_calibration.py`:
  - `expand_gains(gains, antennas, n_ant)` promedia el eje de tiempo si existe y rellena con 1 las antenas no resueltas.
  - `gains_to_visibility_calibration(gains)` devuelve la tabla `complex64 (n_freq, 528, 2, 2)` en el layout del frame GPU.
- `scripts/kotekan_vis_calibrator.py` (CLI: `--gains`, `--antennas`, `--num-antennas`, `--output`).
- `tests/test_kotekan_calibration.py` (6 tests; usa `apply_gains` y `gpu_corr_to_visibilities` como oráculo).

### Tests y benchmark (`test_charts/`)
- `sim/corr_cal.py`: genera el YAML de replay (`rawFileRead → cudaProcess → sink`), lanza kotekan y decodifica frames.
- `tests/test_visibility_calibrator.py`: 8 tests end-to-end contra el binario real y la GPU.
- `sim/benchmark_vis_calibrator.py` y el subcomando `charts_sim.py bench-vis-cal`.

## 4. Cómo usarlo

```bash
# 1. Resolver gains (charts-utils) y guardarlas promediadas
python scripts/cal_pipeline_gpu.py --input <frames_dir> --save-avg-gains -o gains.npy ...

# 2. Construir la tabla para la GPU (antenas del archivo en índices CHARTS)
python scripts/kotekan_vis_calibrator.py -g gains.npy -ant 0,4,6,8 --num-antennas 64 -o vis_cal_64.npy

# 3. Poner la ruta en charts/config/64antennas_corr_cal.yaml y correr kotekan con ese config
```

## 5. Cómo se validó

Todo con el binario real en la RTX 5090.

| Chequeo | Resultado |
|---|---|
| Sin tabla frente a no tener el comando | Bit-exacto |
| GPU frente a `round(vis × tabla)` | Error máximo 0.71 cuentas (límite de `float32` con valores de ~10⁷) |
| GPU frente a `apply_gains` de charts-utils, `freq_offset` 0 y 336 | Error relativo ~4×10⁻⁶ |
| Saturación | Se satura en los límites de `int32`, sin dar la vuelta |
| Tabla de forma incorrecta u offset fuera de rango | kotekan se detiene al iniciar con mensaje claro |

Ejecutar los tests:

```bash
export CHARTS_TEST_LD_LIBRARY_PATH=<dir con un libnvrtc que coincida con el driver>   # ver nota 7.1
python -m pytest test_charts/tests/test_visibility_calibrator.py -q
python -m pytest tests/test_kotekan_calibration.py -q      # en charts-utils
```

## 6. Rendimiento

`python test_charts/charts_sim.py bench-vis-cal --frames 4000 --repetitions 3 --output-dir <dir>` mide con los trackers de kotekan (`/trackers`, requiere `gpu.profiling: true`; guarda solo las últimas 100 muestras, así que se consulta repetidamente) y con `nvidia-smi` a 100 ms. Resultados (RTX 5090, 64 antenas, 336 canales, 1536 muestras/frame):

| | Sin calibrador | Con calibrador |
|---|---|---|
| Correlador Astron | 194.4 µs | 194.3 µs |
| **Calibrador** | no aplica | **5.8 µs** (p99 5.8, máx 7.2) |
| Tiempo de kernels por frame | 225.8 µs | 231.5 µs |
| Ritmo del pipeline (tiempo real: 195.3 Hz) | 238.6 Hz | 238.3 Hz |
| GPU util / potencia / reloj SM | 22 % / 104 W / 2625 MHz | 22 % / 104 W / 2627 MHz |
| Memoria GPU máx | 671 MiB | 689 MiB |

El calibrador usa 0.11 % del período de frame y 3 % del tiempo del correlador. Los +18 MiB son la tabla.

Cuidado al interpretar:
- `cudaSync*`, `cudaInputData` y `cudaOutputData` miden esperas y copias PCIe, no cómputo; por eso se suman aparte los kernels.
- El benchmark repite el mismo frame, así que la tabla y los datos probablemente caben en el L2 (96 MB en la 5090). Con tráfico real podría subir algo (estimo hasta ~10 µs), sin cambiar la conclusión.

## 7. Problemas conocidos del entorno y del repo

1. **NVRTC más nuevo que el driver.** En esta máquina el driver es CUDA 13.2 y el toolkit 13.3. El correlador Astron, que compila en runtime, falla con *"PTX compiled with an unsupported toolchain"*, incluso sin el calibrador. Para las pruebas usé un `libnvrtc` 13.0 vía `LD_LIBRARY_PATH` (`CHARTS_TEST_LD_LIBRARY_PATH`). Falta una solución permanente: igualar driver y toolkit, o compilar a SASS en vez de PTX.
2. **`create_correlator_yaml` de `test_charts/sim/pipeline.py` está desactualizado.** Usa claves viejas de `rawFileRead` (`prefix`, `out_buf`) y los stages sueltos en vez de `cudaProcess`, por lo que `charts_sim.py correlate` y `pipeline` no corren con el kotekan actual. `sim/corr_cal.py` genera una config vigente que se puede reutilizar.
3. **Los frames del replay traen un prefijo de 4 bytes con metadata de tamaño 0**, y `charts_utils.core.io` espera ≥16 bytes. `load_corr_frame` en `sim/corr_cal.py` los lee con ese offset.
4. **`test_live_control_cli.py::test_01_query_status` falla** por conexión rechazada, porque necesita un kotekan vivo. No se verificó si ya fallaba antes de estos cambios.
5. **`test_charts/scratch_charts_sim/` aparece sin rastrear** en git (el `.gitignore` no lo cubre desde ese nivel).

## 8. Próximos pasos

Ordenados por prioridad.

1. **Validación física con datos reales.** La simulación no inyecta gains por antena, así que solo se validó aritmética y layout. Falta comprobar en datos reales (o inyectando gains en el generador del sim) que las fases de la fuente de calibración quedan en ≈0 tras calibrar en el nodo, y compararlo con la calibración offline.
2. **Confirmar con gains reales.** `gains_E_2026-10-19_with0.npy` tiene forma `(672, 4, 7)`: 4 antenas × 7 tiempos. El archivo no indica qué antenas son; las pruebas asumieron 0,4,6,8. Confirmar los índices, y revisar el rango de amplitudes (mín 0.0018, máx 18) para descartar saturación de `int32` en los autos.
3. **Prueba en el nodo con `64antennas_corr_cal.yaml`.** Ajustar la ruta de la tabla, correr con captura real y comprobar que `chartsAccumulate`, `rawFileWrite` y los scripts de charts-utils (`plot_triangular_vis.py`, `cal_pipeline_gpu.py`) funcionan con la salida calibrada.
4. **Medir rendimiento bajo carga real.** Repetir `bench-vis-cal` con el config de producción (2 shards, 15360 muestras por frame) y datos distintos entre frames, para descartar el efecto de L2.
5. **Recarga en caliente por REST.** Hoy se recalibra reiniciando kotekan. Si hace falta, agregar un endpoint con doble buffer en GPU y swap al inicio de un frame.
6. **Aplicar la misma idea al beamforming.** Según la tarea de Notion, en el tracking beam y el multi-beamforming las gains se multiplican en cada etapa, después de la cuantización interna. Es trabajo aparte.
7. **Revisar el promedio de gains.** Con ventanas largas, `nanmean` de complejos puede reducir la amplitud si la fase deriva. Definir un umbral de `phase_scatter_deg` para aceptar una calibración.
8. **Higiene del repo.**
   - Reparar o retirar `create_correlator_yaml` (7.2) y resolver el NVRTC (7.1).
   - Añadir `scratch_charts_sim/` al `.gitignore` (7.5).
   - Hacer commit en kotekan (rama `kotekan_charts`) y en charts-utils, que tiene cambios sin commit ajenos a esto (`pyproject.toml`, `export_directions.py`).
   - Actualizar la tarea *Calibración en nodo* en Notion con este estado.
