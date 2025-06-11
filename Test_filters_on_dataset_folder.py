#Este código es un script de Python que procesa archivos .raw de eventos utilizando diferentes algoritmos de filtrado.
# El script permite al usuario seleccionar un directorio de entrada, un factor de reproducción y un tipo de filtro a aplicar.
# Se generan imágenes de salida a partir de los eventos procesados y se guardan en un directorio específico para comparar la influencia de los filtros en la calidad de los eventos.


import os
import glob
from enum import Enum
from metavision_core.event_io import EventsIterator
from metavision_core.event_io import LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm
from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm, SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import EventLoop
import numpy as np
from matplotlib import pyplot as plt  # graphic library, for plots


# Activity Noise Filter
activity_time_ths = 20000  # Length of the time window for activity filtering (in us)

# STC
stc_filter_ths = 10000  # Length of the time window for filtering (in us)
stc_cut_trail = False  # If true, after an event goes through, it removes all events until change of polarity

# Trail Filter
trail_filter_ths = 1000000  # Length of the time window for activity filtering (in us)

class Filter(Enum):
    NONE = 0,
    ACTIVITY = 1,
    STC = 2,
    TRAIL = 3

def parse_args():
    import argparse
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision Noise Filtering sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-d', '--directory', dest='input_directory', default="",
        help="Path to input directory containing RAW files.")
    parser.add_argument(
        '-r', '--replay_factor', type=float, default=1,
        help="Replay Factor. If greater than 1.0 we replay with slow-motion, otherwise this is a speed-up over real-time.")
    parser.add_argument(
        '-f', '--filter', type=int, default=0,
        help="Algoritmo de filtrado: \n0 = None\n1 = Activity Noise Filter Algorithm'\n2 = Spatio Temporal Contrast Algorithm\n3 = Trail Filter.")
    parser.add_argument(
        '-a', '--accumulation_time', type=int, default=33333,
        help="Accumulation time in us")
    args = parser.parse_args()
    return args


def process_file(raw_file, args, filters, filter_type, height, width):
    """Process an individual .raw file."""
    mv_iterator = EventsIterator(input_path=raw_file, delta_t=10000)

    if args.replay_factor > 0 and not is_live_camera(raw_file):
        mv_iterator = LiveReplayEventsIterator(mv_iterator, replay_factor=args.replay_factor)

    events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    # Event Frame Generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=args.accumulation_time, fps=25 * args.replay_factor)

    # Variable para almacenar el último frame procesado
    last_cd_frame = None

    # Definir callback para actualizar el último frame, pero no guardar aún
    def on_cd_frame_cb(ts, cd_frame):
        nonlocal last_cd_frame
        last_cd_frame = cd_frame

    # Configurar el callback para el generador de frames
    event_frame_gen.set_output_callback(on_cd_frame_cb)

    # Process events
    for evs in mv_iterator:

        # Process events
        if filter_type in filters:
            if args.filter == 4:
                filters[Filter.ACTIVITY].process_events(evs, events_buf)
                filters[Filter.STC].process_events_(events_buf)
            else:
                filters[filter_type].process_events(evs, events_buf)
        
            event_frame_gen.process_events(events_buf)
            
        else:
            event_frame_gen.process_events(evs)

    # Guardar solo el último frame procesado

    output_image_name = f".\\Frames\\{os.path.basename(raw_file)}.png"
    plt.imsave(output_image_name, last_cd_frame)
    print(f"Último frame guardado para {raw_file} en {output_image_name}.")


def main():
    """ Main """
    args = parse_args()

    # Buscar todos los archivos .raw en el directorio especificado
    raw_files = glob.glob(os.path.join(args.input_directory, "*.raw"))

    if not raw_files:
        print("No se encontraron archivos .raw en el directorio especificado.")
        return

    # Supongamos que todos los archivos tienen la misma resolución (altura y anchura)
    first_file = raw_files[0]
    mv_iterator = EventsIterator(input_path=first_file, delta_t=1000)
    height, width = mv_iterator.get_size()  # Camera Geometry

    filters = {
        Filter.ACTIVITY: ActivityNoiseFilterAlgorithm(width, height, activity_time_ths),
        Filter.TRAIL: TrailFilterAlgorithm(width, height, trail_filter_ths),
        Filter.STC: SpatioTemporalContrastAlgorithm(width, height, stc_filter_ths, stc_cut_trail)
    }

    if args.filter == 0:
        filter_type = Filter.NONE
    elif args.filter == 1:
        filter_type = Filter.ACTIVITY
    elif args.filter == 2:
        filter_type = Filter.STC
    elif args.filter == 3:
        filter_type = Filter.TRAIL
    elif args.filter == 4:
        filter_type = Filter.ACTIVITY

    # Procesar cada archivo .raw en la carpeta
    for raw_file in raw_files:
        print(f"Procesando archivo: {raw_file}")
        process_file(raw_file, args, filters, filter_type, height, width)

    print("Procesamiento completo.")

if __name__ == "__main__":
    main()
