import os
import cv2
import numpy as np
import argparse
import pandas as pd
from scipy.spatial import cKDTree
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import TimeSurfaceProducerAlgorithmMergePolarities, MostRecentTimestampBuffer
from metavision_sdk_cv import SparseOpticalFlowAlgorithm, ActivityNoiseFilterAlgorithm
from metavision_sdk_ui import EventLoop

# Configuraciones probadas
CONFIGS = {
    "distance_gain": 0.05,
    "damping": 0.9,
    "omega_cutoff": 7.0,
    "min_cluster_size": 5,
    "max_link_time": 25000,
    "match_polarity": True,
    "use_simple_match": True,
    "full_square": True,
    "last_event_only": False,
    "size_threshold": 10,
    "v_thr": 4 ** 2,
    "filter_thr": 50000
}

def parse_args():
    parser = argparse.ArgumentParser(description="Procesar múltiples archivos RAW.")
    parser.add_argument('-i', '--input-folder', required=True, help="Carpeta de entrada con archivos .raw.")
    parser.add_argument('-o', '--output-folder', required=True, help="Carpeta de salida para los resultados.")
    parser.add_argument('--dt-step', type=int, default=33333, help="Paso temporal en microsegundos.")
    return parser.parse_args()

def process_file(file_path, output_folder, dt_step):
    last_processed_timestamp = 0
    mv_iterator = EventsIterator(input_path=file_path, delta_t=dt_step)
    #if not is_live_camera(file_path):
    #    mv_iterator = LiveReplayEventsIterator(mv_iterator)
    height, width = mv_iterator.get_size()

    accumulation = 500000
    n_detec = 0
    
    flow_algo = SparseOpticalFlowAlgorithm(
        width=width,
        height=height,
        distance_gain=CONFIGS['distance_gain'],
        damping=CONFIGS['damping'],
        omega_cutoff=CONFIGS['omega_cutoff'],
        min_cluster_size=CONFIGS['min_cluster_size'],
        max_link_time=CONFIGS['max_link_time'],
        match_polarity=CONFIGS['match_polarity'],
        use_simple_match=CONFIGS['use_simple_match'],
        full_square=CONFIGS['full_square'],
        last_event_only=CONFIGS['last_event_only'],
        size_threshold=CONFIGS['size_threshold']
    )    

    filter = ActivityNoiseFilterAlgorithm(width, height, CONFIGS["filter_thr"])
    events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()

    time_surface = MostRecentTimestampBuffer(rows=height, cols=width, channels=1)
    ts_prod = TimeSurfaceProducerAlgorithmMergePolarities(width=width, height=height)

    def cb_time_surface(timestamp, data):
        nonlocal last_processed_timestamp
        nonlocal time_surface
        last_processed_timestamp = timestamp
        time_surface = data

    ts_prod.set_output_callback(cb_time_surface)

    img = np.empty((height, width), dtype=np.uint8)
    flow_buffer = SparseOpticalFlowAlgorithm.get_empty_output_buffer()

    file_name = os.path.basename(file_path)[0:-4]
    file_output_folder = os.path.join(output_folder, file_name)
    os.makedirs(file_output_folder, exist_ok=True)

    for evs in mv_iterator:
        evs['t'] += accumulation
        EventLoop.poll_and_dispatch()
        filter.process_events(evs, events_buf)

        ts_prod.process_events(events_buf)
        time_surface.generate_img_time_surface(last_processed_timestamp, accumulation, img)

        flow_algo.process_events(events_buf, flow_buffer)
        flow_np = flow_buffer.numpy()

        if flow_np.size > 0:
            velocities = flow_np["vx"] ** 2 + flow_np["vy"] ** 2
            fast_indices = np.flatnonzero(velocities > CONFIGS["v_thr"])
            if fast_indices.size > 0:
                high_speed_points = np.column_stack((flow_np["center_x"][fast_indices].astype(int), flow_np["center_y"][fast_indices].astype(int)))
                points = np.unique(high_speed_points // 2, axis=0) * 2

                tree = cKDTree(points)
                idx_unique = tree.query_ball_tree(tree, r=20)
                unique_points = []
                for indices in idx_unique:
                    if len(indices) > 0:
                        primary_point = points[indices[0]]
                        if not any(np.linalg.norm(primary_point - np.array(up)) < 20 for up in unique_points):
                            unique_points.append(primary_point)

                unique_points = np.array(unique_points)
                n_detec += len(unique_points)
                for (x, y) in unique_points:
                    save_patch(img, x, y, file_output_folder, width,ts=evs['t'][-1]-accumulation)

    return n_detec

def save_patch(img, x, y, output_dir, width, ts):

    # Asegurar que la carpeta de salida exista
    os.makedirs(output_dir, exist_ok=True)

    # Generar un identificador único para el parche
    pos = y * width + x
    name =f"{pos:06d}_{ts}"

    x_start, x_end = x - 16, x + 16
    y_start, y_end = y - 16, y + 16
    patch = np.zeros((32, 32), dtype=img.dtype)

    x_valid_start = max(0, x_start)
    x_valid_end = min(img.shape[1], x_end)
    y_valid_start = max(0, y_start)
    y_valid_end = min(img.shape[0], y_end)

    patch_x_start = x_valid_start - x_start
    patch_x_end = patch_x_start + (x_valid_end - x_valid_start)
    patch_y_start = y_valid_start - y_start
    patch_y_end = patch_y_start + (y_valid_end - y_valid_start)

    patch[patch_y_start:patch_y_end, patch_x_start:patch_x_end] = img[y_valid_start:y_valid_end, x_valid_start:x_valid_end]

    filename = f"{output_dir}/{name}_({x},{y}).png"
    suffix = 1
    while os.path.exists(filename):
        filename = f"{output_dir}/{name}_({x},{y})_{suffix}.png"
        suffix += 1

    # Guardar el parche
    cv2.imwrite(filename, patch)

def main():
    args = parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder
    dt_step = args.dt_step

    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith('.raw')]

    results = []

    for file_name in files:
        file_path = os.path.join(input_folder, file_name)
        print(f"Processing {file_name}...")
        n_detec = process_file(file_path, output_folder, dt_step)
        results.append({"File": file_name, **CONFIGS, "Detections": n_detec})

    df = pd.DataFrame(results)
    df.to_excel(os.path.join(output_folder, "results.xlsx"), index=False)
    print("Processing completed. Results saved.")

if __name__ == "__main__":
    main()