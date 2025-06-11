import os
import cv2
import numpy as np
import pandas as pd
import itertools
from metavision_core.event_io import EventsIterator
from metavision_sdk_core import TimeSurfaceProducerAlgorithmMergePolarities, MostRecentTimestampBuffer
from metavision_sdk_cv import SparseOpticalFlowAlgorithm, ActivityNoiseFilterAlgorithm
from metavision_sdk_ui import EventLoop

def frange(start, stop, step):
    while start <= stop:
        yield start
        start += step

def generate_test_configs():
    #omega_cutoff_values = [round(x, 2) for x in list(frange(5, 12, 1))]
    #damping_values = [round(x, 2) for x in list(frange(0.5, 1.2, 0.1))]
    #combinations = list(itertools.product(omega_cutoff_values, damping_values))
    #return pd.DataFrame({
    #    'Test N°': [f'Test {i+1}' for i in range(len(combinations))],
    #    'distance_gain': [0.05] * len(combinations),
    #    'damping': [comb[1] for comb in combinations],
    #    'omega_cutoff': [comb[0] for comb in combinations],
    #    'min_cluster_size': [5] * len(combinations),
    #    'max_link_time': [25000] * len(combinations),
    #    'match_polarity': [True] * len(combinations),
    #    'use_simple_match': [True] * len(combinations),
    #    'full_square': [True] * len(combinations),
    #    'last_event_only': [False] * len(combinations),
    #    'size_threshold': [1000] * len(combinations),
    #    'v_thr': [4] * len(combinations),
    #    'filter_thr': [50000] * len(combinations),
    #})
    
    
    combinations=150
    return pd.DataFrame({
        'Test N°': [f'Test {i+1}' for i in range(combinations)],
        'distance_gain': [(i+1)*0.01 for i in range (combinations)],
        'damping': [0.8] * combinations,
        'omega_cutoff': [11.0] * combinations,
        'min_cluster_size': [5] * combinations,
        'max_link_time': [25000] * combinations,
        'match_polarity': [True] * combinations,
        'use_simple_match': [True] * combinations,
        'full_square': [True] * combinations,
        'last_event_only': [False] * combinations,
        'size_threshold': [1000] * combinations,
        'v_thr': [4] * combinations,
        #'filter_thr': [(i+1)*1000 for i in range (combinations)],
        'filter_thr': [50000] * combinations
        
    })

def save_patch(img, points, output_dir, width=None, ts=None):

    os.makedirs(output_dir, exist_ok=True)

    patch = np.zeros((32, 32), dtype=np.uint8)
    for (x, y) in points:
        x_start, x_end = max(0, x - 16), min(img.shape[1], x + 16)
        y_start, y_end = max(0, y - 16), min(img.shape[0], y + 16)
        patch_x_start = max(0, 16 - (y - y_start))
        patch_y_start = max(0, 16 - (x - x_start))
        patch[patch_x_start:(patch_x_start + (y_end - y_start)), patch_y_start:(patch_y_start + (x_end - x_start))] = img[y_start:y_end, x_start:x_end]

        name = y * width + x
    
        filename = f"{output_dir}/{name:06d}_{ts:07d}_({x},{y}).png"
        suffix = 1
        while os.path.exists(filename):
            filename = f"{output_dir}/{name:06d}_{ts:07d}_({x},{y})_{suffix}.png"
            suffix += 1

        # Guardar el parche
        cv2.imwrite(filename, patch)

def process_file(event_file_path, output_base_dir, excel_writer):
    dt_step = 100000
    accumulation = 500000
    test_configs = generate_test_configs()
    test_configs["Detections"] = 0
    file_name = os.path.basename(event_file_path).rsplit('.', 1)[0]
    output_dir = os.path.join(output_base_dir, file_name)
    os.makedirs(output_dir, exist_ok=True)

    for index, row in test_configs.iterrows():
        test_subdir = os.path.join(output_dir, f"Test_{index+1}")
        os.makedirs(test_subdir, exist_ok=True)
        
        mv_iterator = EventsIterator(input_path=event_file_path, delta_t=dt_step)
        height, width = mv_iterator.get_size()
        last_processed_timestamp = 0

        time_surface = MostRecentTimestampBuffer(rows=height, cols=width, channels=1)
    
        def cb_time_surface(timestamp, data):
            nonlocal last_processed_timestamp, time_surface
            last_processed_timestamp = timestamp
            time_surface = data   
        
        filter = ActivityNoiseFilterAlgorithm(width, height, row['filter_thr'])
        events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()                         
        
        flow_algo = SparseOpticalFlowAlgorithm(
            width=width,
            height=height,
            distance_gain=row['distance_gain'],
            damping=row['damping'],
            omega_cutoff=row['omega_cutoff'],
            min_cluster_size=row['min_cluster_size'],
            max_link_time=row['max_link_time'],
            match_polarity=row['match_polarity'],
            use_simple_match=row['use_simple_match'],
            full_square=row['full_square'],
            last_event_only=row['last_event_only'],
            size_threshold=row['size_threshold']
        )
        
        flow_buffer = SparseOpticalFlowAlgorithm.get_empty_output_buffer()
        ts_prod = TimeSurfaceProducerAlgorithmMergePolarities(width=width, height=height)
        ts_prod.set_output_callback(cb_time_surface)
        img = np.empty((height, width), dtype=np.uint8)
        n_detec = 0
        
        for evs in mv_iterator:
            evs['t'] += accumulation
            filter.process_events(evs, events_buf)
            
            ts_prod.process_events(events_buf)
            time_surface.generate_img_time_surface(last_processed_timestamp, accumulation, img)
            flow_algo.process_events(events_buf, flow_buffer)
            
            flow_np = flow_buffer.numpy()

            if flow_np.size > 0:
                velocities = flow_np["vx"]**2 + flow_np["vy"]**2
                fast_indices = np.flatnonzero(velocities > row['v_thr'] ** 2)
                if fast_indices.size > 0:
                    u_points = np.unique(np.column_stack((flow_np["center_x"].astype(int), flow_np["center_y"].astype(int))), axis=0)
                    n_detec += len(u_points)
                    save_patch(img, u_points, output_dir=test_subdir, width=width, ts=evs['t'][0]-accumulation)
        
        test_configs.at[index, "Detections"] = n_detec
    test_configs.to_excel(excel_writer, sheet_name=file_name, index=False)

def main():
    input_dir = "Sates lente sigma"
    output_dir = "C:/Users/optolab/Documents/python/Patches"
    excel_output = "sigma_tests_distance_gain.xlsx"
    with pd.ExcelWriter(excel_output, engine='xlsxwriter') as writer:
        for file in os.listdir(input_dir):
            if file.endswith(".raw"):
                process_file(os.path.join(input_dir, file), output_dir, writer)

if __name__ == "__main__":
    main()
