from metavision_core.event_io import EventsIterator, is_live_camera, LiveReplayEventsIterator
from metavision_sdk_core import TimeSurfaceProducerAlgorithmMergePolarities, MostRecentTimestampBuffer
from metavision_sdk_cv import SparseOpticalFlowAlgorithm, ActivityNoiseFilterAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIKeyEvent
import numpy as np
import cv2
from scipy.spatial.distance import cdist
import os
import glob
from time import perf_counter

CONFIGS = {
    "distance_gain": 0.05,
    "damping": 0.8,
    "omega_cutoff": 12.0,
    "min_cluster_size": 5,
    "max_link_time": 25000,
    "match_polarity": True,
    "use_simple_match": True,
    "full_square": True,
    "last_event_only": False,
    "size_threshold": 500,
    "v_thr": 4 ** 2,
    "filter_thr": 10000
}

def save_patches(img, points,timestamp,dir_name,output_dir):
    for i, (x, y) in enumerate(points):
        patch = np.zeros((32, 32), dtype=np.uint8)
        x_start, x_end = max(0, x - 16), min(img.shape[1], x + 16)
        y_start, y_end = max(0, y - 16), min(img.shape[0], y + 16)
        patch_x_start = max(0, 16 - (y - y_start))
        patch_y_start = max(0, 16 - (x - x_start))
        patch[patch_x_start:(patch_x_start + (y_end - y_start)), patch_y_start:(patch_y_start + (x_end - x_start))] = img[y_start:y_end, x_start:x_end]
        filename = os.path.join(output_dir, f"{timestamp:010d}_({x}_{y})_{dir_name}.png")
        cv2.imwrite(filename,patch)
        

def process_file(input_file):
    base_output_dir = "C:/Users/optolab/Documents/python/Dataset/Paper figures"
     
    dir_name = os.path.splitext(os.path.basename(input_file))[0]
    output_dir = os.path.join(base_output_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    dt_step = 10000
    mv_iterator = EventsIterator(input_path=input_file, delta_t=dt_step)
    height, width = mv_iterator.get_size()
    tau = 500000

    # Inicialización de algoritmos y filtros
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

    flow_buffer = SparseOpticalFlowAlgorithm.get_empty_output_buffer()

    filter = ActivityNoiseFilterAlgorithm(width, height, CONFIGS["filter_thr"])
    events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
    
    time_surface = MostRecentTimestampBuffer(rows=height, cols=width, channels=1)
    ts_prod = TimeSurfaceProducerAlgorithmMergePolarities(width=width, height=height)
    last_processed_timestamp=0
    def cb_time_surface(timestamp, data):
        nonlocal last_processed_timestamp, time_surface
        last_processed_timestamp = timestamp
        time_surface = data

    ts_prod.set_output_callback(cb_time_surface)
    img_ts = np.empty((height, width), dtype=np.uint8)

    with Window(title=input_file, width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        def keyboard_cb(key, scancode, action, mods):
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                window.set_close_flag()

        window.set_keyboard_callback(keyboard_cb)


        for evs in mv_iterator: 
            
            EventLoop.poll_and_dispatch()
            evs['t']+=6*tau
                        
            filter.process_events(evs, events_buf)
            flow_algo.process_events(events_buf, flow_buffer)
            #flow_algo.process_events(evs, flow_buffer)
            #ts_prod.process_events(evs)
            ts_prod.process_events(events_buf)
            exp_ts = np.exp(-(last_processed_timestamp-time_surface.numpy()) / tau)
            img_ts = (exp_ts * 255).astype(np.uint8)
            c_img_ts=cv2.applyColorMap(img_ts, cv2.COLORMAP_JET)
            
            show=(last_processed_timestamp-6*tau)/1000000
            print(f"Record is at {int(show)}s",end="\r")
            flow_np = flow_buffer.numpy()

            if flow_np.size > 0:
                velocities = flow_np["vx"]**2 + flow_np["vy"]**2
                fast_indices = np.flatnonzero(velocities > CONFIGS["v_thr"])

                if fast_indices.size > 0:
                    unique_points = np.unique(np.column_stack((flow_np["center_x"][fast_indices].astype(int), flow_np["center_y"][fast_indices].astype(int))) // 2, axis=0) * 2
                    
                    #if unique_points.size > 0:
                    #    mask = (
                    #        (unique_points[:, 0] >= 8) & 
                    #        (unique_points[:, 0] <= width - 8) & 
                    #        (unique_points[:, 1] >= 8) & 
                    #        (unique_points[:, 1] <= height - 8)
                    #    )
                    #    unique_points = unique_points[mask]
                    
                    #if unique_points.size > 1:
                    #    dist_matrix = cdist(unique_points, unique_points)
                    #    np.fill_diagonal(dist_matrix, np.inf)
                    #    keep_indices = []
                    #    for i, dists in enumerate(dist_matrix):
                    #        if all(d > 20 for d in dists[keep_indices]):
                    #            keep_indices.append(i)
                    #    unique_points = unique_points[keep_indices]
                        
                    if unique_points.size > 0:
                        
                        timestamp=last_processed_timestamp-6*tau
                        save_patches(img_ts, unique_points,timestamp,dir_name,output_dir)

                        for (x, y) in unique_points:
                            color = (255, 255, 255)
                            cv2.rectangle(c_img_ts, (x - 16, y - 16), (x + 16, y + 16), color, 2)
                            text = f"x:{x} y:{y}"
                            cv2.putText(c_img_ts, text, (x - 16, y + 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
                    
            window.show(c_img_ts)

            if window.should_close():
                break

def main():
    input_dir = "C:/Users/optolab/Documents/python/Dataset/Paper figures"
    raw_files = glob.glob(os.path.join(input_dir, "*.raw"))
   
    for raw_file in raw_files:
        print(f"Procesando: {raw_file}")
        process_file(raw_file)      

if __name__ == "__main__":
    main()