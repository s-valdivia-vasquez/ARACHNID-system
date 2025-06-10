#Code for continuous recording and generating a TXT file with all detections and matches.
#Designed for multiple cameras and Akida chips (Uses SNN for satellite detection)
#Requires Metavision SDK and Akida SDK

# Common error on ubuntu:    
#RuntimeError: 
#------------------------------------------------
#LibUSB
#
#Error fffffff5: LibUSB connection error: LIBUSB_ERROR_NO_MEM
#------------------------------------------------

# Solution:
# echo 1024 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
    
    
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage (Akida doesn't need it)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable OneDNN warnings
import sys
sys.path.append("/usr/lib/python3/dist-packages/")
from metavision_sdk_core import TimeSurfaceProducerAlgorithmMergePolarities, MostRecentTimestampBuffer # type: ignore
from metavision_sdk_cv import SparseOpticalFlowAlgorithm, ActivityNoiseFilterAlgorithm # type: ignore
from metavision_sdk_ui import EventLoop, BaseWindow, Window, UIKeyEvent,UIAction # type: ignore
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm # type: ignore
from metavision_core.event_io.raw_reader import initiate_device # type: ignore
from multiprocessing import Process, Manager,SimpleQueue
from time import sleep,perf_counter,strftime,localtime
from metavision_core.event_io import EventsIterator # type: ignore
from metavision_hal import DeviceDiscovery # type: ignore
from queue import Empty
from os import makedirs
import numpy as np 
import cv2 

CONFIGS = {
    "distance_gain": 0.05,
    "damping": 1.0,
    "omega_cutoff": 11.0,
    "min_cluster_size": 5,
    "max_link_time": 25000,
    "match_polarity": True,
    "use_simple_match": True,
    "full_square": True,
    "last_event_only": False,
    "size_threshold": 500,
    "v_thr": 4 ** 2,
    "filter_thr": 100000
}

def generate_patches(img, points):
    patches = np.zeros((len(points), 32, 32), dtype=np.uint8)
    h, w = img.shape
    
    for i, (x, y) in enumerate(points):
        left,right = x - 16, x + 16
        top,bottom = y - 16, y + 16
        x1,x2 = max(left, 0), min(right, w)
        y1,y2 = max(top, 0), min(bottom, h)
        patch_x = max(0, -left)
        patch_y = max(0, -top)
        patch_h = y2 - y1
        patch_w = x2 - x1
        if patch_h > 0 and patch_w > 0:
            patches[i, patch_y:patch_y+patch_h, patch_x:patch_x+patch_w] = img[y1:y2, x1:x2]
    return patches[..., np.newaxis]

def process_camera(serial, in_queue, cam_queue):

    print(f"Opening camera S{serial[-5:]}")
    device = initiate_device(serial)
    mv_iterator = EventsIterator.from_device(device=device,delta_t=100000)
    serial = serial[-5:]
    height, width = mv_iterator.get_size()
    
    biases = device.get_i_ll_biases()    
    biases.set("bias_diff", 299) #Default 299
    biases.set("bias_diff_on", 374) #Default 384
    biases.set("bias_diff_off", 234) #Default 222
    biases.set("bias_fo", 1500) #Default 1477
    biases.set("bias_hpf", 1450) #Default 1499
    biases.set("bias_pr", 1250) #Default 1250
    biases.set("bias_refr", 1500) #Default 1500
    
    # Set ERC (Event Rate Controller) to 10Mev/s
    if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device:
        erc_module = mv_iterator.reader.device.get_i_erc_module()
        if erc_module:
            erc_module.set_cd_event_rate(10000000)
            erc_module.enable(True)        
    
    event_frame_gen = PeriodicFrameGenerationAlgorithm(width, height, accumulation_time_us=50000,fps=15)

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
    
    tau = 50000
    time_surface = MostRecentTimestampBuffer(rows=height, cols=width, channels=1)
    ts_prod = TimeSurfaceProducerAlgorithmMergePolarities(width=width, height=height)
    last_processed_timestamp=0
    def cb_time_surface(timestamp, data):
        nonlocal last_processed_timestamp, time_surface
        last_processed_timestamp = timestamp
        time_surface = data
    
    ts_prod.set_output_callback(cb_time_surface)

    img_ts = np.empty((height, width), dtype=np.uint8)
    is_recording = False
    last_true_time = None
    total_patches = 0
    received_patches = 0
    msg_flag=True
    sates = 0
    coincidencias=0
    fondo = 0
    delays = []
    output_img = np.zeros((height, width, 3), np.uint8)
   
    def on_cd_frame_cb(ts, cd_frame):
        nonlocal output_img
        output_img=cd_frame
    
    event_frame_gen.set_output_callback(on_cd_frame_cb)  

    log_file=None
    t_recording=0

    with Window(title=f"Camera S{serial}", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:
        
        def keyboard_cb(key, scancode, action, mods):
            nonlocal is_recording, log_file, t_recording
            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                print(f"Closing camera {serial}")
                window.set_close_flag()
            elif key == UIKeyEvent.KEY_SPACE:
                if is_recording:
                    device.get_i_events_stream().stop_log_raw_data()
                    is_recording = False
                    print(f"Recording Stopped for camera {serial}")
                    log_file.write(f"{mv_iterator.get_current_time()-t_recording:011d}: Recording stopped at {strftime('%H.%M.%S', localtime())}\n")
                    log_file.close()
                    log_file=None
                else:
                    #dir="/media/optolab/WorkingHD/Recordings " + strftime("%d-%m-%y", localtime())
                    dir="/home/optolab/Recordings " + strftime("%d-%m-%y", localtime())
                    makedirs(dir, exist_ok=True)
                    base_log_path = f"{dir}/S{serial} {strftime('%d-%m-%y_%H.%M.%S', localtime())}"
                    log_path = base_log_path + ".raw"
                    log_txt = base_log_path + ".txt"
                    log_file = open(log_txt, "a")
                    device.get_i_events_stream().log_raw_data(log_path)
                    t_recording = mv_iterator.get_current_time()
                    print(f"Recording to {log_path} for camera {serial}")
                    is_recording = True
        
        window.set_keyboard_callback(keyboard_cb)
        base_timestamp=None
        RESET_THRESHOLD = 1_000_000_000
        warmup_iterations = 5 
        
        for evs in mv_iterator: 
         
            if base_timestamp is None:
                base_timestamp=evs['t'][0]
                
            evs['t']-=base_timestamp
            
            if len(evs['t']) > 0:
                if evs['t'][0] > RESET_THRESHOLD:
                        
                    print("Resetting time base and buffers")
                    base_timestamp += RESET_THRESHOLD 
                    last_processed_timestamp = 0
                    # Reinitialize the time surface and its producer
                    time_surface = MostRecentTimestampBuffer(rows=height, cols=width, channels=1)
                    ts_prod = TimeSurfaceProducerAlgorithmMergePolarities(width=width, height=height)
                    ts_prod.set_output_callback(cb_time_surface)                    
                    # Also reinitialize buffers used in flow and noise filtering
                    filter = ActivityNoiseFilterAlgorithm(width, height, CONFIGS["filter_thr"])
                    flow_buffer = SparseOpticalFlowAlgorithm.get_empty_output_buffer()
                    events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
                    event_frame_gen.reset()
                    continue
            
            EventLoop.poll_and_dispatch()
            filter.process_events(evs, events_buf)
            event_frame_gen.process_events(events_buf)                    
            ts_prod.process_events(events_buf)
            img_ts = (np.exp(-(last_processed_timestamp-time_surface.numpy()) / tau) * 255).astype(np.uint8)
            flow_algo.process_events(events_buf, flow_buffer)
            flow_np = flow_buffer.numpy()
            if warmup_iterations > 0:
                warmup_iterations -= 1
                window.show(output_img)
                continue

            if flow_np.size > 0:
                velocities = flow_np["vx"]**2 + flow_np["vy"]**2
                fast_indices = np.flatnonzero(velocities > CONFIGS["v_thr"])

                if fast_indices.size > 0:
                    u_points = np.unique(np.column_stack((flow_np["center_x"][fast_indices].astype(int), flow_np["center_y"][fast_indices].astype(int))) // 2, axis=0) * 2
                    patches = generate_patches(img_ts, u_points)
                    timestamp = perf_counter()
                    sent_patches = len(patches)
                    total_patches += sent_patches
                    in_queue.put(([serial] * sent_patches, patches, u_points, timestamp))

            if not(msg_flag) and (perf_counter() - last_true_time > 1):
                print(f"S{serial} Number of matches = {coincidencias}")
                if log_file!=None:
                    log_file.write(f" matches: {coincidencias}\n")
                    log_file.write(f"{mv_iterator.get_current_time()-t_recording:011d}: Satellite no longer detected at {strftime('%H.%M.%S', localtime())}\n\n")
                msg_flag=True
                coincidencias=0
                                
            # Procesar resultados obtenidos de la SNN
            while not cam_queue.empty():
                predictions, points, send_time = cam_queue.get()
                received_patches += len(predictions)
                delays.append(perf_counter() - send_time)
                
                true_count = sum(predictions)
                coincidencias+=true_count

                if true_count >= 2:
                    if msg_flag:
                        print(f"Camera S{serial} detected a satellite at {strftime('%H.%M.%S', localtime())}!")
                        if log_file is not None:
                            log_file.write(f"{mv_iterator.get_current_time()-t_recording-200_000:011d}: Satellite detected at {strftime('%H.%M.%S', localtime())}! ")
                        msg_flag = False
                    print(f"S{serial} Number of matches = {coincidencias}", end="\r")
                    last_true_time = perf_counter()
                elif true_count == 1:
                    if last_true_time is not None and (perf_counter() - last_true_time) < 0.05:
                        if msg_flag:
                            print(f"Camera S{serial} detected a satellite at {strftime('%H.%M.%S', localtime())}!")
                            if log_file is not None:
                                log_file.write(f"{mv_iterator.get_current_time()-t_recording-200_000:011d}: Satellite detected at {strftime('%H.%M.%S', localtime())}! ")
                            msg_flag = False
                    print(f"S{serial} Number of matches = {coincidencias}", end="\r")
                    last_true_time = perf_counter()
                for i,pred in enumerate(predictions): 
                    color = (0, 255, 0) if pred else (0, 0, 255)
                    cv2.rectangle(output_img, (points[i][0] - 10, points[i][1] - 10), (points[i][0] + 10, points[i][1] + 10), color, 2)
 
                sates += np.sum(predictions)
                fondo += np.sum(~np.array(predictions))

            window.show(output_img)

            if window.should_close():
                if is_recording:
                    device.get_i_events_stream().stop_log_raw_data()
                break
    
    if is_recording:    
        print(f"Stopping recording for camera {serial} before exit")
        log_file.write(f"\n{mv_iterator.get_current_time()-t_recording:011d}: Stopping recording for camera {serial} at {strftime('%H.%M.%S', localtime())}\n")
        device.get_i_events_stream().stop_log_raw_data()

    print(f"Camera {serial}: Packets sent = {total_patches}, received = {received_patches}, lost = {total_patches - received_patches}")
    if received_patches > 0:
        print(f"Camera {serial}: Average delay = {np.mean(delays):.4f} s, maximum delay = {np.max(delays):.4f} s")
        print(f"Camera {serial}:\tSatellite matches = {sates} \n\t\tBackground matches = {fondo}\n")
        if log_file is not None:
            log_file.write(f"Packets sent = {total_patches}, received = {received_patches}, lost or discarded = {total_patches - received_patches}\n")
            log_file.write(f"Average delay = {np.mean(delays):.4f} s, maximum delay = {np.max(delays):.4f} s\n")
            log_file.write(f"Satellite matches = {sates} \nBackground matches = {fondo}\n")

    if log_file is not None:
        log_file.close() 
               
def SNN_inference(in_queue, out_queues, serial_numbers,device_n):
    from akida_models.model_io import load_model  # type: ignore
    import akida  # type: ignore
        
    modelSNN = load_model('./CNN2SNN for sigma/model_akd.fbz')
    device = akida.devices()[device_n]
    modelSNN.map(device)
    print(f"Model successfully loaded on device {device}")
    
    # pre allocate memory for faster processing
    max_patches = 5000  
    batch_patches = np.empty((max_patches, 32, 32, 1), dtype=np.uint8)
    batch_points = np.empty((max_patches, 2), dtype=np.uint16)
    batch_serials = np.empty(max_patches, dtype=object)
    cam_batches = {serial[-5:]: ([], []) for serial in serial_numbers}
    count = 0
    t1 = perf_counter()

    while True:
        try:
            serial, patches, points, timestamp = in_queue.get(timeout=0.05)  # Espera no bloqueante
            if serial is None:
                break
            
            n_patches = len(patches)
            if n_patches + count < max_patches:
                batch_patches[count:count + n_patches] = patches
                batch_points[count:count + n_patches] = points
                batch_serials[count:count + n_patches] = serial
                count += n_patches
            else:
                available = max_patches - count -1
                batch_patches[count:count+available] = patches[:available]
                batch_points[count:count+available] = points[:available]
                batch_serials[count:count+available] = serial[:available]
                count += available
                print(f"Packets discarded due to exceeding maximum size: {n_patches - available}")
                
        except Empty:
            pass
        
        if perf_counter() - t1 >= 0.05 and count > 0:
            predictions = np.argmax(np.squeeze(modelSNN.predict(batch_patches[:count]), axis=(1, 2)), axis=1).astype('bool')

            for i in range(count):
                serial = batch_serials[i]
                cam_batches[serial][0].append(predictions[i])
                cam_batches[serial][1].append(batch_points[i])

            for serial, (batch_preds, batch_points) in cam_batches.items():
                if batch_preds:  
                    out_queues[serial].put((batch_preds, batch_points, timestamp))
                    cam_batches[serial] = ([], []) 

            count = 0 
            t1 = perf_counter()

    print(f"{device} finished")

def main():
    import akida # type: ignore


    def distribuir_camaras(n_cams: int, n_devs: int) -> list[list[int]]:
        if n_devs <= 0:
            return []
        base = n_cams // n_devs
        resto = n_cams % n_devs
        asignaciones = []
        inicio = 0
        for i in range(n_devs):
            fin = inicio + (base + 1 if i < resto else base)
            asignaciones.append(list(range(inicio, fin)))
            inicio = fin
        return asignaciones

    # Detect devices and cameras
    devices = akida.devices()
    print(f'Available devices: {[dev.desc for dev in devices]}')
    n_chips = len(devices)
    if n_chips == 0:
        raise RuntimeError("No Akida devices found.")
    
    serial_numbers = DeviceDiscovery.list()  # Cameras serial numbers
    n_cams = len(serial_numbers)
    if n_cams == 0:
        raise RuntimeError("No cameras found.")
    
    assignments = distribuir_camaras(n_cams, n_chips)
    
    with Manager() as manager:

        # Create an input queue for each chip
        in_queues = [manager.Queue() for _ in range(n_chips)]
        
        # Map each camera to its corresponding queue
        camera_mapping = {}
        for chip_idx, cameras in enumerate(assignments):
            for cam_idx in cameras:
                camera_mapping[cam_idx] = in_queues[chip_idx]

        # Create output queues (one per camera)
        out_queues = {serial[-5:]: SimpleQueue() for serial in serial_numbers}

        # Start inference processes (one per chip)
        infer_processes = []
        for chip_idx in range(n_chips):
            # Get the cameras assigned to this chip
            assigned_cameras = assignments[chip_idx]
            if not assigned_cameras:  # Skip chips with no cameras
                continue
            # Filter serials and out_queues for this chip
            chip_cam_serials = [serial_numbers[i] for i in assigned_cameras]
            chip_out_queues = {serial[-5:]: out_queues[serial[-5:]] for serial in chip_cam_serials}
            
            print(f'Chip {chip_idx} will handle cameras: {chip_cam_serials}')
            
            p = Process(target=SNN_inference, 
                    args=(
                        in_queues[chip_idx],    # Chip-specific input queue
                        chip_out_queues,        # Only output queues of its cameras
                        chip_cam_serials,       # Only serials of its cameras
                        chip_idx
                    ))
            p.start()
            infer_processes.append(p)

        sleep(3)  # Wait for initialization

        # Start camera processes
        cam_processes = []
        for idx, serial in enumerate(serial_numbers):
            chip_queue = camera_mapping[idx]
            p = Process(target=process_camera,
                       args=(serial, chip_queue, out_queues[serial[-5:]]))
            p.start()
            cam_processes.append(p)
            sleep(1)

        # Wait for cameras to finish
        for p in cam_processes:
            p.join()

        # Send termination signal to all chips
        for q in in_queues:
            q.put((None, None, None, None))

        # Wait for inference processes to finish
        for p in infer_processes:
            p.join()

if __name__ == "__main__":
    main()




