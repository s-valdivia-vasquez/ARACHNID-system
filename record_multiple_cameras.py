from metavision_sdk_cv import ActivityNoiseFilterAlgorithm, TrailFilterAlgorithm, SpatioTemporalContrastAlgorithm
from metavision_sdk_ui import EventLoop, BaseWindow, MTWindow, UIAction, UIKeyEvent, Window
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_core.event_io.raw_reader import initiate_device
from metavision_core.event_io import EventsIterator
from metavision_hal import DeviceDiscovery
from multiprocessing import Process
from threading import Thread
from queue import Queue
from enum import Enum
import tkinter as tk
import argparse
import time
import os

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Metavision SDK Get Started sample.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-f', '--fps', dest='fps', type=int, default=25,
        help="Frame rate")
    parser.add_argument(
        '-a', '--accumulation_time', type=int, default=33333,
        help="Accumulation time in us")    
    parser.add_argument(
        '-o', '--output_dir', dest='output_dir', default="",
        help="Output directory")
    args = parser.parse_args()
    return args

class Filter(Enum):
    NONE = 0,
    ACTIVITY = 1,
    STC = 2,
    TRAIL = 3
    
# Función para actualizar los biases utilizando una cola para la comunicación entre hilos
def update_bias_worker(biases, queue):
    while True:
        bias_name, value = queue.get()
        if bias_name is None:  # Señal de terminación
            break
        biases.set(bias_name, int(value))
        
class BiasControllerApp:
    def __init__(self, master, biases, queue):
        self.master = master
        self.biases = biases
        self.queue = queue

        # Crear sliders para cada bias
        self.create_slider("bias_diff_on", 374, 600)
        self.create_slider("bias_diff_off", 100, 335)
        self.create_slider("bias_fo", 1250, 1800)
        self.create_slider("bias_hpf", 900, 1800)
        self.create_slider("bias_refr", 1300, 1800)

    def create_slider(self, bias_name, min_val, max_val):
        # Obtener valor actual del bias
        current_val = self.biases.get(bias_name)
        
        # Crear un frame para cada slider
        frame = tk.Frame(self.master)
        frame.pack()

        # Etiqueta para el bias
        label = tk.Label(frame, text=bias_name)
        label.pack(side=tk.LEFT)

        # Slider para modificar el valor del bias
        slider = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL, length=300,
                          command=lambda val, b=bias_name: self.queue.put((b, val)))
        slider.set(current_val)
        slider.pack(side=tk.LEFT)

def run_bias_interface(biases, queue, serial):
    """Función que ejecuta la interfaz de Tkinter en un hilo separado."""
    root = tk.Tk()
    root.title(f"S{serial[-5:]} Biases")
    BiasControllerApp(root, biases, queue)

    # Modificar root para permitir el cierre
    def on_closing():
        queue.put((None, None))  # Enviar señal de terminación
        root.quit()  # Finalizar el loop de Tkinter
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

def abrir_ventana(serial, num):
    print(f"Abriendo cámara {num} (Serial No. {serial[-5:]})")
    device = initiate_device(serial)
    is_recording = False
    args = parse_args()
    mv_iterator = EventsIterator.from_device(device=device)
    frame_rate = args.fps
    height, width = mv_iterator.get_size()  # Camera Geometry
    
    # Set ERC (Event Rate Controller) to 10Mev/s
    if hasattr(mv_iterator.reader, "device") and mv_iterator.reader.device:
        erc_module = mv_iterator.reader.device.get_i_erc_module()
        if erc_module:
            erc_module.set_cd_event_rate(10000000)
            erc_module.enable(True)    

    activity_time_ths = 10000  # Length of the time window for activity filtering (in us)

    trail_filter_ths = 100000  # Length of the time window for activity filtering (in us)

    stc_filter_ths = 10000  # Length of the time window for filtering (in us)
    stc_cut_trail = True  # If true, after an event goes through, it removes all events until change of polarity
        
    filters = {Filter.ACTIVITY: ActivityNoiseFilterAlgorithm(width, height, activity_time_ths),
               Filter.TRAIL: TrailFilterAlgorithm(width, height, trail_filter_ths),
               Filter.STC: SpatioTemporalContrastAlgorithm(width, height, stc_filter_ths, stc_cut_trail)
               }

    events_buf = ActivityNoiseFilterAlgorithm.get_empty_output_buffer()
    filter_type = Filter.NONE

    # Event Frame Generator
    event_frame_gen = PeriodicFrameGenerationAlgorithm(sensor_width=width, sensor_height=height, fps=frame_rate,accumulation_time_us=args.accumulation_time,
                                                           palette=ColorPalette.Dark)
         
    with Window(title=f"Cámara {num} (Serial No. {serial[-5:]})", width=width, height=height, mode=BaseWindow.RenderMode.BGR) as window:        
        
        # Variables para controlar el estado de los hilos
        bias_thread = None
        update_thread = None
        queue = None                

        def keyboard_cb(key, scancode, action, mods):
            nonlocal filter_type, is_recording, bias_thread, update_thread, queue

            if action != UIAction.RELEASE:
                return
            if key == UIKeyEvent.KEY_ESCAPE or key == UIKeyEvent.KEY_Q:
                print(f"Cerrando cámara {num} (Serial No. {serial[-5:]})")
                window.set_close_flag()
            elif key == UIKeyEvent.KEY_SPACE:
                if is_recording:
                    device.get_i_events_stream().stop_log_raw_data()
                    is_recording = False
                    print(f"Grabación finalizada para cámara {num} (Serial No. {serial[-5:]})")
                else:  # Start Recording
                    log_path = "S" + serial[-5:] + " " + time.strftime("%d-%m-%y_%H.%M.%S", time.localtime()) + ".raw"
                    if args.output_dir != "":
                        log_path = os.path.join(args.output_dir, log_path)
                    print(f"Recording to {log_path} for camera {num} (Serial No. {serial[-5:]})")
                    device.get_i_events_stream().log_raw_data(log_path)
                    is_recording = True                    
            elif key == UIKeyEvent.KEY_E:
                print(f"Mostrando todos los eventos de cámara {num} (Serial No. {serial[-5:]})")
                filter_type = Filter.NONE
            elif key == UIKeyEvent.KEY_A:
                print(f"Activity Noise Filter en cámara {num} (Serial No. {serial[-5:]})")
                filter_type = Filter.ACTIVITY
            elif key == UIKeyEvent.KEY_T:
                print(f"Trail Filter en cámara {num} (Serial No. {serial[-5:]})")
                filter_type = Filter.TRAIL
            elif key == UIKeyEvent.KEY_S:
                print(f"Spatio Temporal Contrast Filter en cámara {num} (Serial No. {serial[-5:]})")                
                filter_type = Filter.STC 
            elif key == UIKeyEvent.KEY_R:
                print(f"Activity Noise Filter y STC en cámara {num} (Serial No. {serial[-5:]})")                
                filter_type = True 
            elif key == UIKeyEvent.KEY_B:
                if bias_thread is not None and bias_thread.is_alive():
                    return

                biases = device.get_i_ll_biases()
                queue = Queue()
                update_thread = Thread(target=update_bias_worker, args=(biases, queue))
                update_thread.start()
                bias_thread = Thread(target=run_bias_interface, args=(biases, queue,serial))
                bias_thread.start()

            elif key == UIKeyEvent.KEY_D:
                biases = device.get_i_ll_biases()
                print("Cambiando biases a su valor default ...")
                biases.set("bias_diff_on", 384)
                biases.set("bias_diff_off", 222)
                biases.set("bias_fo", 1477)
                biases.set("bias_hpf", 1499)
                biases.set("bias_refr", 1500)
                print("bias_diff_on = 384 \n"
                      "bias_diff_off = 222 \n"
                      "bias_fo = 1477 \n"
                      "bias_hpf = 1499 \n"
                      "bias_refr = 1500")

        window.set_keyboard_callback(keyboard_cb)

        def on_cd_frame_cb(ts, cd_frame):
            #window.show_async(cd_frame) #Para MT Window
            window.show(cd_frame) #Para Window

        event_frame_gen.set_output_callback(on_cd_frame_cb)

        # Process events
        for evs in mv_iterator:
            # Dispatch system events to the window
            EventLoop.poll_and_dispatch()

            # Process events
            if filter_type in filters:
                filters[filter_type].process_events(evs, events_buf)                
                event_frame_gen.process_events(events_buf)
            elif filter_type==True:
                filtro = Filter.ACTIVITY                
                filters[filtro].process_events(evs, events_buf)                
                filtro = Filter.STC             
                filters[filtro].process_events_(events_buf)
                event_frame_gen.process_events(events_buf)                
            else:
                event_frame_gen.process_events(evs)

            if window.should_close():
                if is_recording:
                    device.get_i_events_stream().stop_log_raw_data()
                break
            
    # Finalizar hilos de trabajo al cerrar la ventana
    if queue is not None:
        queue.put((None, None))  # Señal de terminación para el hilo de actualización
    if update_thread is not None:
        update_thread.join()
    if bias_thread is not None:
        bias_thread.join()            
    return

def main():

    serial_numbers = DeviceDiscovery.list()
    
    if serial_numbers:
        print(f"{len(serial_numbers)} Dispositivos conectados:")
        for serial in serial_numbers:
            print(serial)
    else:
        raise RuntimeError("No se encontraron dispositivos conectados.")
    
    processes=[]
    for num, serial in enumerate(serial_numbers):
        time.sleep(1) 
        p = Process(target=abrir_ventana, args=(serial, num), name=f"C_{num}_{serial[-8:]}")
        processes.append(p)
        p.start()
        
        if num == len(serial_numbers) - 1:
            print("Available keyboard options:\n"
                "  - A: Filter events using the activity noise filter algorithm\n"
                "  - T: Filter events using the trail filter algorithm\n"
                "  - S: Filter events using the spatio temporal contrast algorithm\n"
                "  - E: Show all events\n"
                "  - B: Adjust Biases\n"
                "  - SPACE: Start Recording\n"        
                "  - Q/Escape: Close Camera\n")

    for p in processes:
        p.join()

if __name__ == "__main__":
    main()