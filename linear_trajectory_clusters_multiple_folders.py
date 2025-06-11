import os
import re
import math
import shutil

def parse_filename(filename):
    """Extrae timestamp (t), posición x, posición y del nombre del archivo."""
    match = re.match(r'^(\d+)_\((\d+)_(\d+)\)_.*\.png$', filename)
    if match:
        t = int(match.group(1))
        x = int(match.group(2))
        y = int(match.group(3))
        return (t, x, y)
    return None

def group_images(filenames, spatial_tolerance=5.0):
    """Agrupa imágenes en trayectorias lineales basadas en posición y tiempo."""
    entries = []
    for filename in filenames:
        parsed = parse_filename(filename)
        if parsed:
            t, x, y = parsed
            entries.append((t, x, y, filename))
    
    entries.sort(key=lambda e: e[0])
    clusters = []
    
    for entry in entries:
        t, x, y, filename = entry
        added = False
        
        for cluster in reversed(clusters):
            if cluster['max_t'] < t:
                n = cluster['n']
                if n >= 2:
                    sum_t = cluster['sum_t']
                    sum_x = cluster['sum_x']
                    sum_y = cluster['sum_y']
                    sum_t2 = cluster['sum_t2']
                    sum_tx = cluster['sum_tx']
                    sum_ty = cluster['sum_ty']
                    
                    denom = n * sum_t2 - sum_t**2
                    if denom != 0:
                        m_x = (n * sum_tx - sum_t * sum_x) / denom
                        b_x = (sum_x - m_x * sum_t) / n
                        m_y = (n * sum_ty - sum_t * sum_y) / denom
                        b_y = (sum_y - m_y * sum_t) / n
                    else:
                        m_x, b_x = 0, sum_x / n
                        m_y, b_y = 0, sum_y / n
                    
                    pred_x = m_x * t + b_x
                    pred_y = m_y * t + b_y
                else:
                    pred_x = cluster['sum_x'] / n
                    pred_y = cluster['sum_y'] / n
                
                distance = math.hypot(x - pred_x, y - pred_y)
                if distance <= spatial_tolerance:
                    cluster['n'] += 1
                    cluster['sum_t'] += t
                    cluster['sum_x'] += x
                    cluster['sum_y'] += y
                    cluster['sum_t2'] += t**2
                    cluster['sum_tx'] += t * x
                    cluster['sum_ty'] += t * y
                    cluster['max_t'] = t
                    cluster['filenames'].append(filename)
                    added = True
                    break
        
        if not added:
            clusters.append({
                'n': 1,
                'sum_t': t,
                'sum_x': x,
                'sum_y': y,
                'sum_t2': t**2,
                'sum_tx': t * x,
                'sum_ty': t * y,
                'max_t': t,
                'filenames': [filename]
            })
    
    return [cluster['filenames'] for cluster in clusters]

def save_groups_to_folders(grouped_files, source_dir, output_base_dir, mode='copy'):
    """Guarda los archivos agrupados en carpetas separadas."""
    os.makedirs(output_base_dir, exist_ok=True)
    
    group_counter = 1
    total_images = 0
    skipped_groups = 0
    
    for group in grouped_files:
        if len(group) < 5:
            skipped_groups += 1
            continue
            
        group_dir = os.path.join(output_base_dir, f"group_{group_counter}")
        os.makedirs(group_dir, exist_ok=True)
        
        for filename in group:
            src_path = os.path.join(source_dir, filename)
            dst_path = os.path.join(group_dir, filename)
            
            if mode.lower() == 'copy':
                shutil.copy(src_path, dst_path)
            elif mode.lower() == 'move':
                shutil.move(src_path, dst_path)
        
        total_images += len(group)
        group_counter += 1
    
    return total_images, skipped_groups

if __name__ == "__main__":
    main_folder = "C:/Users/optolab/OneDrive - mail.pucv.cl/python/Figura dataset paper/patches"
    spatial_tolerance = 10.0
    file_operation = 'move'
    
    total_images_all = 0
    total_groups_all = 0
    total_valid_groups_all = 0
    total_skipped_groups_all = 0
    total_discarded_images_all = 0
    processed_folders = 0
    
    for root, _, files in os.walk(main_folder):
        filenames = [f for f in files if f.endswith('.png')]
        if not filenames:
            continue
        
        processed_folders += 1
        print(f"\nProcesando carpeta: {root}")
        
        grouped = group_images(filenames, spatial_tolerance)
        total_images, skipped_groups = save_groups_to_folders(
            grouped, root, root, file_operation
        )
        
        total_filenames = len(filenames)
        detected_groups = len(grouped)
        valid_groups = detected_groups - skipped_groups
        discarded_images = total_filenames - total_images
        
        total_images_all += total_images
        total_groups_all += detected_groups
        total_valid_groups_all += valid_groups
        total_skipped_groups_all += skipped_groups
        total_discarded_images_all += discarded_images
        
        print(f"\nResumen para {root}:")
        print(f"Imágenes procesadas: {total_filenames}")
        print(f"Grupos detectados: {detected_groups}")
        print(f"Grupos válidos: {valid_groups}")
        print(f"Grupos descartados: {skipped_groups}")
        print(f"Imágenes en grupos válidos: {total_images}")
        print(f"Imágenes descartadas: {discarded_images}")
    
    if processed_folders > 0:
        print("\n--- Resumen Final ---")
        print(f"Carpetas procesadas: {processed_folders}")
        print(f"Total imágenes procesadas: {total_images_all + total_discarded_images_all}")
        print(f"Total grupos detectados: {total_groups_all}")
        print(f"Total grupos válidos (5+ imágenes): {total_valid_groups_all}")
        print(f"Total grupos descartados: {total_skipped_groups_all}")
        print(f"Imágenes en grupos válidos: {total_images_all}")
        print(f"Imágenes descartadas: {total_discarded_images_all}")
    else:
        print("\nNo se encontraron carpetas con imágenes PNG.")