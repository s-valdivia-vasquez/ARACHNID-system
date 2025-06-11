import os
import re
import math
import shutil

def parse_filename(filename):
    """
    Extrae timestamp (t), posición x, posición y del nombre del archivo.
    Formato esperado: '0124307615_(198_72)_S00433 17-02-25_22.03.18.png'
    """
    #match = re.match(r'^(\d+)_\((\d+)_(\d+)\)*\.png$', filename)
    match = re.match(r'^(\d+)_\((\d+)_(\d+)\)_.*\.png$', filename)
    
    if match:
        t = int(match.group(1))
        x = int(match.group(2))
        y = int(match.group(3))
        return (t, x, y)
    return None

def group_images(filenames, spatial_tolerance=5.0):
    """
    Agrupa imágenes en trayectorias lineales basadas en posición y tiempo.
    
    Args:
        filenames: Lista de nombres de archivo
        spatial_tolerance: Tolerancia espacial en píxeles
        
    Returns:
        Lista de grupos (cada grupo es una lista de nombres de archivo)
    """
    # Procesar y validar archivos
    entries = []
    for filename in filenames:
        parsed = parse_filename(filename)
        if parsed:
            t, x, y = parsed
            entries.append((t, x, y, filename))
    
    # Ordenar por timestamp
    entries.sort(key=lambda e: e[0])
    
    clusters = []
    
    for entry in entries:
        t, x, y, filename = entry
        added = False
        
        # Intentar agregar a un cluster existente (empezando por los más recientes)
        for cluster in reversed(clusters):
            if cluster['max_t'] < t:  # Solo considerar clusters anteriores en el tiempo
                n = cluster['n']
                
                # Calcular predicción de posición
                if n >= 2:
                    # Regresión lineal para n >= 2
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
                        # Caso especial: misma marca de tiempo
                        m_x, b_x = 0, sum_x / n
                        m_y, b_y = 0, sum_y / n
                    
                    pred_x = m_x * t + b_x
                    pred_y = m_y * t + b_y
                else:
                    # Para n=1, usar posición existente
                    pred_x = cluster['sum_x'] / n
                    pred_y = cluster['sum_y'] / n
                
                # Calcular distancia a la predicción
                distance = math.hypot(x - pred_x, y - pred_y)
                
                if distance <= spatial_tolerance:
                    # Actualizar estadísticas del cluster
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
        
        # Si no se agregó a ningún cluster existente, crear uno nuevo
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
    """
    Guarda los archivos agrupados en carpetas separadas
    """
    os.makedirs(output_base_dir, exist_ok=True)
    
    group_counter = 1
    total_images = 0
    skipped_groups = 0
    
    for group in grouped_files:
        # Filtrar grupos con menos de 5 imágenes
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


# Ejemplo de uso
if __name__ == "__main__":
    # Configuración
    source_folder = "C:/Users/optolab/Documents/python/Dataset/Paper figures/S00431 04-09-24_19.19.36"
    output_folder = source_folder
    spatial_tolerance = 10.0  # píxeles
    file_operation = 'move'  # o 'move' para mover en lugar de copiar
    
    # Obtener lista de archivos
    filenames = [f for f in os.listdir(source_folder) if f.endswith('.png')]
    
    # Agrupar imágenes
    grouped = group_images(filenames, spatial_tolerance)
    
    # Guardar en carpetas y obtener estadísticas
    total_images, skipped_groups = save_groups_to_folders(
        grouped_files=grouped,
        source_dir=source_folder,
        output_base_dir=output_folder,
        mode=file_operation
    )
    
    # Mostrar resumen detallado
    print(f"\nResumen final:")
    print(f"Total imágenes procesadas: {len(filenames)}")
    print(f"Grupos detectados: {len(grouped)}")
    print(f"Grupos válidos (5+ imágenes): {len(grouped) - skipped_groups}")
    print(f"Grupos descartados (<5 imágenes): {skipped_groups}")
    print(f"Imágenes en grupos válidos: {total_images}")
    print(f"Imágenes descartadas: {len(filenames) - total_images}")
    print(f"Destino: {output_folder}")