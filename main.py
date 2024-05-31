from Builtin_functions import *
import os
from ImageInfo import *


image_path = select_image()

if not image_path:
    print("No se seleccionó ninguna imagen.")
    exit()
else:

    # ****** Cargar Imagen **************
    binary_image = preprocessing(image_path)

    #plot_pixel_info(image_path)
    sum_y_normalized, sum_x_normalized = imageHist(image_path)

    threshold_y = float(input("Ingresar umbral en y: "))
    threshold_x = float(input("Ingresar umbral en x: "))

    positions_y = detect_positions_above_threshold(sum_y_normalized, threshold_y)
    positions_x = detect_positions_above_threshold(sum_x_normalized, threshold_x)

    # Encontrar regiones de interés en Y y X
    regions_y = find_regions(positions_y)
    regions_x = find_regions(positions_x)

    plot_image_histogram(binary_image,sum_x_normalized,sum_y_normalized,threshold_x,threshold_y)

    plot_rois(binary_image,sum_x_normalized,sum_y_normalized,threshold_x,threshold_y,regions_x,regions_y)

    templates = setTemplates() # Generar templates de caracteres

    # Crear plantillas simples de caracteres
    templates = {
        'A': np.array([[0, 1, 1, 1, 0],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1]]),
        # Añadir más plantillas según sea necesario
    }

    process_image(binary_image, templates)

    

    

    

    

    

    