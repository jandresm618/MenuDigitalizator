import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

def plot_pixel_info(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Obtener las dimensiones de la imagen
    height, width = image.shape

    # Crear una cuadrícula de coordenadas X e Y
    X, Y = np.meshgrid(range(width), range(height))

    # Aplanar las matrices de coordenadas y la imagen
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = image.flatten()

    # Crear una figura y un eje 3D
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar los datos de los píxeles
    scatter = ax.scatter(X_flat, Y_flat, Z_flat, c=Z_flat, cmap='gray', marker='o')

    # Añadir etiquetas y título
    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Intensidad del píxel')
    ax.set_title('Información de los Píxeles')

    # Añadir una barra de color
    fig.colorbar(scatter, ax=ax, label='Intensidad del píxel')

    # Mostrar la gráfica
    plt.show()

# Función para calcular y normalizar las sumas
def calculate_and_normalize_sums(binary_image):
    sum_y = np.sum(binary_image, axis=1)
    sum_x = np.sum(binary_image, axis=0)
    sum_y_normalized = sum_y / np.max(sum_y)
    sum_x_normalized = sum_x / np.max(sum_x)
    return sum_y_normalized, sum_x_normalized

# Función para detectar posiciones que sobrepasan el umbral
def detect_positions_above_threshold(sums, threshold):
    return np.where(sums > threshold)[0]

# Función para encontrar regiones de interés
def find_regions(positions, min_gap=5):
    regions = []
    current_region = [positions[0]]
    for pos in positions[1:]:
        if pos - current_region[-1] <= min_gap:
            current_region.append(pos)
        else:
            regions.append((current_region[0], current_region[-1]))
            current_region = [pos]
    regions.append((current_region[0], current_region[-1]))
    return regions



def imageHist(image_path,filter=False,kernel = (3, 3)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if filter == 'gaussian': # ****** Filtro Gaussiano **************
        imagen_filtered = cv2.GaussianBlur(image, kernel, 2) #Suavizado de la imagen: "Desenfoque"
    else:
        imagen_filtered = image

    # Binarizar la imagen (Umbral en 127 para convertir la imagen a binaria)
    _, binary_image = cv2.threshold(imagen_filtered, 127, 255, cv2.THRESH_BINARY)

    #Calcular la suma de pixeles en la imagen binarizada
    sum_y_normalized, sum_x_normalized = calculate_and_normalize_sums(binary_image)

    # Crear una figura con dos subplots para los histogramas
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Graficar el histograma de la suma a lo largo del eje Y
    axs[0].bar(range(len(sum_y_normalized)), sum_y_normalized, color='black')
    axs[0].set_title('Suma de píxeles a lo largo del eje Y')
    axs[0].set_xlabel('Y')
    axs[0].set_ylabel('Suma de píxeles')

    # Graficar el histograma de la suma a lo largo del eje X
    axs[1].bar(range(len(sum_x_normalized)), sum_x_normalized, color='black')
    axs[1].set_title('Suma de píxeles a lo largo del eje X')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Suma de píxeles')

    # Mostrar las gráficas
    plt.tight_layout()
    plt.show()

    return sum_y_normalized, sum_x_normalized

# Función para graficar los histogramas y las posiciones sobre el umbral
def plot_histograms_and_threshold(sum_y_normalized, sum_x_normalized, threshold_y, threshold_x):
    positions_y = detect_positions_above_threshold(sum_y_normalized, threshold_y)
    positions_x = detect_positions_above_threshold(sum_x_normalized, threshold_x)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    # Graficar el histograma de la suma normalizada a lo largo del eje Y
    axs[0].bar(range(len(sum_y_normalized)), sum_y_normalized, color='black')
    axs[0].axhline(y=threshold_y, color='red', linestyle='--')
    for pos in positions_y:
        axs[0].axvline(x=pos, color='blue', linestyle='--')
    axs[0].set_title('Suma de píxeles normalizada a lo largo del eje Y')
    axs[0].set_xlabel('Y')
    axs[0].set_ylabel('Suma de píxeles (normalizada)')
    
    # Graficar el histograma de la suma normalizada a lo largo del eje X
    axs[1].bar(range(len(sum_x_normalized)), sum_x_normalized, color='black')
    axs[1].axhline(y=threshold_x, color='red', linestyle='--')
    for pos in positions_x:
        axs[1].axvline(x=pos, color='blue', linestyle='--')
    axs[1].set_title('Suma de píxeles normalizada a lo largo del eje X')
    axs[1].set_xlabel('X')
    axs[1].set_ylabel('Suma de píxeles (normalizada)')
    
    plt.tight_layout()
    plt.show()
    
    return positions_y, positions_x

def plot_image_histogram(binary_image,sum_x_normalized,sum_y_normalized,threshold_x,threshold_y):
    positions_y = detect_positions_above_threshold(sum_y_normalized, threshold_y)
    positions_x = detect_positions_above_threshold(sum_x_normalized, threshold_x)

    # Crear la figura y el grid layout
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1])

    # Graficar el histograma de la suma normalizada a lo largo del eje Y
    ax_hist_y = fig.add_subplot(gs[0, 0])
    ax_hist_y.barh(range(len(sum_y_normalized)), sum_y_normalized, color='black')
    ax_hist_y.axvline(x=threshold_y, color='red', linestyle='--')
    for pos in positions_y:
        ax_hist_y.axvline(x=pos, color='green', linestyle='--')
    ax_hist_y.set_title('Suma de píxeles normalizada a lo largo del eje Y')
    ax_hist_y.set_xlabel('Suma de píxeles (normalizada)')
    ax_hist_y.set_ylabel('Y')
    ax_hist_y.invert_xaxis()

    # Graficar el histograma de la suma normalizada a lo largo del eje X
    ax_hist_x = fig.add_subplot(gs[1, 1])
    ax_hist_x.bar(range(len(sum_x_normalized)), sum_x_normalized, color='black')
    ax_hist_x.axhline(y=threshold_x, color='red', linestyle='--')
    for pos in positions_x:
        ax_hist_x.axvline(x=pos, color='green', linestyle='--')
    ax_hist_x.set_title('Suma de píxeles normalizada a lo largo del eje X')
    ax_hist_x.set_xlabel('X')
    ax_hist_x.set_ylabel('Suma de píxeles (normalizada)')

    # Graficar la imagen original
    ax_img = fig.add_subplot(gs[0, 1])
    ax_img.imshow(binary_image, cmap='gray')
    ax_img.set_title('Imagen Binaria')
    ax_img.axis('off')

    plt.tight_layout()
    plt.show()

def plot_rois(binary_image,sum_x_normalized,sum_y_normalized,threshold_x,threshold_y,regions_x,regions_y):
    # Crear la figura y el grid layout
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(2, 2, width_ratios=[1, 4], height_ratios=[4, 1])

    # Graficar el histograma de la suma normalizada a lo largo del eje Y
    ax_hist_y = fig.add_subplot(gs[0, 0])
    ax_hist_y.barh(range(len(sum_y_normalized)), sum_y_normalized, color='black')
    ax_hist_y.axvline(x=threshold_y, color='red', linestyle='--')
    ax_hist_y.set_title('Suma de píxeles normalizada a lo largo del eje Y')
    ax_hist_y.set_xlabel('Suma de píxeles (normalizada)')
    ax_hist_y.set_ylabel('Y')
    ax_hist_y.invert_xaxis()

    # Graficar el histograma de la suma normalizada a lo largo del eje X
    ax_hist_x = fig.add_subplot(gs[1, 1])
    ax_hist_x.bar(range(len(sum_x_normalized)), sum_x_normalized, color='black')
    ax_hist_x.axhline(y=threshold_x, color='red', linestyle='--')
    ax_hist_x.set_title('Suma de píxeles normalizada a lo largo del eje X')
    ax_hist_x.set_xlabel('X')
    ax_hist_x.set_ylabel('Suma de píxeles (normalizada)')

    # Graficar la imagen original y los rectángulos de interés
    ax_img = fig.add_subplot(gs[0, 1])
    ax_img.imshow(binary_image, cmap='gray')
    ax_img.set_title('Imagen Binaria')
    ax_img.axis('off')

    # Dibujar los rectángulos de interés en la imagen
    for (start_y, end_y) in regions_y:
        for (start_x, end_x) in regions_x:
            rect = Rectangle((start_x, start_y), end_x - start_x, end_y - start_y, linewidth=1, edgecolor='r', facecolor='none')
            ax_img.add_patch(rect)

    plt.tight_layout()
    plt.show()

    print(f'Regiones en Y que sobrepasan el umbral: {regions_y}')
    print(f'Regiones en X que sobrepasan el umbral: {regions_x}')