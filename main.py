from Builtin_functions import *
import os

project_dir = os.getcwd()
image_path = os.path.join(project_dir, 'Images', 'MenuExample.png')
img = cv2.imread(image_path,0)
#imgshow(img,"Imagen original")

#kernel = cv2.getGaussianKernel((5, 5), 2)
#print(kernel)
imagen_suavizada = cv2.GaussianBlur(img, (3, 3), 2)
imgshow(imagen_suavizada,"Imagen con filtro Gaussiano")

kernel=np.ones((3, 3), np.uint8)
erode=img.copy()

for i in range(0, 4):
    
    dilate = cv2.dilate(erode, kernel)
    a=("Erosionado{"+str(i+1)+"} times")
    muestra(a,dilate)



