from Builtin_functions import *
import os

project_dir = os.getcwd()
image_path = os.path.join(project_dir, 'Images', 'MenuExample.png')
image_path2 = os.path.join(project_dir, 'Images', 'duetorre.png')
img = cv2.imread(image_path2,0)
print(img.shape)
#imgshow(img,"Imagen original")

#kernel = cv2.getGaussianKernel((5, 5), 2)
#print(kernel)
kernel = (3, 3)
imagen_suavizada = cv2.GaussianBlur(img, kernel, 2)
imgshow(imagen_suavizada,"Imagen con filtro Gaussiano")

kernel=np.ones(kernel, np.uint8)
erode=img.copy()

for i in range(0, 4):
    
    dilate = cv2.dilate(erode, kernel)
    a=("Erosionado{"+str(i+1)+"} times")
    imgshow(dilate,a)



