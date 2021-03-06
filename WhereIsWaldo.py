import numpy as np
import cv2
import matplotlib.pyplot as plt

nm = 1e-9 #sufijo para especificar que una cantidad esta en nanometros
um = 1e-6 #sufijo para especificar que una cantidad esta en micrometros
mm = 1e-3 #sufijo para especificar que una cantidad esta en milimetros

def plot_func(img1,img2):

    # create figure
    fig = plt.figure(figsize=(10, 7))

    # setting values to rows and column variables
    rows = 1
    columns = 2

    # Adds a subplot at the 1st position
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(img1)
    plt.title("First")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(img2)
    plt.title("Second")

    plt.show()

def transformada_fourier(img):
    fft_imagen = np.fft.fftn(img)
    fft_imagen = np.fft.fftshift(fft_imagen)
    return fft_imagen

def transformada_fourier_inversa(img):
    ifft_img = np.fft.ifftn(img)
    ifft_img = np.fft.ifftshift(ifft_img)
    return ifft_img

def modulo_cuadrado(img):
    modulo = np.abs(img)**2
    return modulo

def modulo_cuadrado_log(img):
    modulo = np.log(np.abs(img)**2)
    return modulo

def graficar_modulo_cuadrado(img):
    plt.imshow( np.log(np.abs(img)**2) )
    plt.show()

def paddin(img,img1):


    old_image_height, old_image_width = img.shape
    new_image_height, new_image_width = img1.shape
    
    result = np.full((new_image_height,new_image_width), 0, dtype=np.uint8)
    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2
    # copy img image into center of result image
    #result[y_center:y_center+old_image_height, 
    #x_center:x_center+old_image_width] = img

    result[0:0+old_image_height, 
    0:0+old_image_width] = img

    return result

def mask(img, center=None, radius=None):
    h = img.shape[0]
    w = img.shape[1]

    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius / 10
    return mask


imagen = cv2.imread('c.jpg', cv2.IMREAD_UNCHANGED)
imagen = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
#imagen_gris = cv2.rotate(imagen_gris, cv2.ROTATE_180)

pista = cv2.imread('c_clue.jpg', cv2.IMREAD_UNCHANGED)
pista = cv2.cvtColor(pista,cv2.COLOR_BGR2RGB)
pista_gris = cv2.cvtColor(pista, cv2.COLOR_RGB2GRAY)
pista_gris_padded = paddin(pista_gris, imagen_gris)
#pista_gris_padded = cv2.rotate(pista_gris_padded, cv2.ROTATE_180)

test1 = cv2.imread('E.png', cv2.IMREAD_UNCHANGED)
test1_gris = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)


test2 = cv2.imread('b.png', cv2.IMREAD_UNCHANGED)
#test2_gris = cv2.cvtColor(test2, cv2.COLOR_BGR2GRAY)



fft_pista = transformada_fourier(pista_gris_padded)
fft_imagen = transformada_fourier(imagen_gris)
fft_test1 = transformada_fourier(test1_gris)
fft_test2 = transformada_fourier(test2)

res = fft_imagen * fft_pista
ires = transformada_fourier_inversa(res)
im = modulo_cuadrado_log(ires)


plot_func(modulo_cuadrado_log(fft_imagen),imagen_gris)