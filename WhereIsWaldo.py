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
    result[y_center:y_center+old_image_height, 
    x_center:x_center+old_image_width] = img

    return result


imagen = cv2.imread('c.jpg', cv2.IMREAD_UNCHANGED)
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
#imagen_gris = cv2.rotate(imagen_gris, cv2.ROTATE_180)

pista = cv2.imread('c_clue.jpg', cv2.IMREAD_UNCHANGED)
pista_gris = cv2.cvtColor(pista, cv2.COLOR_BGR2GRAY)

pista_gris_padded = paddin(pista_gris, imagen_gris)

tres = cv2.imread('E.png', cv2.IMREAD_UNCHANGED)
tres_gris = cv2.cvtColor(tres, cv2.COLOR_BGR2GRAY)


let = cv2.imread('letters.jpg', cv2.IMREAD_UNCHANGED)
let_gris = cv2.cvtColor(let, cv2.COLOR_BGR2GRAY)

tres_gris = paddin(tres_gris, let_gris)

fft_pista = transformada_fourier(pista_gris_padded)
fft_imagen = transformada_fourier(imagen_gris)
ftt_tres = transformada_fourier(tres_gris)
fft_let = transformada_fourier(let_gris)


#TF = (fft_imagen * fft_pista)
#ITF = transformada_fourier_inversa(TF)
graficar_modulo_cuadrado (fft_pista)

#plot_func(pista_gris,modulo_cuadrado_log(ITF))