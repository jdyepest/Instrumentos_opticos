import numpy as np
import cv2
import matplotlib.pyplot as plt

nm = 1e-9 #sufijo para especificar que una cantidad esta en nanometros
um = 1e-6 #sufijo para especificar que una cantidad esta en micrometros
mm = 1e-3 #sufijo para especificar que una cantidad esta en milimetros

imagen = cv2.imread('c.jpg')
imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

pista = cv2.imread('c_clue.jpg')
pista_gris = cv2.cvtColor(pista, cv2.COLOR_BGR2GRAY)

cv2.imshow('df',imagen_gris)
cv2.waitKey()


