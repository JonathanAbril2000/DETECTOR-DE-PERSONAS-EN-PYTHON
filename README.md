# DETECTOR-DE-PERSONAS-EN-PYTHON
Este código Python utiliza la biblioteca OpenCV para la detección de movimiento en un video específico. La aplicación se centra en la identificación y conteo de personas en dos áreas de interés predeterminadas dentro del cuadro del video. 

El script comienza inicializando la captura de video desde un archivo y configura un sustractor de fondo MOG2 para identificar cambios en la escena. Dos áreas de interés (ROIs) son definidas mediante conjuntos de puntos, y el código monitorea el movimiento de las personas dentro de estas áreas específicas.

El programa realiza el procesamiento de imagen y el conteo de personas en cada área de interés. La información del conteo se visualiza en tiempo real sobre el cuadro del video original, permitiendo una rápida evaluación de la actividad en las regiones designadas.

Este código es útil para situaciones donde se desea realizar un seguimiento y conteo de personas en áreas específicas de un entorno, como la vigilancia de zonas críticas o la gestión de flujos de tráfico en tiempo real.



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''CÓDIGO DE PROGRAMACIÓN'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''


import tkinter
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# Inicializar la captura de video desde el archivo 'aeropuerto.mp4'
cap = cv2.VideoCapture('aeropuerto.mp4')

# Crear un sustractor de fondo MOG2 para la detección de movimiento
fgbg = cv2.createBackgroundSubtractorMOG2()

# Definir el kernel para operaciones morfológicas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Definir las coordenadas de las áreas de interés (ROIs)
area1_pts = np.array([[240, 320], [480, 320], [620, cap.get(4)], [50, cap.get(4)]])
area2_pts = np.array([[640, 0], [cap.get(3), 0], [cap.get(3), cap.get(4)], [700, cap.get(4)]])

# Contadores para el número de personas en cada área
cont_personas_area1 = 0
cont_personas_area2 = 0

while True:
    # Leer un cuadro del video
    ret, frame = cap.read()

    # Salir del bucle si no hay más cuadros
    if ret == False:
        break

    # Convertir el cuadro a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Procesar la primera área de interés
    imAux1 = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux1 = cv2.drawContours(imAux1, [area1_pts], -1, (255), -1)
    image_area1 = cv2.bitwise_and(gray, gray, mask=imAux1)

    # Aplicar sustracción de fondo y operaciones morfológicas para la primera área
    fgmask1 = fgbg.apply(image_area1)
    fgmask1 = cv2.morphologyEx(fgmask1, cv2.MORPH_OPEN, kernel)
    fgmask1 = cv2.dilate(fgmask1, None, iterations=2)

    # Encontrar contornos y contar personas en la primera área
    cnts_area1, _ = cv2.findContours(fgmask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt_area1 in cnts_area1:
        if cv2.contourArea(cnt_area1) > 500:
            x, y, w, h = cv2.boundingRect(cnt_area1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cont_personas_area1 += 1

    # Procesar la segunda área de interés
    imAux2 = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
    imAux2 = cv2.drawContours(imAux2, [area2_pts], -1, (255), -1)
    image_area2 = cv2.bitwise_and(gray, gray, mask=imAux2)

    # Aplicar sustracción de fondo y operaciones morfológicas para la segunda área
    fgmask2 = fgbg.apply(image_area2)
    fgmask2 = cv2.morphologyEx(fgmask2, cv2.MORPH_OPEN, kernel)
    fgmask2 = cv2.dilate(fgmask2, None, iterations=2)

    # Encontrar contornos y contar personas en la segunda área
    cnts_area2, _ = cv2.findContours(fgmask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt_area2 in cnts_area2:
        if cv2.contourArea(cnt_area2) > 500:
            x, y, w, h = cv2.boundingRect(cnt_area2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cont_personas_area2 += 1

    # Mostrar el cuadro original y el número de personas en cada área
    cv2.putText(frame, f'Area 1: {cont_personas_area1} personas', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Area 2: {cont_personas_area2} personas', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("frame", frame)

    # Manejo de eventos del teclado
    k = cv2.waitKey(70) & 0xFF
    if k == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()





LINK DE DESCARGA DEL VIDEO: https://drive.google.com/file/d/1ZNlC-0FhU69GL5pN-Ngy_brOALvTycVO/view?usp=sharing




