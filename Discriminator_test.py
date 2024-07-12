import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Conv2D, Conv2DTranspose
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import cv2
import os



#data_dir_test = 'D:/Documents/UNIVERSIDAD/SEMESTRE 2-23/VISION DE MAQUINA/final_project_VM_victor_gonzalez_christian_flores/final_project_VM_victor_gonzalez_christian_flores/capsule/test/crack/crack_001.png'
#data_dir_test = 'D:/Documents/UNIVERSIDAD/SEMESTRE 2-23/VISION DE MAQUINA/final_project_VM_victor_gonzalez_christian_flores/final_project_VM_victor_gonzalez_christian_flores/capsule/test/squeeze/010.png'
data_dir_test = 'D:/Documents/UNIVERSIDAD/SEMESTRE 2-23/VISION DE MAQUINA/final_project_VM_victor_gonzalez_christian_flores/final_project_VM_victor_gonzalez_christian_flores/capsule/capsula_defectuosa_3.webp'
#data_dir_test = 'D:/Documents/UNIVERSIDAD/SEMESTRE 2-23/VISION DE MAQUINA/final_project_VM_victor_gonzalez_christian_flores/final_project_VM_victor_gonzalez_christian_flores/capsule/train/good_bad/009.png'

img_test = cv2.imread(os.path.join(data_dir_test))
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
img_test = cv2.resize(img_test, (28, 28))  # Redimensionar a 28x28
img_test = (img_test.astype(np.float32) - 127.5) / 127.5  # Normalizar píxeles al rango [-1, 1]
img_test = np.expand_dims(img_test, axis=-1)  # Agregar la dimensión del canal


# Realizar la clasificación
loaded_discriminator = load_model('discriminator_model.h5')

#test = loaded_discriminator.predict(img_test)
test = loaded_discriminator.predict(np.expand_dims(img_test, axis=0))
# Imprimir la clasificación
print(f'Clasificación: {test[0, 0]:.2f}')