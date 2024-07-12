import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input, Conv2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.datasets import mnist
import matplotlib.pyplot as plt
import cv2
import os
#from tensorflow.keras.models import load_model

# Directorio que contiene tus imágenes
data_dir = 'D:/Documents/UNIVERSIDAD/SEMESTRE 2-23/VISION DE MAQUINA/final_project_VM_victor_gonzalez_christian_flores/final_project_VM_victor_gonzalez_christian_flores/capsule/train/good_bad'
#data_dir = './dataset'

# Cargar imágenes desde el directorio
image_list = []
for filename in os.listdir(data_dir):
    img = cv2.imread(os.path.join(data_dir, filename))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
    img = cv2.resize(img, (28, 28))  # Redimensionar a 28x28
    img = (img.astype(np.float32) - 127.5) / 127.5  # Normalizar píxeles al rango [-1, 1]
    img = np.expand_dims(img, axis=-1)  # Agregar la dimensión del canal
    image_list.append(img)

X_train = np.array(image_list)



# Generador
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(7 * 7 * 128, input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))

    model.add(Conv2DTranspose(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(64, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    model.add(Conv2DTranspose(1, kernel_size=3, strides=1, padding='same', activation='tanh'))
    return model

# Discriminador
def build_discriminator(img_shape):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Construir y compilar el discriminador
img_shape = (28, 28, 1)
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

# Construir y compilar el generador
latent_dim = 100
generator = build_generator(latent_dim)

# Construir la GAN
discriminator.trainable = False
z = Input(shape=(latent_dim,))
img = generator(z)
validity = discriminator(img)
gan = Model(z, validity)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Entrenamiento
epochs = 3000
batch_size = 64
half_batch = int(batch_size / 2)

for epoch in range(epochs):
    # Entrenar
    idx = np.random.randint(0, X_train.shape[0], half_batch)
    imgs = X_train[idx]

    #entrenar el generador
    noise = np.random.normal(0, 1, (half_batch, latent_dim))
    gen_imgs = generator.predict(noise)

    #entrenar al discriminador con un batch de imagenes reales y un batch de imagenes falsas
    discriminator.trainable = True
    d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
    d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    discriminator.trainable = False
    # Entrenar la GAN
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    valid_y = np.array([1] * batch_size)
    g_loss = gan.train_on_batch(noise, valid_y)

    # Mostrar el progreso
    if epoch % 100 == 0:
        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")

        # Mostrar imágenes generadas
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, latent_dim))
        gen_imgs = generator.predict(noise) * 0.5 + 0.5  # Reescalar las imágenes generadas al rango [0, 1]

        # Clasificar los objetos generados con el discriminador
        classification = discriminator.predict(gen_imgs)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                axs[i, j].set_title(f'Clasificación: {classification[cnt, 0]:.2f}',fontsize=6)
                cnt += 1
        plt.show()

# Guardar el modelo del discriminador después del entrenamiento
discriminator.save('discriminator_model.h5')
data_dir_test = './capsule/test/crack/crack_001.png'
#data_dir_test = './capsule/test/squeeze/010.png'

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