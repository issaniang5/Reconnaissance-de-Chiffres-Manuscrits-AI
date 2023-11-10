import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ssl
import certifi

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = ssl._create_unverified_context

print("\033[1;33m***** Bienvenue dans la Reconnaissance des Chiffres Manuscrits *****\033[0m")



#Décider s'il faut charger un modèle existant ou en entraîner un nouveau.
train_new_model = True

if train_new_model:
    # Chargement du jeu de données MNIST avec des échantillons et séparation
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

   #Normalisation des données (rendre la longueur égale à 1)
    X_train = tf.keras.utils.normalize(X_train, axis=1)
    X_test = tf.keras.utils.normalize(X_test, axis=1)

 # Créer un modèle de réseau de neurones
 # Ajouter une couche d'entrée aplatie pour les pixels
 # Ajouter deux couches cachées denses
 # Ajouter une couche de sortie dense pour les 10 chiffres
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

   # Compiler et optimiser le modèle
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Entraîner le modèle
    model.fit(X_train, y_train, epochs=3)

    # Évaluer le modèle
    val_loss, val_acc = model.evaluate(X_test, y_test)
    print(val_loss)
    print(val_acc)

    # Enregistrer le modèle
    model.save('reconnaissance-chiffres-manuscrits.keras')
else:
   # Charger le modèle
    model = tf.keras.models.load_model('reconnaissance-chiffres-manuscrits.keras')

# Charger des images personnalisées et les prédire
image_number = 1
while os.path.isfile('digits/digit{}.png'.format(image_number)):
    try:
        img = cv2.imread('digits/digit{}.png'.format(image_number))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print("Le chiffre est probablement {}".format(np.argmax(prediction)))
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        image_number += 1
    except:
        print("Erreur de lecture de l'image ! Passage à l'image suivante...")
        image_number += 1
