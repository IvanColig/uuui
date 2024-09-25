import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Funkcija za učitavanje i pripremu slike
def prepare_image(img_path, target_size=(128, 128)):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array
    return img_array

# Učitaj najbolji model
model = tf.keras.models.load_model("./model/CNN/model.keras")

# Definiraj klase
class_names = ['Garbage Bag', 'Paper Bag', 'Plastic Bag']

# Put do slike koju želite predvidjeti
img_path = './test/plasticBag.jpg'

# Pripremite sliku
img_array = prepare_image(img_path)

# Napravite predikciju
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)
predicted_class_name = class_names[predicted_class]
confidence = predictions[0][predicted_class]

print(predictions)

# Prikaz rezultata
plt.imshow(tf.keras.preprocessing.image.load_img(img_path))
plt.title(f'Predicted: {predicted_class_name} ({confidence*100:.2f}%)')
plt.axis('off')
plt.show()
