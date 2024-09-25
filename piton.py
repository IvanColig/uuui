import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models
from keras.layers import Dropout, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

def get_model(num_classes):
    cnn = models.Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    cnn.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    return cnn

data_location = "./bags/images/bags"

train_data = tf.keras.utils.image_dataset_from_directory(
    data_location,
    validation_split=0.3,
    subset="training",
    seed=5415123,
    image_size=(128, 128),
    batch_size=16,
    shuffle=True
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    data_location,
    validation_split=0.3,
    subset="validation",
    seed=5415123,
    image_size=(128, 128),
    batch_size=16,
    shuffle=True
)

test_data = validation_data.take(len(validation_data) // 2).cache()
validation_data = validation_data.skip(len(validation_data) // 2).cache()

class_names = train_data.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_data.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()

model = get_model(len(class_names))
print(model.summary())

# Training callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="./model/CNN/model.keras", verbose=1, save_best_only=True
)

model.fit(
    train_data,
    validation_data=validation_data,
    epochs=30,
    callbacks=[early_stopping, model_checkpoint]
)

print("Ucitavam najbolji model")
model = tf.keras.models.load_model("./model/CNN/model.keras")

test_loss, test_acc = model.evaluate(test_data)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_acc}")

y_pred = np.argmax(model.predict(test_data), axis=1)
y_real = np.concatenate([y for x, y in test_data], axis=0)

conf_matrix = confusion_matrix(y_real, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

df = pd.DataFrame(conf_matrix, index=class_names, columns=class_names)
print(df)

# Izračunaj precision, recall i F1 score
precision = precision_score(y_real, y_pred, average='weighted')
recall = recall_score(y_real, y_pred, average='weighted')
f1 = f1_score(y_real, y_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")