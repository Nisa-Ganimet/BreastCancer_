# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 11:21:40 2025

@author: nisag
"""

# %% 
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

img = load_img(r"C:\Users\nisag\Desktop\klasor\BreaKHis_v1\histology_slides\breast\benign\SOB\adenosis\SOB_B_A_14-22549AB\40X\SOB_B_A-14-22549AB-40-001.png")
plt.figure(figsize=(3,3))
plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)
print(x.shape)

# %%

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2
)


train_generator = datagen.flow_from_directory(
    directory=r"C:\Users\nisag\Desktop\klasor\BreaKHis_v1\histology_slides\breast",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training",
    shuffle=True
)

test_generator = datagen.flow_from_directory(
    directory=r"C:\Users\nisag\Desktop\klasor\BreaKHis_v1\histology_slides\breast",
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

# %%

from tensorflow.keras.applications import DenseNet121

base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False

# %%

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# %%

from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(1e-4), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

##%

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)

# %%

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=10,
    callbacks=[early_stop, reduce_lr]
)

# %%

model.save_weights("model_weights.weights.h5") 

import json, codecs
import pandas as pd

with codecs.open("history.json", "w", "utf-8") as f:
    json.dump(history.history, f)

pd.DataFrame(history.history).to_csv("history.csv", index=False)

# %%

plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.title("Loss Grafiği")
plt.show()

plt.figure()

plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.legend()
plt.title("Accuracy Grafiği")
plt.show()

##%











