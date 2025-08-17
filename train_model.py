import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Optional: Force CPU if GPU issues
# tf.config.set_visible_devices([], 'GPU')

# Dataset path
DATA_DIR = 'dataset/asl_alphabet_train'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15

# Check dataset
print("CWD:", os.getcwd())
if not os.path.exists(DATA_DIR):
    raise SystemExit(f"❌ DATA_DIR not found: {DATA_DIR}")
print("✅ Dataset folder found. Classes:", sorted(os.listdir(DATA_DIR)))

# Data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)

train_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

val_data = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Save label list
labels = list(train_data.class_indices.keys())
with open("labels.txt", "w") as f:
    f.write("\n".join(labels))

# Base model setup
base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze convolution layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
predictions = Dense(len(labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Steps calculation
steps_per_epoch = train_data.samples // train_data.batch_size
validation_steps = val_data.samples // val_data.batch_size

# Sanity check
x_batch, y_batch = next(train_data)
print("✅ Data sanity check:", x_batch.shape, y_batch.shape)

# Train the model
print("🟡 Training started...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1
)
print("✅ Training completed.")

# Save the trained model
model.save("mobilenet_model.h5")
print("✅ Model saved as mobilenet_model.h5")
