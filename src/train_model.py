import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -------------------------------
# Paths
# -------------------------------
BASE_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset')
BASE_PATH = os.path.abspath(BASE_PATH)
TRAIN_PATH = os.path.join(BASE_PATH, 'asl_alphabet_train')

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'sign_language_model.h5')
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

print("Training dataset path:", TRAIN_PATH)
print("Model will be saved to:", MODEL_SAVE_PATH)

# -------------------------------
# Image parameters
# -------------------------------
IMG_WIDTH, IMG_HEIGHT = 64, 64
BATCH_SIZE = 32
EPOCHS = 15

# -------------------------------
# Data augmentation and preprocessing
# -------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2  # Split 20% for validation
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

# Validation generator
validation_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# -------------------------------
# Determine number of classes dynamically
# -------------------------------
num_classes = len(train_generator.class_indices)
print(f"Number of classes detected: {num_classes}")

# -------------------------------
# Build CNN model
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# Callbacks
# -------------------------------
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# -------------------------------
# Train the model
# -------------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[checkpoint, early_stop]
)

# -------------------------------
# Plot training history
# -------------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Training and Validation Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
