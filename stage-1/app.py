import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os

# Define Paths to Dataset
base_dir = './dataset'  

# Ensure the dataset exists
if not os.path.exists(base_dir):
    raise FileNotFoundError(f"Dataset folder not found at {base_dir}")

# Image Data Generators for Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 20% data for validation
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15
)

# Prepare Training and Validation Data
train_data = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),  # Resize all images to 224x224
    batch_size=32,
    class_mode='binary',  # Binary classification (liver vs noLiver)
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Build the CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.6),  # Increased dropout to reduce overfitting
    Dense(1, activation='sigmoid')  # Single output neuron for binary classification
])

# Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print Model Summary
model.summary()

# Callbacks for Early Stopping and Learning Rate Reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

# Train the Model
epochs = 10
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs,
    callbacks=[early_stopping, lr_schedule]
)

# Save the Trained Model
model.save('liver_classification_model.h5')
print("Model saved as 'liver_classification_model.h5'")

# Plot Training and Validation Results
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig('training_history.png')  # Save the plot as an image
    print("Training history plot saved as 'training_history.png'")

plot_training_history(history)
