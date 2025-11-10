import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

def create_emotion_model(num_classes=2, input_shape=(224, 224, 3)):
    """
    Create a transfer learning model for emotion classification
    using MobileNetV2 as the base model.
    """
    # Load pre-trained MobileNetV2 without top layers
    base_model = MobileNetV2(weights='imagenet', 
                            include_top=False, 
                            input_shape=input_shape)
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom classification layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def prepare_data_generators(train_dir, validation_split=0.2, batch_size=32, img_size=(224, 224)):
    """
    Prepare data generators for training and validation.
    """
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=validation_split
    )
    
    # Only rescaling for validation set
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = val_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_generator, validation_generator

def train_model(train_dir, model_save_path='models/emotion_model.h5', epochs=20):
    """
    Train the emotion classification model.
    """
    # Create model
    model = create_emotion_model()
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Prepare data generators
    train_gen, val_gen = prepare_data_generators(train_dir)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True)
    ]
    
    # Train model
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks
    )
    
    # Save final model
    model.save(model_save_path)
    
    # Plot training history
    plot_training_history(history)
    
    return model

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train the model
    model = train_model('dataset')
    print("Model training completed!")