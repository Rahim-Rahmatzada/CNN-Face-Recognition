import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Dataset from https://www.kaggle.com/datasets/apollo2506/facial-recognition-dataset

class TrainingModel:
    """
    Class responsible for building, training, and saving an emotion classification model
    """

    def __init__(self):
        """
        Initialize directories for training/testing data
        Build CNN model
        """
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.train_dir = os.path.join(base_dir, "Training")
        self.test_dir = os.path.join(base_dir, "Testing")
        self.model = self.build_model()

    def build_model(self):
        """
        Build and compile Convolutional Neural Network

        Returns:
            model: Compiled CNN model
        """
        model = Sequential([
            # First convolutional layer with input shape for grayscale images
            Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
            BatchNormalization(),  # Normalization layer
            MaxPooling2D(2, 2),
            Dropout(0.25),  # Regularization layer

            # Second convolutional layer
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Third convolutional layer
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),

            # Flatten and fully connected layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(6, activation='softmax')  # Six categories for emotion classification
        ])

        # Compile the model with Adam optimizer
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self):
        """
        Train the model using training data

        Apply data augmentation techniques to training set
        early stopping, model checkpointing,learning rate reduction

        Returns:
            history: Training history containing accuracy and loss metrics
        """
        # Data augmentation configuration for training data
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )

        # No augmentation for validation data
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(48, 48),
            color_mode="grayscale",
            batch_size=32,
            class_mode="categorical",
            subset="training"
        )

        validation_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(48, 48),
            color_mode="grayscale",
            batch_size=32,
            class_mode="categorical",
            subset="validation"
        )

        # Callbacks for training process
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        check_point = ModelCheckpoint('best_weights.h5', save_best_only=True, monitor='val_accuracy', mode='max')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

        # Train the model
        history = self.model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // 32,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // 32,
            epochs=40,
            callbacks=[early_stop, check_point, reduce_lr]
        )

        # Save the final model
        self.model.save("emotion_model_V3")

        return history


# Train the model
model = TrainingModel()
model.train()
