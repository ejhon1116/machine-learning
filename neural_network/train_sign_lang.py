import os
import numpy as np
import pandas
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, saving
from pathlib import Path
import matplotlib.pyplot as plt
import tempfile
from kaggle_data.paths import *

# Custom learning rate scheduler with better decay strategy
@saving.register_keras_serializable(package="LR")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate: float, decay_steps: float, warmup_steps=1000):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        # Warmup phase
        warmup_lr = (self.initial_learning_rate * step / self.warmup_steps)
        
        # Cosine decay phase
        decay_progress = (step - self.warmup_steps) / (self.decay_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + tf.cos((tf.constant(np.pi, dtype=tf.float64) * decay_progress)))
        cosine_lr = self.initial_learning_rate * cosine_decay
        
        # Combine warmup and decay
        return tf.where(step < self.warmup_steps, tf.cast(warmup_lr, tf.float32), tf.cast(cosine_lr, tf.float32))
    def get_config(self):
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_steps': self.decay_steps,
            'warmup_steps': self.warmup_steps
        }
    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_improved_model():
    """Create improved CNN model for sign language recognition"""
    inputs = layers.Input(shape=(28, 28, 1))
    
    # Image augmentation
    x = layers.RandomRotation(0.1)(inputs)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomTranslation(0.1, 0.1)(x)

    # Initial convolution block
    x = layers.Conv2D(16, 3, activation=None, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    
    # First conv block with residual connection
    x = layers.Conv2D(32, 3, padding='same')(x)
    skip = x
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])  # Residual connection
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    
    # Second conv block
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    #third
    x = layers.Conv2D(96, 3, padding='same')(x)
    skip = x
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(96, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])  # Residual connection
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Conv2D(128, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Dense layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(26, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

def train_model(train_images, train_labels):
    """Train the model with improved training loop"""
    
    # Create model
    model = create_improved_model()
    
    # Learning rate schedule
    total_steps = (len(train_images) // 32) * 30 # based on epoch count
    lr_schedule = WarmupCosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=total_steps,
        warmup_steps=total_steps // 10
    )
    
    # Compile model
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join('neural_network', 'sign_lang_models', 'epoch_{epoch:02d}.model.keras'),
            save_freq='epoch'
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join('neural_network', 'sign_lang_models', 'max_accuracy.model.keras'),
            monitor='accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join('neural_network', 'sign_lang_models', 'min_loss.model.keras'),
            monitor='loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )

    ]
    
    # Train
    history = model.fit(
        train_images,
        train_labels,
        epochs=30,
        callbacks=callbacks_list
    )
    
    return model, history

if __name__ == "__main__":
    data = pandas.read_csv(Path(sign_lang_path + "sign_mnist_train.csv")).values
    train_labels = data[:, 0]
    train_images = ((data[:, 1:]).reshape(27455, 28, 28) / 255.0)

    model, history = train_model(train_images, train_labels)