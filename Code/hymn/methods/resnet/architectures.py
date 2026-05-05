from tensorflow.keras import layers
from tensorflow.keras.models import Model

def residual_block(x, filters, kernel_size=3, stride=1, dropout_rate=0.3):
    shortcut = x  # Save input for residual connection

    # First convolutional layer
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # x = layers.Dropout(dropout_rate)(x)  # Dropout added

    # Second convolutional layer
    x = layers.Conv2D(filters, kernel_size, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Adjust shortcut if dimensions change (due to stride)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding="same", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add residual connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    # x = layers.Dropout(dropout_rate)(x)  # Dropout added

    return x

# ResNet Model for Regression with Dropout
def ResNetRegression(input_shape=(16, 62, 62), dropout_rate=0.3):
    inputs = layers.Input(shape=input_shape)

    # Initial Conv Layer
    x = layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # x = layers.Dropout(dropout_rate)(x)  # Dropout added
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    # Residual Blocks
    x = residual_block(x, 64, dropout_rate=dropout_rate)
    x = residual_block(x, 128, stride=2, dropout_rate=dropout_rate)  # Downsampling
    x = residual_block(x, 256, stride=2, dropout_rate=dropout_rate)  # Downsampling
    x = residual_block(x, 512, stride=2, dropout_rate=dropout_rate)  # Downsampling

    # Global Average Pooling & Regression Output
    # x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)  # Dropout before final dense layer
    
    outputs = layers.Dense(2, activation="linear")(x)  # Output (x, y coordinates)

    model = Model(inputs, outputs, name="ResNet_Regression")
    return model