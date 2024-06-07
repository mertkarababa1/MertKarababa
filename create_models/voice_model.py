import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Dense, Dropout, BatchNormalization

# Directory where the RAVDESS dataset is located
dataset_path = '/Users/mertkarababa/Desktop/mk8/.venv/data/ravdess'  # RAVDESS dataset directory

# Directory to save the model
model_save_path = '/Users/mertkarababa/Desktop/mk8/.venv/models'  # Directory to save the model
os.makedirs(model_save_path, exist_ok=True)  # Create the model directory if it does not exist

# Load audio files and labels
def load_data(dataset_path):
    data = []
    labels = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for file in filenames:
            if file.endswith('.wav'):
                file_path = os.path.join(dirpath, file)
                label = file.split('-')[2]  # Extract label according to the RAVDESS filename format
                labels.append(label)
                data.append(file_path)
    return data, labels

# Extract MFCC and Mel Spectrogram features from audio files
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # Data augmentation
    audio = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=2)  # Pitch shifting
    audio = librosa.effects.time_stretch(audio, rate=0.8)  # Time stretching

    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)  # Extract MFCC features
    mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)  # Extract Mel Spectrogram features
    mfccs = np.mean(mfccs.T, axis=0)  # Calculate the mean of MFCCs
    mel = np.mean(mel.T, axis=0)  # Calculate the mean of Mel Spectrograms
    return np.hstack((mfccs, mel))  # Combine MFCC and Mel features

# Load data and extract features
data, labels = load_data(dataset_path)  # Load data and get labels
features = np.array([extract_features(file) for file in data])  # Extract features

# Convert labels to categorical format
label_encoder = LabelEncoder()  # Initialize LabelEncoder
labels_encoded = to_categorical(label_encoder.fit_transform(labels))  # Convert labels to categorical format

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)  # Split data into training and testing sets

# Create the model
model = tf.keras.models.Sequential([
    Dense(512, activation='relu', input_shape=(X_train.shape[1],)),  # First dense layer
    Dropout(0.5),  # Dropout layer
    Dense(256, activation='relu'),  # Second dense layer
    Dropout(0.5),  # Dropout layer
    Dense(128, activation='relu'),  # Third dense layer
    Dropout(0.5),  # Dropout layer
    Dense(labels_encoded.shape[1], activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model

# Train and validate the model
checkpoint = ModelCheckpoint(os.path.join(model_save_path, 'best_model1.keras'), monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)  # Checkpoint to save the best model
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)  # Early stopping

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[checkpoint, early_stopping]  # Use checkpoint and early stopping callbacks
)

# Load and evaluate the best model
best_model = tf.keras.models.load_model(os.path.join(model_save_path, 'best_model1.keras'))  # Load the best model
loss, accuracy = best_model.evaluate(X_test, y_test)  # Evaluate the model
print(f'Best model accuracy: {accuracy * 100:.2f}%')  # Print model accuracy

# Create a more complex model with Conv1D and GRU layers
def build_complex_model(input_shape):
    model = tf.keras.models.Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),  # First 1D convolutional layer
        MaxPooling1D(pool_size=2),  # First max pooling layer
        BatchNormalization(),  # First batch normalization layer
        Conv1D(128, kernel_size=3, activation='relu'),  # Second 1D convolutional layer
        MaxPooling1D(pool_size=2),  # Second max pooling layer
        BatchNormalization(),  # Second batch normalization layer
        GRU(128, return_sequences=True),  # First GRU layer
        GRU(64),  # Second GRU layer
        Dense(256, activation='relu'),  # Dense layer
        Dropout(0.5),  # Dropout layer
        Dense(128, activation='relu'),  # Dense layer
        Dropout(0.5),  # Dropout layer
        Dense(labels_encoded.shape[1], activation='softmax')  # Output layer
    ])
    return model

# Reshape data to 3D
X_train_3d = np.expand_dims(X_train, axis=2)  # Expand training data to 3D
X_test_3d = np.expand_dims(X_test, axis=2)  # Expand testing data to 3D

# Create the complex model
complex_model = build_complex_model((X_train_3d.shape[1], X_train_3d.shape[2]))  # Create the complex model

# Compile the complex model
complex_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the complex model

# Train and validate the complex model
complex_checkpoint = ModelCheckpoint(os.path.join(model_save_path, 'best_complex_model1.keras'), monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)  # Checkpoint to save the best complex model
complex_early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)  # Early stopping for the complex model

complex_history = complex_model.fit(
    X_train_3d, y_train,
    validation_data=(X_test_3d, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[complex_checkpoint, complex_early_stopping]  # Use checkpoint and early stopping callbacks for the complex model
)

# Load and evaluate the best complex model
best_complex_model = tf.keras.models.load_model(os.path.join(model_save_path, 'best_complex_model1.keras'))  # Load the best complex model
complex_loss, complex_accuracy = best_complex_model.evaluate(X_test_3d, y_test)  # Evaluate the complex model
print(f'Best complex model accuracy: {complex_accuracy * 100:.2f}%')  # Print complex model accuracy
