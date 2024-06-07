from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Defining data paths
train_data_dir = '/Users/mertkarababa/Desktop/mk8/.venv/data/fer-2013/train' # Training data directory
validation_data_dir = '/Users/mertkarababa/Desktop/mk8/.venv/data/fer-2013/test' # Validation data directory

# Setting up data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255, # Rescale pixel values to [0, 1]
    rotation_range=30, # Randomly rotate images by up to 30 degrees
    shear_range=0.3, # Apply shear transformations
    zoom_range=0.3, # Randomly zoom in on images
    horizontal_flip=True, # Randomly flip images horizontally
    fill_mode='nearest') # Fill in newly created pixels after transformations

# Only rescaling for validation data
validation_datagen = ImageDataGenerator(rescale=1./255) # Rescale pixel values to [0, 1]

# Creating the training data generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir, # Directory for training data
    color_mode='grayscale', # Convert images to grayscale
    target_size=(48, 48), # Resize images to 48x48 pixels
    batch_size=32, # Number of images to yield per batch
    class_mode='categorical', # Use categorical labels
    shuffle=True) # Shuffle the data

# Creating the validation data generator
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir, # Directory for validation data
    color_mode='grayscale', # Convert images to grayscale
    target_size=(48, 48), # Resize images to 48x48 pixels
    batch_size=32, # Number of images to yield per batch
    class_mode='categorical', # Use categorical labels
    shuffle=True) # Shuffle the data

# Defining class labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
img, label = train_generator.__next__() # Get a batch of images and labels

# Creating the model
model = Sequential() # Initialize a sequential model
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1))) # Add a 2D convolutional layer with 32 filters
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu')) # Add another 2D convolutional layer with 64 filters
model.add(MaxPooling2D(pool_size=(2, 2))) # Add a max pooling layer
model.add(Dropout(0.1)) # Add dropout to prevent overfitting

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu')) # Add a 2D convolutional layer with 128 filters
model.add(MaxPooling2D(pool_size=(2, 2))) # Add a max pooling layer
model.add(Dropout(0.1)) # Add dropout to prevent overfitting

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu')) # Add a 2D convolutional layer with 256 filters
model.add(MaxPooling2D(pool_size=(2, 2))) # Add a max pooling layer
model.add(Dropout(0.1)) # Add dropout to prevent overfitting

model.add(Flatten()) # Flatten the 3D output to 1D
model.add(Dense(512, activation='relu')) # Add a fully connected layer with 512 units
model.add(Dropout(0.2)) # Add dropout to prevent overfitting

model.add(Dense(7, activation='softmax')) # Add the output layer with 7 units (one for each class)

# Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model with Adam optimizer and categorical crossentropy loss
print(model.summary()) # Print the model summary

# Training the model
epochs = 30 # Number of epochs to train the model

history = model.fit(
    train_generator, # Training data generator
    steps_per_epoch=train_generator.samples // train_generator.batch_size, # Number of steps per epoch
    epochs=epochs, # Number of epochs
    validation_data=validation_generator, # Validation data generator
    validation_steps=validation_generator.samples // validation_generator.batch_size) # Number of validation steps

# Saving the model
model.save('models/cam_model.h5', include_optimizer=True) # Save the trained model
