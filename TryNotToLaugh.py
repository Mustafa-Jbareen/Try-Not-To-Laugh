from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.api.preprocessing.image import img_to_array
from keras.api.utils import to_categorical
from keras.src.callbacks import ReduceLROnPlateau
from keras.api.optimizers import Adam
from lenet import LeNet
import numpy as np
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Paths to the GENKI-4K data
images_file = r"Genki_Data\GENKI-R2009a\Subsets\GENKI-4K\GENKI-4K_Images.txt"
labels_file = r"Genki_Data\GENKI-R2009a\Subsets\GENKI-4K\GENKI-4K_Labels.txt"
images_dir = r"Genki_Data\GENKI-R2009a\Subsets\GENKI-4K\files"

# Initialize the list of data and labels
data = []
labels = []

# Load the image paths and corresponding labels
with open(images_file, 'r') as img_file, open(labels_file, 'r') as lbl_file:
    image_filenames = img_file.read().splitlines()
    label_list = lbl_file.read().splitlines()

    # Loop over the images and labels
    for image_filename, label in zip(image_filenames, label_list):
        # Slice the last 8 characters (4 digits + '.jpg') from the filename
        image_filename = "file" + image_filename[-8:]

        # Construct the full path to the image
        image_path = os.path.join(images_dir, image_filename)

        # Load the image
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Ensure at least one face was found
        if len(faces) > 0:
            # Assume the first detected face is the desired face
            (x, y, w, h) = faces[0]
            face = gray[y:y + h, x:x + w]

            # Resize the face to 28x28 pixels
            face = cv2.resize(face, (64, 64))
            face = img_to_array(face)
            data.append(face)

            # Append the label (convert '1' to 'smiling' and '0' to 'not_smiling')
            label = int(label[0])  # Get the first character (either '0' or '1')
            label = 'smiling' if label == 1 else 'not_smiling'
            labels.append(label)
        else:
            # If no face is detected, skip this image
            continue

# Convert the data and labels to NumPy arrays
data = np.array(data, dtype='float') / 255.0  # Normalize pixel values to the range [0, 1]
labels = np.array(labels)

# Convert the labels from strings to integers and then to one-hot vectors
le = LabelEncoder().fit(labels)
labels = to_categorical(le.transform(labels), 2)

# Account for class imbalance
classTotals = labels.sum(axis=0)
classWeight = dict()
for i in range(0, len(classTotals)):
    classWeight[i] = classTotals.max() / classTotals[i]

# Split the data into training and testing sets (80% training, 20% testing)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# Initialize data augmentation for training
aug = ImageDataGenerator(
    rotation_range=30,  # Rotate the image by up to 30 degrees
    width_shift_range=0.1,  # Shift the width by up to 10%
    height_shift_range=0.1,  # Shift the height by up to 10%
    shear_range=0.15,  # Shear the image by up to 15%
    zoom_range=0.25,  # Zoom the image by up to 25%
    horizontal_flip=True,  # Allow horizontal flips
    fill_mode='nearest'  # Fill any pixels lost after transformation
)

# Initialize the model
print('[INFO] compiling model...')
optimizer = Adam(learning_rate=0.001)
model = LeNet.build(width=64, height=64, depth=1, classes=2)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the network with data augmentation
num_epochs = 70
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
H = model.fit(
    aug.flow(trainX, trainY, batch_size=32),  # Use data augmentation
    validation_data=(testX, testY),
    class_weight=classWeight,
    epochs=num_epochs,
    verbose=1,
    callbacks=[reduce_lr]
)

# Evaluate the network
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# Save the model to disk
print('[INFO] serializing network...')
model.save("model.h5")

# Plot training history
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, num_epochs), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, num_epochs), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, num_epochs), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, num_epochs), H.history['val_accuracy'], label='val_accuracy')
plt.title('Training Loss and Accuracy with Data Augmentation')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()
