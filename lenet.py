# import the necessary packages
from keras import backend as K
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Activation, Flatten, Dense, GlobalAveragePooling2D
from keras.api.layers import LeakyReLU  # Import LeakyReLU

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        '''
        Build the LeNet architecture with Leaky ReLU.
        '''
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using 'channels first', update the input shape
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        # First set of CONV => LeakyReLU => POOL
        model.add(Conv2D(32, (5, 5), padding='same', input_shape=inputShape))
        model.add(BatchNormalization())  # Batch normalization
        model.add(LeakyReLU(alpha=0.1))  # Leaky ReLU with alpha=0.1
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second set of CONV => LeakyReLU => POOL
        model.add(Conv2D(64, (5, 5), padding='same'))
        model.add(BatchNormalization())  # Batch normalization
        model.add(LeakyReLU(alpha=0.1))  # Leaky ReLU with alpha=0.1
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Flattening the feature maps and applying fully connected layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())  # Batch normalization
        model.add(LeakyReLU(alpha=0.1))  # Leaky ReLU with alpha=0.1
        model.add(Dropout(0.5))  # Dropout to prevent overfitting

        # Output layer with softmax activation for classification
        model.add(Dense(classes, activation='softmax'))

        return model



# class LeNet:
#     @staticmethod
#     def build(width, height, depth, classes):
#         '''
#         initialize the model
#         '''
#         model = Sequential()
#         inputShape = (height, width, depth)
#
#         # if we are using 'channels first', update the input shape
#         if K.image_data_format() == 'channels_first':
#             inputShape = (depth, height, width)
#
#         # first set of CONV => ReLU => POOL layers
#         model.add(Conv2D(20, (5, 5), padding='same', input_shape=inputShape))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
#         # second layer of CONV => ReLU => POOL layers
#         model.add(Conv2D(50, (5, 5), padding='same'))
#         model.add(Activation('relu'))
#         model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
#         # first (and only) set of FC => ReLU layeres
#         model.add(Flatten())
#         model.add(Dense(500))
#         model.add(Activation('relu'))
#
#         # softmax classifier
#         model.add(Dense(classes))
#         model.add(Activation('softmax'))
#
#         return model



# class LeNet:
#     @staticmethod
#     def build(width, height, depth, classes):
#         model = Sequential()
#         inputShape = (height, width, depth)
#
#         if K.image_data_format() == 'channels_first':
#             inputShape = (depth, height, width)
#
#         # First set of CONV => ReLU => POOL
#         model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputShape))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#         model.add(Dropout(0.25))  # Spatial dropout
#
#         # Second set of CONV => ReLU => POOL
#         model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#         model.add(Dropout(0.25))
#
#         # Third set of CONV => ReLU => POOL
#         model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
#         model.add(BatchNormalization())
#         model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#         model.add(Dropout(0.25))
#
#         # Global Average Pooling instead of Flattening
#         model.add(GlobalAveragePooling2D())
#
#         # Fully connected layer
#         model.add(Dense(512, activation='relu'))
#         model.add(BatchNormalization())
#         model.add(Dropout(0.5))
#
#         # Output layer
#         model.add(Dense(classes, activation='softmax'))
#
#         return model