from quickdraw import QuickDrawDataGroup
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.optimizers import Adam

"""
    This script is used to train classification models for the Google Quickdraw dataset
    More information about this dataset can be found here : https://quickdraw.withgoogle.com/data
    
    Author :        Martijn Folmer 
    Date created :  26-03-2023
"""



def get_model(input_shape, n_classes):
    """
    Create a classification model we can train for the purposes of classifying the google quickdraw images
    :param input_shape: The size of the google quickdraw images
    :param n_classes: How many classes of images there exist
    :return: The model itself
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())

    model.add(Dense(300, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))

    model.summary()
    return model


def generate_class_images(name, max_drawings, recognized, image_size):
    """
    Used to download the Google quickdraw data

    :param name: The name of the category we want to download
    :param max_drawings: The maximum number of image we want to download
    :param recognized: Whether they need to be images that were recognized by the google model
    :param image_size: the size of the images we want
    """
    directory = "dataset/" + name

    if not os.path.exists(directory):
        os.mkdir(directory)

    images = QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized)
    for img in images.drawings:
        filename = directory + "/" + str(img.key_id) + ".png"
        img.get_image(stroke_width=6).resize(image_size).save(filename)


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X_filenames, y_output, batch_size, shuffle, num_classes, input_size):
        self.X_filenames = X_filenames      # this is a list of filenames
        self.y_output = y_output            # this is the category

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.X_filenames))
        self.num_classes = num_classes
        self.input_size = input_size

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_filenames) / self.batch_size))

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # get this batch of training data
        X_batch = [cv2.imread(filename, 0) for filename in self.X_filenames[indexes]]
        X_batch = np.asarray([cv2.resize(img, self.input_size[:2]) for img in X_batch])

        for i_img, img in enumerate(X_batch):
            img = np.asarray(img/255, dtype=np.uint8)
            img = img * 255
            X_batch[i_img] = img

        X_batch = np.reshape(X_batch, (self.batch_size, self.input_size[0], self.input_size[1], self.input_size[2]))

        y_batch = self.y_output[indexes]

        # turn output into one-hot encoding
        y_batch = to_categorical(y_batch, self.num_classes)

        return np.asarray(X_batch, dtype=np.float32), np.asarray(y_batch, dtype=np.float32)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


if __name__ == '__main__':

    # Load the quickdraw data for training
    image_size = (100, 100)
    # We want to download from these categories
    names_of_categories = ['snowman', 'airplane', 'castle', 'dragon', 'flying saucer', 'flower', 'hurricane',
                           'hot tub', 'knife', 'ladder']
    tot_labels = len(names_of_categories)
    if not os.path.exists('dataset'):  # If the dataset folder doesn't exist yet, make one
        os.mkdir('dataset')

    frac_trainingData = 0.8   # what percentage of the total data is used for training
    batch_size = 64
    n_epochs = 25              # how many epochs we train
    input_shape = (100, 100, 1) # The size of the images we want to download
    tflite_name = 'final_model_100x100.tflite'  # The name under which the tflite is saved

    # Get all the data we want
    names_of_categories = ['dataset/'+foldername for foldername in names_of_categories]
    n_categories = len(names_of_categories)

    # Download all of the images
    for i_label, label in enumerate(names_of_categories):
        generate_class_images(label, max_drawings=5000, recognized=True, image_size=image_size)
        print(f"We are at : {i_label} / {tot_labels}")

    # Find all of the category names and save them into a npy file we need for when we decode
    names_of_categories_simple = [name.split("/")[-1] for name in names_of_categories]
    np.save('category_to_name.npy', np.asarray(names_of_categories_simple))  # this is what we use to find the

    # create the model and compile it
    classification_model = get_model(input_shape, n_categories)
    classification_model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=["accuracy"])

    # prepare the datagenerators
    X_filenames = []
    y_categories = []
    for i_category, foldername in enumerate(names_of_categories):

        # find the names and the categories they belong to
        curNames = [foldername + f"/{filename}" for filename in os.listdir(foldername)]
        curCategories = [i_category for _ in range(len(curNames))]

        # extend the X_filenames and y_categories
        X_filenames.extend(curNames)
        y_categories.extend(curCategories)

    X_filenames = np.asarray(X_filenames)
    y_categories = np.asarray(y_categories)

    X_train, X_val, y_train, y_val = train_test_split(X_filenames, y_categories, test_size=(1-frac_trainingData))

    # Show how many images we have
    print(f"\nNumber of training images : {X_train.shape[0]}")
    print(f"Number of validation images : {X_val.shape[0]}\n")

    # Initialize our data generators
    trainGenerator = DataGenerator(X_train, y_train, batch_size, shuffle=True, num_classes=n_categories, input_size=input_shape)
    valGenerator = DataGenerator(X_val, y_val, batch_size, shuffle=True, num_classes=n_categories, input_size=input_shape)

    # fit the model
    classification_model.fit(
        trainGenerator,
        validation_data=valGenerator,
        epochs=n_epochs,
        verbose=1,
    )

    # save our model as keras
    save_model(classification_model, 'final_model')

    # save it as a tflite float_16 model
    converter = tf.lite.TFLiteConverter.from_keras_model(classification_model)  # path to model
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_quant_model = converter.convert()  # actually convert the model
    # Save the model.
    with open(tflite_name, 'wb') as f:
        f.write(tflite_quant_model)
    f.close()

    # Run the model and show some results
    if os.path.exists('results') == False:
        os.mkdir('results')
    [os.remove(f'results/{filename}') for filename in os.listdir('results')]

    # find out how many correct and incorrect classifications we have on our validation data
    kn = 0
    k_correct = 0
    k_false = 0
    for idx in range(10):
        # show some results
        try:
            validationData = valGenerator.__getitem__(idx)
        except:
            break

        # Get some results from this model
        imgData = validationData[0]
        groundTruth = validationData[1]
        output_model = classification_model.predict(validationData[0])

        groundTruth = np.argmax(groundTruth, axis=1)
        output_model = np.argmax(output_model, axis=1)

        for i_img, (img, gt, outp) in enumerate(zip(imgData, groundTruth, output_model)):

            groundTruth_cat = names_of_categories_simple[gt]
            outp_cat = names_of_categories_simple[outp]

            img = cv2.resize(img, (500, 500))

            img = cv2.putText(img, f'Groundtruth : {groundTruth_cat}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 1, cv2.LINE_AA)
            img = cv2.putText(img, f'From model: {outp_cat}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

            # check how many are correct
            if groundTruth_cat == outp_cat:
                k_correct += 1
                correct_or_false = "correct"
                correct_or_false_n = k_correct
            else:
                k_false += 1
                correct_or_false = "false"
                correct_or_false_n = k_false

            cv2.imwrite(f'results/{correct_or_false}_{correct_or_false_n}.png', img)
            kn += 1

    # give numbers for results
    print(f"Total number of images in results : {k_correct + k_false}")
    print(f"    Total correct images : {k_correct}")
    print(f"    Total false images : {k_false}")
    print(f"    Success rate of {int(100 * k_correct/(k_correct+k_false))}%")
