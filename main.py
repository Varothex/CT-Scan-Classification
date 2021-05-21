import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
import seaborn as sn


def read():
    print('Reading files')

    input_file = open('inputML/validation.txt')
    images, labels1, images_4d1 = [], [], []
    line = input_file.readline()
    while line != '':
        image_name, label = line.split(',')
        im = Image.open('inputML/validation/' + image_name).convert("RGB")
        im = np.array(im, dtype='uint8')
        images.append(im)
        labels1.append(int(label))
        img_mat = image.load_img('inputML/validation/' + image_name, target_size=(50, 50, 3))
        img_mat = image.img_to_array(img_mat)
        img_mat = img_mat / 255
        images_4d1.append(img_mat)
        line = input_file.readline()
    input_file.close()

    input_file = open('inputML/train.txt')
    images, labels2, images_4d2 = [], [], []
    line = input_file.readline()
    while line != '':
        image_name, label = line.split(',')
        im = Image.open('inputML/train/' + image_name).convert("RGB")
        im = np.array(im, dtype='uint8')
        images.append(im)
        labels2.append(int(label))
        img_mat = image.load_img('inputML/train/' + image_name, target_size=(50, 50, 3))
        img_mat = image.img_to_array(img_mat)
        img_mat = img_mat / 255
        images_4d2.append(img_mat)
        line = input_file.readline()
    input_file.close()

    input_file = open('inputML/test.txt')
    image_names = []
    images, labels3, images_4d3 = [], [], []
    line = input_file.readline()
    while line != '':
        image_name = line.split()[0]
        label = '0'
        im = Image.open('inputML/test/' + image_name).convert("RGB")
        im = np.array(im, dtype='uint8')
        image_names.append(image_name)
        images.append(im)
        labels3.append(int(label))
        img_mat = image.load_img('inputML/test/' + image_name, target_size=(50, 50, 3))
        img_mat = image.img_to_array(img_mat)
        img_mat = img_mat / 255
        images_4d3.append(img_mat)
        line = input_file.readline()
    input_file.close()

    return np.array(labels1), np.array(labels2), np.array(labels3), image_names, np.array(images_4d1), np.array(images_4d2), np.array(images_4d3)


def transfer_learning(nr_gen, train_images4d, train_labels, valid_images4d, valid_labels, test_images4d):
    # Vom crea un model de bază folosindu-ne de modelul MobileNetV2.
    base_model = tf.keras.applications.MobileNetV2(input_shape=(50, 50, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    # După vom crea un nou model folosindu-ne de cel de bază, după care vom folosi funcția GlobalAveragePooling.
    model = tf.keras.Sequential([base_model,
                                 tf.keras.layers.Conv2D(64, 3, padding="same", input_shape=(50, 50, 3)),
                                 tf.keras.layers.ELU(alpha=0.1),
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dropout(0.25),
                                 tf.keras.layers.Flatten(),
                                 tf.keras.layers.BatchNormalization(),
                                 tf.keras.layers.Dense(3, activation="softmax")])

    # Compilăm modelul:
    base_learning_rate = 0.01
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Normalizăm:
    nparray_train_labels = np.array(train_labels)
    nparray_valid_labels = np.array(valid_labels)

    history = model.fit(train_images4d, nparray_train_labels, epochs=nr_gen, validation_data=(valid_images4d, nparray_valid_labels))

    prediction_val = model.predict_classes(valid_images4d)
    predictions_val = prediction_val.reshape(1, -1)[0]
    prediction_test = model.predict_classes(test_images4d)
    predictions_test = prediction_test.reshape(1, -1)[0]

    # grafice
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(nr_gen)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    confusion = confusion_matrix(valid_labels, prediction_val)
    ax = plt.axes()
    sn.heatmap(confusion, annot=True, annot_kws={"size": 3}, xticklabels=[0, 1, 2], yticklabels=[0, 1, 2], ax=ax)
    ax.set_title('Confusion matrix')
    plt.show()
    ax = plt.axes()
    sn.heatmap(confusion, annot=True, annot_kws={"size": 3}, xticklabels=[0, 1, 2], yticklabels=[0, 1, 2], ax=ax)
    ax.set_title('Confusion matrix')
    plt.show()

    return predictions_val, predictions_test


if __name__ == '__main__':
    print('Write number of generations:')
    nr_gen = int(input())
    valid_labels, train_labels, test_labels, test_images_names, valid_images4d, train_images4d, test_images4d = read()
    predicted_labels_valid, predicted_labels_test = transfer_learning(nr_gen, train_images4d, train_labels, valid_images4d, valid_labels, test_images4d)

    g = open('rezultat.txt', 'w')
    g.write('id,label\n')
    for img, label in zip(test_images_names, predicted_labels_test):
        g.write(img + ',' + str(label) + '\n')
