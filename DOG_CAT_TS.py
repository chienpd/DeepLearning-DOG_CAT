import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from random import shuffle, randint
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import matplotlib.pyplot as plt
# import tensorflow as tf
#
# tf.reset_default_graph()

TRAIN_DIR = os.path.dirname(os.path.realpath(__file__)) + "/data/train"
TEST_DIR = os.path.dirname(os.path.realpath(__file__)) + "/data/test"
# REAL_TEST_DIR = os.path.dirname(os.path.realpath(__file__)) + "/data/real_test"
IMG_SIZE = 48
LR = 1e-3

MODEL_NAME = 'dogsvscats2.model'


def label_img(img):
    word_label = img.split('.')[-3]
    if word_label == 'cat':
        return [1, 0]
    elif word_label == 'dog':
        return [0, 1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    # shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


# def real_test_data():
#     testing_data = []
#     for img in tqdm(os.listdir(REAL_TEST_DIR)):
#         path = os.path.join(REAL_TEST_DIR, img)
#         img_num = img.split('.')[0]
#         img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
#         testing_data.append([np.array(img), img_num])
#     # shuffle(testing_data)
#     np.save('real_test_data.npy', testing_data)
#     return testing_data


if os.path.exists('train_data.npy'):
    train_data = np.load('train_data.npy')
    print('train data loaded !')
else:
    print('train data creating !')
    train_data = create_train_data()
    print('train data created !')

# CNN MODEL
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 256, 3, activation='relu')
convnet = conv_2d(convnet, 256, 3, activation='relu')
# convnet = conv_2d(convnet, 256, 3, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

# TRAINING MODEL
train = train_data[:-1000]
test = train_data[-1000:]

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

if os.path.exists('test_data.npy'):
    test_data = np.load('test_data.npy')
    print('test data loaded !')
else:
    print('test data processing !')
    test_data = process_test_data()
    print('test data processed !')

# if os.path.exists('real_test_data.npy'):
#     real_test_data = np.load('real_test_data.npy')
#     print('real test data loaded !')
# else:
# print('real test data processing !')
# real_test_data = real_test_data()
# print('real test data processed !')

# print('real test data processing !')
# real_test_data = real_test_data()
# print('real test data processed !')

fig = plt.figure()


def test():
    rand = randint(1, len(test_data) - 12)
    # print(rand)
    for num, data in enumerate(test_data[rand:rand + 12]):
        # cat: [1,0]
        # dog: [0,1]

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 1:
            str_label = 'Dog: ' + str(round(model_out[1], 4)*100)
        else:
            str_label = 'Cat: ' + str(round(1 - model_out[1], 4)*100)

        y.imshow(orig, cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()


test()

# fig2 = plt.figure()
#
#
# # print(len(real_test_data))
# def real_test():
#     rand = randint(1, len(real_test_data) - 12)
#     # print(rand)
#     for num, data in enumerate(real_test_data[rand:rand + 12]):
#         # cat: [1,0]
#         # dog: [0,1]
#
#         img_num = data[1]
#         img_data = data[0]
#
#         y = fig2.add_subplot(3, 4, num + 1)
#         orig = img_data
#         data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
#         model_out = model.predict([data])[0]
#
#         if np.argmax(model_out) == 1:
#             str_label = 'Dog: ' + str(round(model_out[1], 4)*100)
#         else:
#             str_label = 'Cat: ' + str(round(1 - model_out[1], 4)*100)
#
#         y.imshow(orig, cmap='gray')
#         plt.title(str_label)
#         y.axes.get_xaxis().set_visible(False)
#         y.axes.get_yaxis().set_visible(False)
#     plt.show()
# real_test()


print("Save Submission !")
with open('submission_file.csv', 'w') as f:
    f.write('id,label\n')

with open('submission_file.csv', 'a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        # orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num, model_out[1]))

print("Save Submission 2 !")
with open('submission_file_real.csv', 'w') as f:
    f.write('id,label\n')
#
# with open('submission_file_real.csv', 'a') as f:
#     for data in tqdm(real_test_data):
#         img_num = data[1]
#         img_data = data[0]
#         # orig = img_data
#         data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
#         model_out = model.predict([data])[0]
#         f.write('{},{}\n'.format(img_num, model_out[1]))
