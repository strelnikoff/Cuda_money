import cv2 as cv
import numpy as np
import argparse


def get_image(image_name):
    img = cv.imread(image_name)
    return img

def image_show(image):
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def get_contour(image):
    contour = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    contour = cv.GaussianBlur(image, (7,7), 1.5)
    contour = cv.Canny(image, 0, 50)
    return contour

#Ищем все кружки и возвращаем их координаты
def get_points(image):
    img = cv.medianBlur(image, 5)
    gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gimg,cv.HOUGH_GRADIENT,1,160, param1=60,param2=40,minRadius=70,maxRadius=150)
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        cv.circle(img,(i[0],i[1]),i[2],(40,150,40),2)
 #   cv.imshow('Circles',img)
  #  cv.waitKey(0)
   # cv.destroyAllWindows()
    return circles

#Обрезаем кружочки и приводим к одному квадратному размеру
def get_images_by_point(image, circles):
    croped_images = []
    for c in circles[0,:]:
        w = int(c[2] * 2 * 1.1)
        h = int(c[2] * 2 * 1.1)
        x = int(c[0] - w / 2)
        y = int(c[1] - h / 2)
        crop_img = image[y:y + h, x:x + w]
        cv.imshow('cr', crop_img)
        crop_img = cv.resize(crop_img, (256, 256))
        cv.imshow('cr',crop_img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        croped_images.append(crop_img)
    return croped_images

def load_images(path):
    from os import listdir
    from os.path import isfile, join
    files = [f for in listdir(path) if isfile(join(path, f))]
    images = []
    for image_file in files:
        img = cv.imread(join(path, image_file))
        img = cv2.resize(img, (256, 256), interpolation = cv.INTER_CUBIC)
        #в примере делают дичь пока не буду
        images.append(img)
    return images

#Далее пишем классификатор

PATH1 = 'Dataset/1r'
PATH2 = 'Dataset/2r'
PATH3 = 'Dataset/5k'
PATH4 = 'Dataset/5r'
PATH5 = 'Dataset/10k'
PATH6 = 'Dataset/10r'
PATH7 = 'Dataset/50k'
#Берет список папок и возвращает датасет и индексы
def get_data(PATHS):
    images = np.empty(0)
    lables = np.empty(0)
    for i, path in enumerate(PATHS):
        img = load_images(PATH)
        images = np.hstack((images, img))
        lables = np.hstack((labels, np.full((len(img)), i))
    return images, labels
'''
coin1r_images = load_images(PATH1)
coin1r_lables = np.full((len(coin1r_images)), 0)
coin2r_images = load_images(PATH2)
coin2r_lables = np.full((len(coin2r_images)), 1)
coin5k_images = load_images(PATH3)
coin5k_lables = np.full((len(coin5k_images)), 2)
coin5r_images = load_images(PATH4)
coin5r_lables = np.full((len(coin5r_images)), 3)
coin10k_images = load_images(PATH5)
coin10k_lables = np.full((len(coin10k_images)), 4)
coin10r_images = load_images(PATH6)
coin10r_lables = np.full((len(coin10r_images)), 5)
coin50k_images = load_images(PATH7)
coin50k_lables = np.full((len(coin50k_images)), 6)
coin_lables = np.hstack((coin1r_lables, coin2r_lables, coin5k_images, coin5r_images, coin10k_lables, coin10r_lables, coin50k_lables)
coin_images = np.hstack((coin1r_images, coin2r_images, coin5k_images, coin5r_images,coin10k_images, coin10r_images, coin50k_images))
'''

#Строим сеть
import theano
import theano.tensor as T
import lasagne
def build_cnn(input_var=None):
    network = lasagne.layers.InputLayer(shape=(None, 3, 256, 256),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, 
            num_filters=64,
            filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    #
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    #
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=128,
            nonlinearity=lasagne.nonlinearities.rectify)
    #
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    #
    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    #
    assert len(inputs) == len(targets)
    #
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    #
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        #
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]






if __name__ == "__main__":
'''
    image = get_image("frontnear.jpg")
    circles = get_points(image)
    get_images_by_point(image, circles)
'''
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network = build_cnn(input_var)
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),                     dtype=theano.config.floatX)
    pred = test_prediction.argmax(-1)
    f_predict = theano.function([input_var], pred)
    train_fn = theano.function([input_var, target_var], loss,updates=updates, allow_input_downcast=True)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc],allow_input_downcast=True)
    #Обучаем
    num_epochs = 10
    BATCH_SIZE = 128
    #Тут надо считать датасет для обучения и проверки
    for epoch in range(num_epochs):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(image, labels, BATCH_SIZE, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(image_test, labels_test, BATCH_SIZE, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        
        print("Epoch: {}, train:{}, val:{}".format(epoch, 
                                               train_err/train_batches,
                                               val_err/val_batches))

