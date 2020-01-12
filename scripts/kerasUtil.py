import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
from tensorflow.keras import models, layers, optimizers
import matplotlib.pyplot as plt
import math
import cv2
import os
import pickle
import time
from itertools import product
from PIL import *
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'data/train'
validation_data_dir = 'data/val'

# number of epochs to train top model
epochs = 35
# batch size used by flow_from_directory and predict_generator
batch_size = 16


def callModel(model_picked = 'vgg16'):
    '''function returns the model picked based on input
    Input choices:
        'vgg16'     - VGG16
        'vgg19'     - VGG19
        'res50'     - ResNet50
        'xception'  - Xception
        'inception' - InceptionV3
        'monet'     - MobileNetV2
    '''
    #The models have a series of convolutional layers and then they have dense(deeply connected layers)
    #include_top = False only gets the convo layers and ignores the dense layer
    #imagenet is a huge image dataset on which the models are trained. if weights ='imagenet' means the weights are acquired from that.
    if model_picked == 'vgg16':
        model = applications.VGG16(include_top=False, weights='imagenet')
    elif model_picked =='vgg19':
        model = applications.VGG19(include_top=False, weights='imagenet')
    elif model_picked == 'res50':
        model = applications.ResNet50(include_top=False, weights='imagenet')
    elif model_picked == 'xception':
        model = applications.Xception(include_top=False, weights='imagenet')
    elif model_picked == 'inception':
        model = applications.InceptionV3(include_top=False, weights='imagenet')
    elif model_picked == 'monet':
        model = applications.MobileNetV2(include_top=False, weights='imagenet',
        input_shape=(224,224,3))
    return model

def callOptimizer(opt='rmsprop'):
    '''Function returns the optimizer to use in .fit()
    options:
        adam, sgd, rmsprop, ada_grad,ada_delta,ada_max
    '''
    opt_dict = {'adam': optimizers.Adam(),
                'sgd' : optimizers.SGD(),
                'rmsprop' : optimizers.RMSprop(),
                'ada_grad' : optimizers.Adagrad(),
                'ada_delta': optimizers.Adadelta(),
                'ada_max'  : optimizers.Adamax()}

    return opt_dict[opt]

def save_bottleneck_features(model_picked='vgg16'):
    '''function returns the model picked based on input
    Input choices:
        model_picked - 'vgg16', 'vgg19', 'res50', 'xception'
                        'inception', 'monet'
    '''

    # build the convnet base network
    model = callModel(model_picked)

    datagen = ImageDataGenerator(rescale=1. / 255)
    
    #batch_size 
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,  # this means our generator will only yield batches of data, no labels
        shuffle=False)

    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))

    nb_train_samples = len(generator.filenames)
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save(f'models/{model_picked}_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None, # this means our generator will only yield batches of data, no labels
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))
    # the predict_generator method returns the output of a model, given
    # a generator that yields batches of numpy data

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)
    # save the output as a Numpy array
    np.save(f'models/{model_picked}_features_validation.npy',
            bottleneck_features_validation)
    return model_picked

def train_top_model(model_picked, last_act_func='softmax', my_optimizer= 'rmsprop'):
    '''Function returns tuple of history dictionary, loss and accuracy (values)
    INPUT:
    model_picked  - the same input/output from save_bottleneck_features
    last_act_func - Options include: softmax, sigmoid
    my_optimizer  - passed into callOptimizer(), w/ options including: rmsprop,
                    adam, sgd, ada_grad, ada_delta, ada_max
    '''

    datagen_top = ImageDataGenerator(rescale=1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    # save the class indices to use use later in predictions
    np.save('models/train_class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load(f'models/{model_picked}_features_train.npy')

    # get the class labels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load(f'models/{model_picked}_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
                            validation_labels, num_classes=num_classes)
    print(train_data.shape)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation=last_act_func))

    model.compile(optimizer= callOptimizer(my_optimizer),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    top_model_weights_path = f'models/{model_picked}_fc_model.h5'
    model.save_weights(top_model_weights_path)

    model.save('models/trained_model.hdf5')
    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    plotCurves(history)
    return history, eval_loss, eval_accuracy

def plotCurves(hist):
    plt.figure(1, figsize=(8,12))
    # summarize history for accuracy
    plt.subplot(211)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.subplot(212)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.show();

def collectCombo(input_models=None, input_optims= None, input_funcs=None):
    '''function returns a tuple (df, dictionary), i.e., a summary & history from
    the fit method
    INPUT
        input_models - e.g., ['vgg16','vgg19'], or None (default) runs all
        input_optims - e.g., ['rmsprop','sgd'], or None (default) runs all
        input_funcs  - e.g., ['softmax','sigmoid'], or None (default) runs all
    '''
    if input_models==None:
        list_models = ['vgg16','vgg19','monet','xception','inception','res50']
        countModels = 6
    else:
        list_models = input_models
        countModels= len(input_models)

    if input_optims==None:
        list_optims = ['rmsprop','adam','sgd','ada_grad','ada_delta','ada_max']
        countOptims = 6
    else:
        list_optims = input_optims
        countOptims= len(input_optims)

    if input_funcs==None:
        list_funcs = ['softmax','sigmoid']
        countFuncs = 2
    else:
        list_funcs= input_funcs
        countFuncs=len(input_funcs)

    countMax= countModels * countOptims * countFuncs
    count = 1
    collectHist = dict()
    listMod, listFunc, listOpt, listLoss, listAcc, listTime = [],[],[],[],[],[]
    for mod, func, opt in product(list_models, list_funcs, list_optims):
        t1 = time.time()
        savemodel = save_bottleneck_features(mod)
        history, loss, acc = train_top_model(savemodel, last_act_func=func,
                                my_optimizer=opt)
        deltatime = (time.time()-t1)/60
        collectHist[(mod,opt)] = history
        listMod.append(mod)
        listFunc.append(func)
        listOpt.append(opt)
        listAcc.append(acc)
        listLoss.append(loss)
        listTime.append(deltatime)
        print(str(count),' / ',str(countMax))
        count +=1

    df = pd.DataFrame(np.c_[listMod, listFunc,listOpt,listAcc,listLoss, listTime],
                columns =['model','function','optimizer','accuracy','loss', 'time'])

    with open('models/collectionDf.pkl','wb') as fout:
        pickle.dump(df, fout)
    with open('models/collectionCombo.pkl','wb') as fout:
        pickle.dump(collectHist, fout)
    return df, collectHist

def predict(model_picked='vgg16', last_act_func='softmax', class_picked=None, pic_no=None):
    '''function returns prediction of images
    Input:
        last_act_func = softmax or sigmoid
        model_picked  = vgg16, vgg19, res50
        class_picked = is a string of classes ('bellpepper', 'bokchoy','broccoli','carrot','mushroom','tomato'
        pic_no       = which picture to predict?
    '''
    # load the class_indices saved in the earlier step
    class_dictionary = np.load('models/class_indices.npy').item()

    num_classes = len(class_dictionary)

    # add the path to your test image below
    image_path_dir = 'data/val/'
    # If class is specified
    if class_picked != None:
        list_image_choices= os.listdir(image_path_dir + class_picked)
        if pic_no !=None:
            pic_picked = os.listdir(image_path_dir + class_picked)[pic_no]
            picture = image_path_dir + class_picked + '/'+ pic_picked
        else: # IF picture is NOT specified, then randomly pick
            pic_picked = os.listdir(image_path_dir + class_picked)[np.random.randint(1,21)]
            picture = image_path_dir + class_picked + '/'+ pic_picked

    # If not specified, then choose CLASS randomly
    else:
        class_picked = os.listdir(image_path_dir)[np.random.randint(1,num_classes+1)]
        # If pic selection is specified
        if pic_no !=None:
            pic_picked = os.listdir(image_path_dir + class_picked)[pic_no]
            picture = image_path_dir + class_picked + '/'+ pic_picked
        else: # IF picture is NOT specified, then randomly pick
            pic_picked = os.listdir(image_path_dir + class_picked)[np.random.randint(1,21)]
            picture = image_path_dir + class_picked + '/'+ pic_picked

    orig = cv2.imread(picture)

    print("[INFO] loading and preprocessing image...")
    image = load_img(picture, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = callModel(model_picked)

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    # model = Sequential()
    # model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    # model.add(Dense(256, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation=last_act_func))
    #
    # top_model_weights_path = f'models/{model_picked}_fc_model.h5'
    # model.load_weights(top_model_weights_path)
    model = load_model('models/trained_model.hdf5')
    # use the bottleneck prediction on the top model to get the final
    # classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    inID = class_predicted[0]

    inv_map = {v: k for k, v in class_dictionary.items()}

    label = inv_map[inID]

    # get the prediction label
    WINDOW_NAME ='Classification'
    print("Image ID: {},  Label: {}".format(inID, label))
    print("Prediction: {}".format(class_picked))
    #print(probabilities)
    cv2.startWindowThread()
    cv2.namedWindow(WINDOW_NAME)

    # display the predictions with the image
    cv2.putText(orig, "Predicted: {}".format(label), (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

    cv2.imshow(WINDOW_NAME, orig)
    cv2.waitKey(1000)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    cv2.waitKey(1)