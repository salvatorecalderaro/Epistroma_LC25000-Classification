import numpy as np
import os
from PIL import Image
import tensorflow as tf
import tensorflow_addons as tfa
from keras.applications.resnet import ResNet152
from keras_balanced_batch_generator import make_generator
from datetime import timedelta
from time import perf_counter as pc
import pickle
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

img_size = 224
batch_size = 32
epochs = 20
lr = 1e-5
wd = 1e-4
m = 0.1
emb_size = 512
distance ='L2'


def create_train_test_set():
    train_path = "../dataset/train"
    test_path = "../dataset/test"

    train_paths=[]
    test_paths=[]

    x_train , y_train = [] , []
    x_test , y_test = [] , []
    print("Training and test set creation...")
    for path in sorted(os.listdir(train_path)):
        img_path = os.path.join(train_path, path)
        img = np.array(Image.open(img_path).resize((img_size,img_size)))
        x_train.append(img)
        y_train.append(int(path[0]))
        train_paths.append(img_path)
    
    for path in sorted(os.listdir(test_path)):
        img_path = os.path.join(test_path, path)
        img = np.array(Image.open(img_path).resize((img_size,img_size)))
        x_test.append(img)
        y_test.append(int(path[0]))
        test_paths.append(img_path)
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print("Done!")

    path="../paths/train_path"
    with open(path, "wb") as fp:   
        pickle.dump(train_paths, fp)
    fp.close()
    
    path="../paths/test_path"
    with open(path, "wb") as fp:   
        pickle.dump(test_paths, fp)
    fp.close()

    return x_train,y_train,x_test,y_test

def create_triplet_net():
    triplet_net = tf.keras.models.Sequential()
    resnet152=ResNet152(include_top=False,
                   input_shape=(224,224,3),
                   pooling='max',classes=None,
                   weights='imagenet')

    triplet_net.add(resnet152)
    triplet_net.add(tf.keras.layers.Flatten())
    triplet_net.add(tf.keras.layers.Dense(1024,activation ='relu'))
    triplet_net.add(tf.keras.layers.Dense(emb_size, activation=None)) 
    triplet_net.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    opt = tf.keras.optimizers.Adam(learning_rate = lr, decay = wd)
    loss_fn = tfa.losses.TripletSemiHardLoss(margin=m, distance_metric = distance)
    triplet_net.compile(optimizer=opt,loss=loss_fn)
    
    #tf.keras.utils.plot_model(triplet_net, to_file="../triplet_net.png",expand_nested=True, show_shapes=True,show_layer_activations=True)
    return triplet_net

def plot_loss(history):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label="Training loss")
    plt.title('Triplet net loss')
    plt.legend()
    plt.ylabel('triplet margin loss')
    plt.xlabel('epoch')
    plt.xticks(epochs)
    plt.tight_layout()
    plt.show()

def train_net(triplet_net,x_train,y_train):
    steps_per_epoch = int( np.ceil(x_train.shape[0] / batch_size) )
    y_cat = tf.keras.utils.to_categorical (y_train)
    train_gen = make_generator(x_train, y_cat, batch_size=batch_size,
               categorical=False,
               seed=None)
    print("Training.....")
    start = pc()
    history = triplet_net.fit(train_gen,epochs=epochs,steps_per_epoch=steps_per_epoch)
    end = pc()-start
    training_time = timedelta(seconds=end)
    print("Training ended in:", str(training_time))
    path = "../models/triplet_net.h5"
    triplet_net.save(path)
    print("Model saved !")
    plot_loss(history)
    return triplet_net

def save_embedding(triplet_net,x_train,y_train,x_test,y_test):
    print("Embeddings creation....")
    x_train_emb = triplet_net.predict(x_train,verbose=1)
    x_test_emb = triplet_net.predict(x_test,verbose=1)

    train_emb = "../embeddings/train_emb.npz"
    test_emb = "../embeddings/test_emb.npz"

    np.savez_compressed(train_emb,x_train_emb,y_train)
    np.savez_compressed(test_emb,x_test_emb,y_test)
    print("Embedding saved !")

def main():
    x_train,y_train,x_test,y_test = create_train_test_set()
    triplet_net = create_triplet_net()
    triplet_net = train_net(triplet_net,x_train,y_train)
    save_embedding(triplet_net,x_train,y_train,x_test,y_test)

if __name__ == '__main__':
    main()
