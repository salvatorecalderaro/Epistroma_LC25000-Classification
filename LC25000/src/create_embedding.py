import os
import numpy as np 
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
import tensorflow_addons as tfa
from keras.applications.resnet import ResNet152
from keras_balanced_batch_generator import make_generator
from datetime import timedelta
from time import perf_counter as pc
import matplotlib.pyplot as plt
import pickle

img_size = 224
nfolds = 5
batch_size = 32
epochs = 20
lr = 1e-5
wd = 1e-4
m = 0.1
emb_size = 512
distance = 'L2'


def upload_data():
    c_aca_path = "../../dataset/lung_colon_image_set/colon_image_sets/colon_aca"
    c_b_path = "../../dataset/lung_colon_image_set/colon_image_sets/colon_n"
    l_aca_path = "../../dataset/lung_colon_image_set/lung_image_sets/lung_aca"
    l_b_path = "../../dataset/lung_colon_image_set/lung_image_sets/lung_n"
    l_scc_path = "../../dataset/lung_colon_image_set/lung_image_sets/lung_scc"
    data , labels , paths = [] , [] , []

    print("Data upload...")
    for path in os.listdir(c_aca_path):
        img_path = os.path.join(c_aca_path,path)
        img = np.array(Image.open(img_path).resize((img_size,img_size)))
        data.append(img)
        labels.append(0)
        paths.append(img_path)
    
    for path in os.listdir(c_b_path):
        img_path = os.path.join(c_b_path,path)
        img = np.array(Image.open(img_path).resize((img_size,img_size)))
        data.append(img)
        labels.append(1)
        paths.append(img_path)


    for path in os.listdir(l_aca_path):
        img_path = os.path.join(l_aca_path,path)
        img = np.array(Image.open(img_path).resize((img_size,img_size)))
        data.append(img)
        labels.append(2)
        paths.append(img_path)
    
    for path in os.listdir(l_b_path):
        img_path = os.path.join(l_b_path,path)
        img = np.array(Image.open(img_path).resize((img_size,img_size)))
        data.append(img)
        labels.append(3)
        paths.append(img_path)
    
    for path in os.listdir(l_scc_path):
        img_path = os.path.join(l_scc_path,path)
        img = np.array(Image.open(img_path).resize((img_size,img_size)))
        data.append(img)
        labels.append(4)
        paths.append(img_path)

    print("Done!")
    data = np.array(data)
    labels = np.array(labels)

    return data , labels,paths

def create_triplet_net():
    triplet_net = tf.keras.models.Sequential()
    resnet152=ResNet152(include_top=False,
                   input_shape=(224,224,3),
                   pooling='max',classes=None,
                   weights='imagenet')

    triplet_net.add(resnet152)
    triplet_net.add(tf.keras.layers.Flatten())
    triplet_net.add(tf.keras.layers.Dense(emb_size, activation=None)) 
    triplet_net.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

    opt = tf.keras.optimizers.Adam(learning_rate = lr, decay = wd)
    loss_fn = tfa.losses.TripletSemiHardLoss(margin=m, distance_metric = distance)
    triplet_net.compile(optimizer=opt,loss=loss_fn)
    
    tf.keras.utils.plot_model(triplet_net, to_file="../triplet_net.png",expand_nested=True, show_shapes=True,show_layer_activations=True)
    return triplet_net

def plot_loss(history , f):
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'b', label="Training loss")
    plt.title('Triplet net loss - Fold %d' %(f))
    plt.legend()
    plt.ylabel('triplet margin loss')
    plt.xlabel('epoch')
    plt.xticks(epochs)
    plt.tight_layout()
    plt.show()



def train_net(triplet_net,x_train,y_train,f):
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
    path = "../models/triplet_net_%d.h5" %(f)
    triplet_net.save(path)
    print("Model saved !")
    #plot_loss(history,f)
    return triplet_net

def save_embedding(triplet_net,x_train,y_train,x_test,y_test,f):
    print("Embeddings creation....")
    x_train_emb = triplet_net.predict(x_train,verbose=1)
    x_test_emb = triplet_net.predict(x_test,verbose=1)

    train_emb = "../embeddings/train/train_emb_f%d.npz" %(f)
    test_emb = "../embeddings/test/test_emb_f%d.npz" %(f)

    np.savez_compressed(train_emb,x_train_emb,y_train)
    np.savez_compressed(test_emb,x_test_emb,y_test)
    print("Embedding saved !")

def save_list(train_paths,test_paths,f):
    path="../paths/train_pathf%d" %(f)
    with open(path, "wb") as fp:   
        pickle.dump(train_paths, fp)
    fp.close()
    
    path="../paths/test_pathf%d" %(f)
    with open(path, "wb") as fp:   
        pickle.dump(test_paths, fp)
    fp.close()

def apply_k_fold(data , labels,paths):
    f = 1
    skf = StratifiedKFold(nfolds,random_state=2022,shuffle=True)
    for train_index , test_index in skf.split(data , labels):
        print("==================================================================================================")
        print("FOLD %d" %(f))
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        train_paths = [paths[i] for i in train_index]
        test_paths = [paths[i] for i in test_index]
        save_list(train_paths,test_paths,f)
        #triplet_net = create_triplet_net()
        #triplet_net = train_net(triplet_net,x_train,y_train,f)
        #save_embedding(triplet_net,x_train,y_train,x_test,y_test,f)
        f += 1
        print("==================================================================================================")


def main():
    data , labels,paths= upload_data()
    apply_k_fold(data,labels,paths)

if __name__ == '__main__':
    main()
