import torch
import numpy as np
from PIL import  Image
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle

def upload_paths(f):
    path="../paths/train_pathf%d" %(f)
    with open(path, 'rb') as fp:
        train_paths = pickle.load(fp)
    
    path="../paths/test_pathf%d" %(f)
    with open(path, "rb") as fp:   
        test_paths = pickle.load(fp)    
    return train_paths,test_paths
    

def upload_embeddings(f):
    train_path = "../embeddings/train/train_emb_f%d.npz" %(f)
    test_path = "../embeddings/test/test_emb_f%d.npz" %(f)
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    x_train = train_data['arr_0']
    y_train = train_data['arr_1']
    x_test = test_data['arr_0']
    y_test = test_data['arr_1']
    y_train = np.reshape(y_train,-1)
    y_test = np.reshape(y_test,-1)
    return x_train,y_train,x_test,y_test

def apply_knn(train_paths, test_paths,x_train,y_train,x_test,y_test,f):
    mapping = {0:'colon_aca',1:'colon_n',2:'lung_aca',
3:'lung_n',4:'lung_scc'}
    clf = KNeighborsClassifier(n_neighbors=2,p=2)
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    nn = clf.kneighbors(x_test)
    i = 0
    for test_image , dist,index,y,p in zip(test_paths,nn[0],nn[1],y_test,pred):
        if y !=p:
            path = "../results knn/%d/ti_%d_m.png" %(f,i)
            fig, axs = plt.subplots(1, 3)
            fig.set_size_inches(18.5, 10.5)
            im =np.array(Image.open(test_image).resize((224,224)))
            
            axs[0].imshow(im)
            axs[0].set_title("Test Image \n Prediction: %s \n True label: %s" %(mapping[p],mapping[y]))
            axs[0].axis('off')

            p1=train_paths[index[0]]
            axs[1].imshow(np.array(Image.open(p1).resize((224,224))))
            axs[1].set_title("Training image label:\n %s \n Distance: %f" %(mapping[y_train[index[0]]],dist[0]),color='r')
            axs[1].axis('off')

            p2=train_paths[index[1]]
            axs[2].imshow(np.array(Image.open(p2).resize((224,224))))
            axs[2].set_title("Training image label:\n %s \n Distance: %f" %(mapping[y_train[index[1]]],dist[1]),color='r')
            axs[2].axis('off')

            print(path)
            fig.savefig(path, bbox_inches='tight')
            i+=1
            plt.cla()
        else:
            path = "../results knn/%d/ti_%d.png" %(f,i)
        
        
        


def main():
    folds=[1,2,3,4,5]
    for f in folds:
        train_paths,test_paths=upload_paths(f)
        x_train,y_train,x_test,y_test = upload_embeddings(f)
        apply_knn(train_paths,test_paths,x_train,y_train,x_test,y_test,f)
        break

if __name__ == '__main__':
    main()