import numpy as np
from PIL import  Image
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle

def upload_paths():
    path="../paths/train_path"
    with open(path, 'rb') as fp:
        train_paths = pickle.load(fp)
    
    path="../paths/test_path"
    with open(path, "rb") as fp:   
        test_paths = pickle.load(fp)    
    return train_paths,test_paths
    

def upload_embeddings():
    train_path = "../embeddings/train_emb.npz"
    test_path = "../embeddings/test_emb.npz"
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    x_train = train_data['arr_0']
    y_train = train_data['arr_1']
    x_test = test_data['arr_0']
    y_test = test_data['arr_1']
    y_train = np.reshape(y_train,-1)
    y_test = np.reshape(y_test,-1)
    return x_train,y_train,x_test,y_test

def apply_knn(train_paths, test_paths,x_train,y_train,x_test,y_test):
    mapping={0: "Epithelium",1:'Stroma'}
    clf = KNeighborsClassifier(n_neighbors=9,p=2)
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    nn = clf.kneighbors(x_test)
    i = 0
    for test_image , dist,index,y,p in zip(test_paths,nn[0],nn[1],y_test,pred):
        print(index)
        if y !=p:
            path = "../results knn/ti_%d_m.png" %(i)
        else:
            path = "../results knn/ti_%d.png" %(i)
        
        fig, axs = plt.subplots(2, 5)
        fig.set_size_inches(18.5, 10.5)
        im =np.array(Image.open(test_image).resize((224,224)))
            
        axs[0][0].imshow(im)
        axs[0][0].set_title("Test Image \n Prediction: %s \n True label: %s" %(mapping[p],mapping[y]))
        axs[0][0].axis('off')

        p1=train_paths[index[0]]
        axs[0][1].imshow(np.array(Image.open(p1).resize((224,224))))
        axs[0][1].set_title("Training image label:\n %s \n Distance: %f" %(mapping[y_train[index[0]]],dist[0]),color='r')
        axs[0][1].axis('off')

        p2=train_paths[index[1]]
        axs[0][2].imshow(np.array(Image.open(p2).resize((224,224))))
        axs[0][2].set_title("Training image label:\n %s \n Distance: %f" %(mapping[y_train[index[1]]],dist[1]),color='r')
        axs[0][2].axis('off')

        p3=train_paths[index[2]]
        axs[0][3].imshow(np.array(Image.open(p3).resize((224,224))))
        axs[0][3].set_title("Training image label:\n %s \n Distance: %f" %(mapping[y_train[index[2]]],dist[2]),color='r')
        axs[0][3].axis('off')

        p4=train_paths[index[3]]
        axs[0][4].imshow(np.array(Image.open(p4).resize((224,224))))
        axs[0][4].set_title("Training image label:\n %s \n Distance: %f" %(mapping[y_train[index[3]]],dist[3]),color='r')
        axs[0][4].axis('off')
    
        p5=train_paths[index[4]]
        axs[1][0].imshow(np.array(Image.open(p5).resize((224,224))))
        axs[1][0].set_title("Training image label:\n %s \n Distance: %f" %(mapping[y_train[index[4]]],dist[4]),color='r')
        axs[1][0].axis('off')

        p6=train_paths[index[5]]
        axs[1][1].imshow(np.array(Image.open(p6).resize((224,224))))
        axs[1][1].set_title("Training image label:\n %s \n Distance: %f" %(mapping[y_train[index[5]]],dist[5]),color='r')
        axs[1][1].axis('off')
        
        p7=train_paths[index[6]]
        axs[1][2].imshow(np.array(Image.open(p7).resize((224,224))))
        axs[1][2].set_title("Training image label:\n %s \n Distance: %f" %(mapping[y_train[index[6]]],dist[6]),color='r')
        axs[1][2].axis('off')

        p8=train_paths[index[7]]
        axs[1][3].imshow(np.array(Image.open(p8).resize((224,224))))
        axs[1][3].set_title("Training image label:\n %s \n Distance: %f" %(mapping[y_train[index[7]]],dist[7]),color='r')
        axs[1][3].axis('off')

        p9=train_paths[index[8]]
        axs[1][4].imshow(np.array(Image.open(p9).resize((224,224))))
        axs[1][4].set_title("Training image label:\n %s \n Distance: %f" %(mapping[y_train[index[8]]],dist[8]),color='r')
        axs[1][4].axis('off')

        fig.savefig(path, bbox_inches='tight')
        i+=1
        
        
        


def main():
    train_paths,test_paths=upload_paths()
    x_train,y_train,x_test,y_test = upload_embeddings()
    apply_knn(train_paths,test_paths,x_train,y_train,x_test,y_test)

if __name__ == '__main__':
    main()