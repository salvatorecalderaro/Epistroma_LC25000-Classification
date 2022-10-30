import numpy as np
import umap 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import pairwise_distances

def upload_train_test_set(f):
    train_path = "../embeddings/train/train_emb_f%d.npz" %(f)
    test_path = "../embeddings/test/test_emb_f%d.npz" %(f)
    mapping={0:'colon_aca',1:'colon_n',2:'lung_aca',
3:'lung_n',4:'lung_scc'}
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    x_train = train_data['arr_0']
    y_train = train_data['arr_1']
    x_test = test_data['arr_0']
    y_test = test_data['arr_1']
    y_train = np.reshape(y_train,-1)
    y_test = np.reshape(y_test,-1)
    aus = y_train.tolist()
    y_train = list(map(mapping.get, aus))
    y_train = np.asarray(y_train)
    aus = y_test.tolist()
    y_test = list(map(mapping.get, aus))
    y_test = np.asarray(y_test)
    return x_train, y_train, x_test, y_test

def plot_embedding(x_train,y_train,x_test,y_test,f):
    np.random.seed(123)
    embedder = umap.UMAP()
    emb_train = embedder.fit_transform(x_train)
    emb_test = embedder.transform(x_test)

    path="../plots/emb_f%d.png" %(f)

    hue_order = ['colon_aca','colon_n','lung_aca','lung_n','lung_scc']
    
    plt.figure()
    plt.suptitle("Embedding - Triplet net \n LC25000  Dataset Fold %d" %(f) ,fontsize = 14)
    plt.subplot(211)
    plt.title("Training set")
    sns.scatterplot(x=emb_train[:,0],y=emb_train[:,1],hue=y_train, hue_order=hue_order)
    plt.legend()
    plt.subplot(212)
    plt.title("Test set")
    sns.scatterplot(x=emb_test[:,0],y=emb_test[:,1],hue=y_test,hue_order=hue_order)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)

def plot_distance_matrix(x_train,x_test,f):
    train_dist = pairwise_distances(x_train,metric='euclidean')
    plt.figure()
    plt.title("LC 25000 Dataset \n Train set distance matrix")
    sns.heatmap(train_dist,xticklabels=False,yticklabels=False)
    path = "../plots/train_dist_f%s.png" %(f)
    plt.savefig(path,dpi=100)

    test_dist = pairwise_distances(x_test,metric='euclidean')
    plt.figure()
    plt.title("LC 25000 Dataset \n Test set distance matrix")
    sns.heatmap(test_dist,xticklabels=False,yticklabels=False)
    path = "../plots/test_dist_f%s.png" %(f)
    plt.savefig(path,dpi=100)

def main():
    folds = [1,2,3,4,5]
    for f in folds:
        x_train , y_train , x_test , y_test = upload_train_test_set(f)
        plot_embedding(x_train , y_train , x_test , y_test,f)
        #plot_distance_matrix(x_train,x_test,f)

if __name__ == '__main__':
    main()