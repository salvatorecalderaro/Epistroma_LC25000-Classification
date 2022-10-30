import numpy as np
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def upload_train_test_set():
    train_path = "../embeddings/train_emb.npz"
    test_path = "../embeddings/test_emb.npz"

    mapping={0: "Epithelium",1:'Stroma'}
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

def plot_embedding(x_train,y_train,x_test,y_test):
    np.random.seed(123)
    embedder = umap.UMAP()
    emb_train = embedder.fit_transform(x_train)
    emb_test = embedder.transform(x_test)

    plt.figure()
    plt.suptitle("Embedding - Triplet net \n Epistroma Dataset" ,fontsize = 14)
    plt.subplot(211)
    plt.title("Training set")
    sns.scatterplot(x=emb_train[:,0],y=emb_train[:,1],hue=y_train)
    plt.legend()
    plt.subplot(212)
    plt.title("Test set")
    sns.scatterplot(x=emb_test[:,0],y=emb_test[:,1],hue=y_test)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    x_train,y_train,x_test,y_test = upload_train_test_set()
    plot_embedding(x_train,y_train,x_test,y_test)

if __name__ == '__main__':
    main()
    