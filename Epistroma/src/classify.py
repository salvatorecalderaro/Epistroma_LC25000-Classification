import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix, plot_confusion_matrix
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

    return x_train, y_train,x_test,y_test

def knn(x_train,y_train,x_test):
    clf = KNeighborsClassifier(n_neighbors=9,p=2)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    return y_pred

def create_confusion_matrix(y_test,y_pred):
    cm = confusion_matrix(y_test,y_pred)
    
    labels = ["Epithelium","Stroma"]
    plt.figure(figsize=(4,4))
    #plt.suptitle("Epistroma Dataset Classification \n Confusion Matrix")
    ax=sns.heatmap(np.array(cm), annot=True,fmt='g',cmap='Blues',cbar=False,annot_kws={"size": 14})
    ax.set_xticklabels(labels,rotation=45,fontsize=12)
    ax.set_yticklabels(labels,rotation=45,fontsize=12)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.tight_layout()
    plt.show()


def compute_metrics(y_test,y_pred):
    cr = classification_report(y_test,y_pred)
    print(cr)
    create_confusion_matrix(y_test,y_pred)

def main():
    x_train , y_train, x_test ,y_test = upload_train_test_set()
    y_pred = knn(x_train,y_train,x_test)
    compute_metrics(y_test,y_pred)

if __name__ == '__main__':
    main()
