import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def upload_train_test_set(f):
    train_path = "../embeddings/train/train_emb_f%d.npz" %(f)
    test_path = "../embeddings/test/test_emb_f%d.npz" %(f)
    data = np.load(train_path)
    x_train = data['arr_0']
    y_train = data['arr_1']

    data = np.load(test_path)
    x_test = data['arr_0']
    y_test = data['arr_1']

    return x_train , y_train , x_test , y_test


def apply_knn(x_train,y_train,x_test,y_test):
    k_values = list(range(1,21,1))
    acc = []
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k , p=2)
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        acc.append(accuracy_score(y_test,y_pred))
    
    return acc

def find_optimal_k(accuracies):
    accuracies = np.array(accuracies)
    k_values = list(range(1,21,1))
    avg_acc = np.mean(accuracies , axis=0)
    index = np.argmax(avg_acc)
    k = k_values[index]
    print("Optimal value of k %d - Avg. Accuracy %f " %(k,avg_acc[index]))

    plt.figure()
    #plt.suptitle("LC25000 Dataset \n KNN - k vs avg. Accuracy")
    plt.plot(k_values,avg_acc,marker="o")
    plt.xticks(k_values)
    plt.xlabel("k")
    plt.ylabel("avg. Accuracy")
    plt.tight_layout()
    plt.show()


    
def main():
    accuracies = []
    folds = [1,2,3,4,5]
    for f in folds:
        x_train , y_train , x_test , y_test = upload_train_test_set(f)
        acc = apply_knn(x_train , y_train , x_test , y_test)
        accuracies.append(acc)
    
    find_optimal_k(accuracies)


if __name__ == '__main__':
    main()
