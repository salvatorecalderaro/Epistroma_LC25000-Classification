import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def upload_train_test_set():
    train_path = "../embeddings/train_emb.npz"
    test_path = "../embeddings/test_emb.npz"
    train_data = np.load(train_path)
    test_data = np.load(test_path)
    x_train = train_data['arr_0']
    y_train = train_data['arr_1']
    x_test = test_data['arr_0']
    y_test = test_data['arr_1']

    return x_train,y_train,x_test,y_test

def find_optimal_k(x_train,y_train,x_test,y_test):
    k_values = list(range(3,22,2))
    accuracies = []

    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k,p=2)
        clf.fit(x_train,y_train)
        y_pred = clf.predict(x_test)
        accuracies.append(accuracy_score(y_test,y_pred))
    
    max_acc = max(accuracies)
    index = accuracies.index(max_acc)

    optimal_k = k_values[index]
    
    plt.figure()
    plt.suptitle("Epistroma Dataset - Classification \n  KNN: k vs   Accuracy",fontsize=16)
    plt.plot(k_values,accuracies,marker = 'o')
    plt.xticks(k_values)
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.show()
    

    return optimal_k,max_acc
    


def main():
    x_train,y_train,x_test,y_test = upload_train_test_set()
    optimal_k , max_acc = find_optimal_k(x_train,y_train,x_test,y_test)
    print("The optimal value of k is %d Accuracy: %f" %(optimal_k,max_acc))

if __name__ == '__main__':
    main()
     
