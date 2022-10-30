from operator import imod
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score,recall_score,f1_score
from statistics import pstdev

def upload_train_test_set(f):
    train_path = "../embeddings/train/train_emb_f%d.npz" %(f)
    test_path = "../embeddings/test/test_emb_f%d.npz" %(f)
    mapping = {0:'colon_aca',1:'colon_n',2:'lung_aca',
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

def apply_knn(x_train,y_train,x_test):
    clf = KNeighborsClassifier(n_neighbors=2 , p=2)
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    return y_pred

def compute_class_metrics(y_test,y_pred,f):
    cr = classification_report(y_test, y_pred)
    print(cr)
    acc = accuracy_score(y_test,y_pred)
    cm = confusion_matrix(y_test,y_pred)
    prec=precision_score(y_test,y_pred,average='weighted')
    rec=recall_score(y_test,y_pred,average='weighted')
    f1=f1_score(y_test,y_pred,average='weighted')
    plot_cm(cm,f)
    return acc , prec,rec,f1,cm

def plot_cm(cm,f=None):
    if f != None:
        title = "LC25000 Dataset Classifier: KNN \n  Confusion Matrix - Fold %d" %(f)
        path="../plots/cm_f%d.png" %(f)
    else:
        title = "LC25000 Dataset Classifier: KNN \n Average Confusion Matrix"
        path="../plots/avg_cm.png"
    
    labels = ['colon_aca','colon_n','lung_aca','lung_n','lung_scc']
    
    plt.figure(figsize=(4,4))
    plt.suptitle(title)
    ax=sns.heatmap(np.array(cm), annot=True,fmt='g',cmap='Blues',cbar=False,annot_kws={"size": 14})
    ax.set_xticklabels(labels,rotation=45,fontsize=12)
    ax.set_yticklabels(labels,rotation=45,fontsize=12)
    plt.xlabel('Predicted label', fontsize=14)
    plt.ylabel('True label', fontsize=14)
    plt.tight_layout()
    plt.savefig(path)

def avg_cm(cms,nf):
    avg_cm = sum(cms)
    avg_cm = (avg_cm/nf)
    plot_cm(avg_cm,None)

def main():
    folds = [1,2,3,4,5]
    accuracies = []
    precisions = []
    recalls = []
    f1scores = []
    cms = []
    for f in folds:
        print("=============================================================================")
        print("FOLD %d" %(f))
        x_train , y_train , x_test , y_test = upload_train_test_set(f)
        y_pred = apply_knn(x_train,y_train,x_test)
        acc ,prec,rec,f1, cm = compute_class_metrics(y_test,y_pred,f)
        cms.append(cm)
        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1scores.append(f1)
        print("=============================================================================")
    
    avg_acc = sum(accuracies)/len(accuracies)
    sd = pstdev(accuracies)
    print("Mean Accuracy: %f SD: %f" %(avg_acc,sd))
    avg_precision = sum(precisions)/len(precisions)
    sd=pstdev(precisions)
    print("Mean Precision %f SD: %f" %(avg_precision,sd))

    avg_recall = sum(recalls)/len(recalls)
    sd=pstdev(recalls)
    print("Mean Recall %f SD: %f" %(avg_recall,sd))

    avg_f1 = sum(f1scores)/len(f1scores)
    sd=pstdev(f1scores)
    print("Mean f1-score %f SD: %f" %(avg_f1,sd))
    avg_cm(cms,len(folds))

if __name__ == '__main__':
    main()
