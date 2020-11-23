import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          decimals=2):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=decimals)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:,}".format(cm[i, j]), horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_label_distribution(labels, classes=['No Diabetis', 'Diabetis']):
    """
    This function plots the distribution of the given labels
    """
    label_count = [int(list(labels).count(x)) for x in range(len(classes))]
    y_pos = np.arange(len(classes))

    plt.bar(y_pos, label_count, align='center')
    plt.xticks(y_pos, classes)
    plt.ylabel('Occurences')
    plt.title('Label')
    plt.show()

if __name__ == '__main__':
    y_train = [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,]

    plot_label_distribution(y_train)
