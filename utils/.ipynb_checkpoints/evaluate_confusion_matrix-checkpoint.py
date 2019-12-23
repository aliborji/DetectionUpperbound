import os
import pickle
import itertools
import sys

import class_names


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
#     import pdb; pdb.set_trace()
#     cm_add = np.zeros((1,cm.shape[0]))
#     cm  = np.vstack((cm,cm_add))
    
    print(cm.shape)

    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f' if normalize else 'd'
#     fmt = 'd' #'.1f' if normalize else 'd'
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        u = j
        v = i + .2
        print(i)
        if i == cm.shape[0]-1:
            v = cm.shape[0] - 1
        
        if cm[i, j] > 0.01:
#             plt.text(u, v, format(int(cm[i, j]*100), fmt),
            plt.text(u, v, format(cm[i, j], fmt),                     
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=10)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    
#     import seaborn as sn
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     array = [[33,2,0,0,0,0,0,0,0,1,3], 
#             [3,31,0,0,0,0,0,0,0,0,0], 
#             [0,4,41,0,0,0,0,0,0,0,1], 
#             [0,1,0,30,0,6,0,0,0,0,1], 
#             [0,0,0,0,38,10,0,0,0,0,0], 
#             [0,0,0,3,1,39,0,0,0,0,4], 
#             [0,2,2,0,4,1,31,0,0,0,2],
#             [0,1,0,0,0,0,0,36,0,2,0], 
#             [0,0,0,0,0,0,1,5,37,5,1], 
#             [3,0,0,0,0,0,0,0,0,39,0], 
#             [0,0,0,0,0,0,0,0,0,0,38]]
#     df_cm = pd.DataFrame(array, index = [i for i in "ABCDEFGHIJK"],
#                       columns = [i for i in "ABCDEFGHIJK"])
#     plt.figure(figsize = (10,7))
#     sn.heatmap(df_cm, annot=True)    
    
    
    



# def plot_confusion_matrix(cm,
#                           target_names,
#                           title='Confusion matrix',
#                           cmap=None,
#                           normalize=True):
#     """
#     given a sklearn confusion matrix (cm), make a nice plot

#     Arguments
#     ---------
#     cm:           confusion matrix from sklearn.metrics.confusion_matrix

#     target_names: given classification classes such as [0, 1, 2]
#                   the class names, for example: ['high', 'medium', 'low']

#     title:        the text to display at the top of the matrix

#     cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
#                   see http://matplotlib.org/examples/color/colormaps_reference.html
#                   plt.get_cmap('jet') or plt.cm.Blues

#     normalize:    If False, plot the raw numbers
#                   If True, plot the proportions

#     Usage
#     -----
#     plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
#                                                               # sklearn.metrics.confusion_matrix
#                           normalize    = True,                # show proportions
#                           target_names = y_labels_vals,       # list of names of the classes
#                           title        = best_estimator_name) # title of graph

#     Citiation
#     ---------
#     http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

#     """
#     import matplotlib.pyplot as plt
#     import numpy as np
#     import itertools

#     accuracy = np.trace(cm) / float(np.sum(cm))
#     misclass = 1 - accuracy

#     if cmap is None:
#         cmap = plt.get_cmap('Blues')

#     plt.figure(figsize=(8, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()

#     if target_names is not None:
#         tick_marks = np.arange(len(target_names))
#         plt.xticks(tick_marks, target_names, rotation=45)
#         plt.yticks(tick_marks, target_names)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")


#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
#     plt.show()


    
    
    
    
if __name__ == "__main__":

    db, context = sys.argv[1], sys.argv[2]


    # import pdb; pdb.set_trace()
    with open(os.path.join('./output/' + db, context, 'predictions.pkl'), 'rb') as f:
        predictions = pickle.load(f)

    with open(os.path.join('./output/' + db, context, 'target.pkl'), 'rb') as f:
        targets = pickle.load(f)

#     import pdb; pdb.set_trace()
    preds = np.array(predictions[0])
    gts = np.array(targets)
    gts = gts[0,:]
#     filter_idx = np.where(gts != -1)[0]
#     preds = preds[filter_idx]
#     gts = gts[filter_idx]
    assert(preds.shape == gts.shape)

    accuracy = np.mean(preds == gts)
    print("Age detection accuracy is {}".format(accuracy))
    cnf_matrix = confusion_matrix(gts, preds)
    np.set_printoptions(precision=1)
    fig= plt.figure(figsize=(12,12))
#     classess= [str(i) for i in range(1,21)]
    plot_confusion_matrix(cnf_matrix, classes=class_names.labels[db], title='Confusion matrix, with normalization', normalize=True)
    plt.show()
    fig.savefig(db+'.png')


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     with open("predictions.pkl", "rb") as fid:
#         gender_preds, age_preds = pickle.load(fid)

#     with open("target.pkl", "rb") as fid:
#         gender_target, age_target = pickle.load(fid)


#     preds = np.array(age_preds)
#     gts = np.array(age_target)
#     filter_idx = np.where(gts != -1)[0]
#     preds = preds[filter_idx]
#     gts = gts[filter_idx]
#     assert(preds.shape == gts.shape)

#     accuracy = np.mean(preds == gts)
#     print("Age detection accuracy is {}".format(accuracy))
#     cnf_matrix = confusion_matrix(gts, preds)
#     np.set_printoptions(precision=4)
#     plt.figure()
#     plot_confusion_matrix(cnf_matrix, classes=["children", "adult", "toddlers"],
#                       title='Confusion matrix, with normalization', normalize=True)
#     plt.show()
