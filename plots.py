import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import numpy as np
import pandas as pd
from numpy import savetxt

def plotComparative(history, item1, item2, saveName, xlabel, ylabel, title, labels):
    fig = plt.figure(figsize=(8, 2))
    for c in [item1, item2]:
        plt.plot(
            history[c], label=labels[c])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.savefig(saveName + '.png')

def plotLosses(history, model):
    xlabel = 'Época'
    ylabel = 'Negative Log Likelihood'
    title = 'Losses treinamento e validação'

    saveName = 'plotLosses_' + model
    labels = {
        'train_loss': 'Loss treinamento', 
        'valid_loss': 'Loss validacao'
    }
    plotComparative(history, 'train_loss', 'valid_loss', saveName, xlabel, ylabel, title, labels)

def plotAcc(history, model):
    xlabel = 'Época'
    ylabel = 'Acurácia'
    title = 'Acurácia treinamento e validação'

    saveName = 'plotAcc_'+model
    labels = {
        'train_acc': 'Acurácia treinamento', 
        'validation_acc': 'Acurácia validação'
    }
    plotComparative(history, 'train_acc', 'validation_acc', saveName, xlabel, ylabel, title, labels)

def plotSensitividade(history, model):
    xlabel = 'Época'
    ylabel = 'Sensitividade'
    title = 'Sensitividade treinamento e validação'

    saveName = 'plotSensitividade_'+model
    labels = {
        'train_sensitividade': 'Sensitividade treinamento',
        'validation_sensitividade': 'Sensitividade validação'
    }
    plotComparative(history, 'train_sensitividade', 'validation_sensitividade', saveName, xlabel, ylabel, title,labels)

def plotEspecificidade(history, model):
    xlabel = 'Época'
    ylabel = 'Especificidade'
    title = 'Especificidade treinamento e validação'

    saveName = 'plotEspecificidade_' + model
    labels = {
        'train_especificidade': 'Especificidade treinamento',
        'validation_especificidade': 'Especificidade validação'
    }
    plotComparative(history, 'train_especificidade', 'validation_especificidade', saveName, xlabel, ylabel, title, labels)

def plotF1Score(history, model):
    xlabel = 'Época'
    ylabel = 'F1Score'
    title = 'F1Score treinamento e validação'
    saveName = 'plotF1Score_' + model
    labels = {
        'train_f1Score': 'F1-Score treinamento',
        'validation_f1Score': 'F1-Score validação'
    }
    plotComparative(history, 'train_f1Score', 'validation_f1Score', saveName, xlabel, ylabel, title, labels)   

def plotData(history, model):
    plotAcc(history, model)
    plotSensitividade(history, model)
    plotEspecificidade(history, model)
    plotF1Score(history, model)
    plotLosses(history, model)

def plotTestingAcc(results, model):
    # Plot using seaborn
    fig = sns.lmplot(y='acuracia', x='Treinamento', data=results, height=6)
    plt.xlabel('images')
    plt.ylabel('Accuracy (%)')
    plt.title('Top 1 Accuracy vs Number of Training Images')
    plt.ylim(-5, 105)
    fig.savefig('plotTestingAcc_'+model+'.png')

def plotTransformedImages(images, i, typeImg):
    inputs = images[0]
    teste = inputs.numpy()
    print('inputs.numpy() = ',teste )
    print('teste = ',teste.shape)
    inputs = inputs.permute(1, 2, 0)
    fig = plt.figure()
    plt.imshow(inputs.numpy())
    fig.savefig(typeImg + '_transformeImg_' + str(i) +'.png')
    teste = teste.flatten()
    savetxt(typeImg+'data_' + str(i) +'_.csv', teste, delimiter=',')
    plt.close()
    
def plotConfusionMatrix(cm):
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(['Doente', 'Saudavel'])
    ax.yaxis.set_ticklabels(['Doente', 'Saudavel'])
    plt.show()

def prepareAllDF(test, trainValidation):
    print('trainValidation', trainValidation)
    print('test', test)
    all = trainValidation.copy()
    print('all 1', all)
    all = all.assign(test=test.acuracia)
    print('all 2', all)
    all = all.assign(test=test.loss)
    print('all 3', all)

def plotAll(test, trainValidation):

    fig = plt.figure(figsize=(8, 2))
    for c in ['train_acc', 'valid_acc']:
        plt.plot(100 * trainValidation[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    fig.savefig('plotAcc.png')


# from https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot
    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph
    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    fig.savefig('confusion.png')