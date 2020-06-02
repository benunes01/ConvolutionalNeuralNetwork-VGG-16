import numpy as np
import pandas as pd
import torch
from util import calcMetrics

def accuracy(output, target):
    """Compute the topk accuracy(s)"""

    with torch.no_grad():
        batch_size = target.size(0)
        # Find the predicted classes and transpose
        _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
    
        pred = pred.t()
        
        # Determine predictions equal to the targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # Find the percentage of correct
        correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
        res = (correct_k.mul_(100.0 / batch_size).item())
        return res, pred


def evaluate(model, test_loader, criterion, n_classes):
    """Measure the performance of a trained PyTorch model
    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
        topk (tuple of ints): accuracy to measure
    Returns
    --------
        results (DataFrame): results for each category
    """

    classes = []
    losses = []
    # Hold accuracy results
    acc_results = np.zeros(len(test_loader.dataset))
    i = 0
    test_error_count = 0.0
    model.eval()
    allTestingTarget = []
    allTestingPredicted = []
    with torch.no_grad():

        # Testing loop
        for data, targets in test_loader:
            
            # Raw model output
            out = model(data)
            test_error_count += float(torch.sum(torch.abs(targets - out.argmax(1))))
            #print('out = ', out)
            # Iterate through each example
            for pred, target in zip(out, targets):
                # Find topk (top 1) accuracy

                acc_results[i], predictedClass = accuracy(pred.unsqueeze(0), target.unsqueeze(0))
                classes.append(model.idx_to_class[target.item()])
                # Calculate the loss
                loss = criterion(pred.view(1, n_classes), target.view(1))
                losses.append(loss.item())

                predictedClass = predictedClass.numpy()[0]
                target = [target.numpy()]
                allTestingPredicted = np.concatenate((allTestingPredicted, predictedClass), axis=0)
                allTestingTarget = np.concatenate((allTestingTarget, target), axis=0)

                i += 1
        #train_acc, train_especificidade, train_sensitividade, train_f1Score, cmTrain = calcMetrics(allTrainingTarget, allTrainingPredicted)
    
    test_acc, test_especificidade, test_sensitividade, test_f1Score, cmTest = calcMetrics(allTestingTarget, allTestingPredicted)
    history = pd.DataFrame({
        'test_acc': [test_acc], 'test_especificidade': [test_especificidade],
        'test_sensitividade': [test_sensitividade], 'test_f1Score': [test_f1Score]})
    print('\nTesting result\n', history)
    # Send results to a dataframe and calculate average across classes
    results = pd.DataFrame({'acuracia': acc_results, 'class':classes, 'loss': losses})
    results = results.groupby(classes).mean()
    #print('Results', results)
    return results.reset_index().rename(columns={'index': 'class'}), test_error_count, history, cmTest