from timeit import default_timer as timer
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from util import calcMetrics

def train(model, criterion, optimizer, trainLoader, validLoader, save_file_name,
        max_epochs_stop=3, n_epochs=20, print_every=2):
    """Train a PyTorch Model
    Params
    --------
        model (PyTorch model): cnn to train
        criterion (PyTorch loss): objective to minimize
        optimizer (PyTorch optimizer): optimizer to compute gradients of model parameters
        trainLoader (PyTorch dataloader): training dataloader to iterate through
        validLoader (PyTorch dataloader): validation dataloader used for early stopping
        save_file_name (str ending in '.pt'): file path to save the model state dict
        max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
        n_epochs (int): maximum number of training epochs
        print_every (int): frequency of epochs to print training stats
    Returns
    --------
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    # try:
    #     print(f'Model has been trained for: {model.epochs} epochs.\n')
    # except:
    #     model.epochs = 0
    #     print(f'Starting Training from Scratch.\n')

    overall_start = timer()
    model.epochs = 0

    # Main loop
    for epoch in range(n_epochs):
        #print('Epoca = ', epoch)
        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0
        validation_acc = 0.0

        allValidationPredicted = []
        allValidationTarget = []
        allTrainingPredicted = []
        allTrainingTarget = []

        # Set to training
        model.train()
        start = timer()

        # Training loop
        for ii, (data, target) in enumerate(trainLoader):
            # Clear gradients
            optimizer.zero_grad()
            # Predicted outputs are log probabilities
            output = model(data)

            # Loss and backpropagation of gradients
            #loss = F.cross_entropy(outputs, labels)
            loss = criterion(output, target.long())
            loss.backward()

            # Update the parameters
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1)
            
            # correct_tensor = pred.eq(target.data.view_as(pred))
            # Need to convert correct tensor from int to float to average
            # accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
            # Multiply average accuracy times the number of examples in batch
            # print('accuracy', accuracy)
            # train_acc += accuracy.item() * data.size(0)
            
            # Neste cenario, 0 eh doente e 1 saudavel
            allTrainingPredicted = np.concatenate((allTrainingPredicted, pred.numpy()), axis=0)
            allTrainingTarget = np.concatenate((allTrainingTarget, target.numpy()), axis=0)
            
            # train_especificidade += calcEspecificidade(tp, fn) * data.size(0)
            # train_sensitividade += calcSensitividade(tn, fp) * data.size(0)
            # teste = calcAcc(tn, fp, fn, tp)
            # print('calcAcc', teste)
            # train_accTest += teste * data.size(0)
            # Track training progress
            # print(
            #     f'Epoch: {epoch}\t{100 * (ii + 1) / len(trainLoader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
            #     end='\r\n' )

        # After training loops ends, start validation
        else:
            model.epochs += 1

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()

                # Validation loop
                for data, target in validLoader:
                    # Forward pass
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target.long())
                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    # correct_tensor = pred.eq(target.data.view_as(pred))
                    # accuracy = torch.mean(
                    #     correct_tensor.type(torch.FloatTensor))
                    # # Multiply average accuracy times the number of examples
                    # valid_acc += accuracy.item() * data.size(0)
                    # Neste cenario, 0 eh doente e 1 saudavel
                    allValidationPredicted = np.concatenate((allValidationPredicted, pred.numpy()), axis=0)
                    allValidationTarget = np.concatenate((allValidationTarget, target.numpy()), axis=0)
            
                # Calculate average losses
                train_loss = train_loss / len(trainLoader.dataset)
                valid_loss = valid_loss / len(validLoader.dataset)

                # Calculate average accuracy
                #train_acc = train_acc / len(trainLoader.dataset)
                #valid_acc = valid_acc / len(validLoader.dataset)
                train_acc, train_especificidade, train_sensitividade, train_f1Score, cmTrain = calcMetrics(allTrainingTarget, allTrainingPredicted)
                validation_acc, validation_especificidade, validation_sensitividade, validation_f1Score, cmValidation = calcMetrics(allValidationTarget, allValidationPredicted)
                

                history.append([
                    train_acc, validation_acc,
                    train_especificidade, validation_especificidade,
                    train_sensitividade, validation_sensitividade,
                    train_f1Score, validation_f1Score,
                    train_loss, valid_loss ])

                # Print training and validation results
                if (epoch) % print_every == 0:
                    print(
                        f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \t\tValidation Loss: {valid_loss:.4f}'
                    )
                    print(
                        f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * validation_acc:.2f}%'
                    )
                    print(
                        f'\t\tTraining F1-Score: {train_f1Score:.2f}\t Validation F1-Score: {validation_f1Score:.2f} \n'
                    )

                # Save the model if validation loss decreases
                if valid_loss < valid_loss_min:
                    # Save model
                    torch.save(model.state_dict(), save_file_name)
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = validation_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1
                    # Trigger early stopping
                    if epochs_no_improve >= max_epochs_stop:
                        print(
                            f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * validation_acc:.2f}%'
                        )
                        total_time = timer() - overall_start
                        print(
                            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                        )

                        # Load the best state dict
                        model.load_state_dict(torch.load(save_file_name))
                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(
                            history,
                            columns=[
                                'train_acc', 'validation_acc', 
                                'train_especificidade', 'validation_especificidade',
                                'train_sensitividade', 'validation_sensitividade',
                                'train_f1Score', 'validation_f1Score',
                                'train_loss', 'valid_loss'
                            ])
                        return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * validation_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(history, columns=['train_acc', 'validation_acc', 
                                'train_especificidade', 'validation_especificidade',
                                'train_sensitividade', 'validation_sensitividade',
                                'train_f1Score', 'validation_f1Score',
                                'train_loss', 'valid_loss'])
    
    #print('Trained model', model)
    print('\nHistorico treinamento e teste \n', history)

    return model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation