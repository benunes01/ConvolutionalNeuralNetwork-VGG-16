from torchvision import transforms, datasets, models
import torch.nn as nn
from torch import optim

def prepareModelPNGVGG(dataset, n_classes):

    model = models.vgg16(pretrained=True)
    #print('model', model)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.classifier[6].in_features

    # Add on classifier
    model.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

    total_params = sum(p.numel() for p in model.parameters())
    #print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'{total_trainable_params:,} training parameters.')

    #summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')
    #print(model.classifier[6])
    print('dataset.class_to_idx', dataset.class_to_idx)
    model.class_to_idx = dataset.class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }
    print('model.idx_to_class ', model.idx_to_class )
    #print('model.idx_to_class', model.idx_to_class)
    return model

def prepareVGG16ModelWithTXT(n_classes):

    model = models.vgg16(pretrained=True)
    #print('model', model)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.classifier[6].in_features

    # Add on classifier
    model.classifier[6] = getFullyConnectedStructure(n_inputs, n_classes)

    total_params = sum(p.numel() for p in model.parameters())
    #print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'{total_trainable_params:,} training parameters.')
    #print('model', model)
    #summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')
    #print(model.classifier[6])

    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}
    #print('model.idx_to_class', model.idx_to_class)
    return model


def prepareResnetModelWithTXT(n_classes):
    model = models.resnet50(pretrained=True)
    #print('model', model)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features

    # Add on classifier
    model.fc = getFullyConnectedStructure(n_inputs, n_classes)

    total_params = sum(p.numel() for p in model.parameters())
    #print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'{total_trainable_params:,} training parameters.')

    #summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')
    #print(model.classifier[6])
    #print('model', model)
    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}
    #print('model.idx_to_class', model.idx_to_class)
    return model


def prepareDensenetModelWithTXT(n_classes):
    model = models.densenet201(pretrained=True)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    n_inputs = model.classifier.in_features

    # Add on classifier
    #model.classifier = nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    model.classifier = getFullyConnectedStructure(n_inputs, n_classes)

    total_params = sum(p.numel() for p in model.parameters())
    #print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'{total_trainable_params:,} training parameters.')

    #summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')
    #print(model.classifier[6])
    #print('model', model)
    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}
    #print('model.idx_to_class', model.idx_to_class)
    return model

def getFullyConnectedStructure(n_inputs, n_classes):
    #nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    #lastLayer = nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(), nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    lastLayer = nn.Sequential(nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))
    #lastLayer = nn.Sequential(nn.Linear(n_inputs, 512), nn.ReLU(), nn.Linear(512, n_classes), nn.LogSoftmax(dim=1))
    print('lastLayer', lastLayer)
    return lastLayer
    #return nn.Sequential(nn.Linear(n_inputs, n_classes), nn.ReLU())

def prepareTrainingLoss():
    criterion = nn.NLLLoss()
    return criterion

def prepareTrainingOptimizer(model):

    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    #for p in optimizer.param_groups[0]['params']:
    #    if p.requires_grad:
    #        print(p.shape)
    return optimizer