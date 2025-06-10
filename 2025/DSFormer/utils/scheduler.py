import torch.optim as optim


def load_scheduler(model_name, model):
    optimizer, scheduler = None, None
    
    if model_name == 'cnn2d':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [100, 200], gamma=0.1)

    elif model_name == 'DSFormer':
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.00001)
        scheduler = None

    return optimizer, scheduler


