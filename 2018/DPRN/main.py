import argparse
import auxil
from hyper_pytorch import *
import torch
import torch.nn.parallel
from torchvision.transforms import *
import models.resnet as RN
import models.presnet as PYRM


def load_hyper(args):
    data, labels, numclass = auxil.loadData(args.dataset, num_components=args.components)
    pixels, labels = auxil.createImageCubes(data, labels, windowSize=args.spatialsize, removeZeroLabels = True)
    bands = pixels.shape[-1]; numberofclass = len(np.unique(labels))
    if args.tr_percent < 1: # split by percent
        x_train, x_test, y_train, y_test = auxil.split_data(pixels, labels, args.tr_percent)
    else: # split by samples per class
        x_train, x_test, y_train, y_test = auxil.split_data_fix(pixels, labels, args.tr_percent)
    if args.use_val:
        x_val, x_test, y_val, y_test = auxil.split_data(x_test, y_test, args.val_percent)
    del pixels, labels
    train_hyper = HyperData((np.transpose(x_train, (0, 3, 1, 2)).astype("float32"),y_train))
    test_hyper  = HyperData((np.transpose(x_test, (0, 3, 1, 2)).astype("float32"),y_test))
    if args.use_val:
        val_hyper = HyperData((np.transpose(x_val, (0, 3, 1, 2)).astype("float32"),y_val))
    else:
        val_hyper = None
    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(train_hyper, batch_size=args.tr_bsize, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(test_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    val_loader  = torch.utils.data.DataLoader(val_hyper, batch_size=args.te_bsize, shuffle=False, **kwargs)
    return train_loader, test_loader, val_loader, numberofclass, bands


def train(trainloader, model, criterion, optimizer, epoch, use_cuda):
    model.train()
    accs   = np.ones((len(trainloader))) * -1000.0
    losses = np.ones((len(trainloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # print('inputs=', inputs.size())

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses[batch_idx] = loss.item()
        accs[batch_idx] = auxil.accuracy(outputs.data, targets.data)[0].item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (np.average(losses), np.average(accs))


def test(testloader, model, criterion, epoch, use_cuda):
    model.eval()
    accs   = np.ones((len(testloader))) * -1000.0
    losses = np.ones((len(testloader))) * -1000.0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        outputs = model(inputs)
        losses[batch_idx] = criterion(outputs, targets).item()
        accs[batch_idx] = auxil.accuracy(outputs.data, targets.data, topk=(1,))[0].item()
    return (np.average(losses), np.average(accs))


def predict(testloader, model, criterion, use_cuda):
    model.eval()
    predicted = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda: inputs = inputs.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        [predicted.append(a) for a in model(inputs).data.cpu().numpy()] 
    return np.array(predicted)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups: 
        param_group['lr'] = lr

def main():
    parser = argparse.ArgumentParser(description='PyTorch DCNNs Training')
    parser.add_argument('--epochs', default=300, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--components', default=None, type=int, help='dimensionality reduction')
    parser.add_argument('--dataset', default='IP', type=str, help='dataset (options: IP, UP, SV, KSC)')
    parser.add_argument('--tr_percent', default=0.1, type=float, help='samples of train set')
    parser.add_argument('--tr_bsize', default=100, type=int, help='mini-batch train size (default: 100)')
    parser.add_argument('--te_bsize', default=1000, type=int, help='mini-batch test size (default: 1000)')
    parser.add_argument('--depth', default=32, type=int, help='depth of the network (default: 32)')
    parser.add_argument('--alpha', default=48, type=int, help='number of new channel increases per depth (default: 12)')
    parser.add_argument('--inplanes', dest='inplanes', default=16, type=int, help='bands before blocks')
    parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false', help='to use basicblock (default: bottleneck)')
    parser.add_argument('--spatialsize', dest='spatialsize', default=11, type=int, help='spatial-spectral patch dimension')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--use_val', action='store_true', help='Use validation set')
    parser.add_argument('--val_percent', default=0.1, type=float, help='samples of val set')

    parser.set_defaults(bottleneck=True)
    best_err1 = 100

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}

    train_loader, test_loader, val_loader, num_classes, n_bands = load_hyper(args)

    # Use CUDA
    use_cuda = torch.cuda.is_available()
    if use_cuda: torch.backends.cudnn.benchmark = True

    if args.spatialsize < 9: avgpoosize = 1
    elif args.spatialsize <= 11: avgpoosize = 2
    elif args.spatialsize == 15: avgpoosize = 3
    elif args.spatialsize == 19: avgpoosize = 4
    elif args.spatialsize == 21: avgpoosize = 5
    elif args.spatialsize == 27: avgpoosize = 6
    elif args.spatialsize == 29: avgpoosize = 7
    else: print("spatialsize no tested")

    model = PYRM.pResNet(args.depth, args.alpha, num_classes, n_bands, avgpoosize, args.inplanes, bottleneck=args.bottleneck)
    if use_cuda: model = model.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)

    best_acc = -1
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        if args.use_val: test_loss, test_acc = test(val_loader, model, criterion, epoch, use_cuda)
        else: test_loss, test_acc = test(test_loader, model, criterion, epoch, use_cuda)

        print("EPOCH", epoch, "TRAIN LOSS", train_loss, "TRAIN ACCURACY", train_acc, end=',')
        print("LOSS", test_loss, "ACCURACY", test_acc)
        # save model
        if test_acc > best_acc:
            state = {
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
            }
            torch.save(state, "best_model.pth.tar")
            best_acc = test_acc

    checkpoint = torch.load("best_model.pth.tar")
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    test_loss, test_acc = test(test_loader, model, criterion, epoch, use_cuda)
    print("FINAL:      LOSS", test_loss, "ACCURACY", test_acc)
    classification, confusion, results = auxil.reports(np.argmax(predict(test_loader, model, criterion, use_cuda), axis=1), np.array(test_loader.dataset.__labels__()), args.dataset)
    print(classification)
    print(args.dataset, results)


if __name__ == '__main__':
	main()

