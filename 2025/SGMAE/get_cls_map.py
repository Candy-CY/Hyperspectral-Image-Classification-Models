import numpy as np
import matplotlib.pyplot as plt

def get_classification_map(y_pred, y, iii):

    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))

    if iii == 1:
        for i in range(height):
            for j in range(width):
                target = int(y[i, j])

                cls_labels[i][j] = y_pred[k]
                k += 1

    if iii == 2:
        for i in range(height):
            for j in range(width):
                target = int(y[i, j])


                if target == 0:
                    continue
                else:
                    cls_labels[i][j] = y_pred[k]
                    k += 1

    return  cls_labels

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.

        if item == 1:
            y[index] = np.array([147, 67, 46]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 100, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 123]) / 255.
        if item == 5:
            y[index] = np.array([164, 75, 155]) / 255.
        if item == 6:
            y[index] = np.array([101, 174, 255]) / 255.
        if item == 7:
            y[index] = np.array([118, 254, 172]) / 255.
        if item == 8:
            y[index] = np.array([60, 91, 112]) / 255.
        if item == 9:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 10:
            y[index] = np.array([255, 255, 125]) / 255.
        if item == 11:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 12:
            y[index] = np.array([100, 0, 255]) / 255.
        if item == 13:
            y[index] = np.array([0, 172, 254]) / 255.
        if item == 14:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 15:
            y[index] = np.array([171, 175, 80]) / 255.
        if item == 16:
            y[index] = np.array([101, 193, 60]) / 255.
        elif item == 17:
            y[index] = np.array([255, 102, 102]) / 255.
        elif item == 18:
            y[index] = np.array([255, 158, 204]) / 255.
        elif item == 19:
            y[index] = np.array([204, 204, 255]) / 255.
        elif item == 20:
            y[index] = np.array([255, 204, 153]) / 255.
        elif item == 21:
            y[index] = np.array([153, 255, 153]) / 255.
        elif item == 22:
            y[index] = np.array([0, 204, 204]) / 255.
        elif item == 23:
            y[index] = np.array([204, 153, 255]) / 255.
        elif item == 24:
            y[index] = np.array([153, 76, 0]) / 255.

    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def test(device, net, test_loader):
    count = 0
    net.eval()
    y_pred_test = 0
    y_test = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def generate_classification_map(net, device, data_loader, y, map_index, output_prefix):
    y_pred, _ = test(device, net, data_loader)
    cls_labels = get_classification_map(y_pred, y, map_index)
    x = np.ravel(cls_labels)
    y_list = list_to_colormap(x)
    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))

    classification_map(y_re, y, 300, f'{output_prefix}.png')
    return cls_labels


def get_cls_map(net, device, all_data_loader, all_data_loader_whole, y):

    generate_classification_map(net, device, all_data_loader, y, 2, './pictures/' + 'pred'+'_1')
    generate_classification_map(net, device, all_data_loader_whole, y, 1, './pictures/' + 'pred'+'_2')

    gt = y.flatten()
    y_gt = list_to_colormap(gt)
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    classification_map(gt_re, y, 300, './pictures/' + '_gt.png')

    print('------Get classification maps successful-------')