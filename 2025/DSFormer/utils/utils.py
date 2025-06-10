import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import torch
import os
from sklearn.decomposition import PCA

def split_info_print(train_gt, test_gt, labels):
    train_class_num = []
    test_class_num = []
    for i in range(len(labels)):
        train_class_num.append(np.sum(train_gt == i))
        test_class_num.append(np.sum(test_gt == i))
    print("class", "train", "test")
    for i in range(len(labels)):
        print(labels[i], train_class_num[i], test_class_num[i])


def sliding_window(image, step=1, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.
    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size
    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    for x in range(0, W - w + step, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + step, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h
            else:
                yield x, y, w, h


def count_sliding_window(image, step=1, window_size=(20, 20)):
    """ Count the number of windows in an image.
    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(image, step, window_size, with_data=False)
    return sum(1 for _ in sw)


def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.
    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable
    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, accuracy by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=bool)
    ignored_mask[target < 0] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]
    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy /= float(total)

    results["Accuracy"] = accuracy * 100.0

    # Compute accuracy of each class
    class_acc = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            acc = cm[i, i] / np.sum(cm[i, :])
        except ZeroDivisionError:
            acc = 0.
        class_acc[i] = acc

    results["class acc"] = class_acc * 100.0
    results['AA'] = np.mean(class_acc) * 100.0
    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
         float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa * 100.0

    return results


def show_results(results, label_values=None, agregated=False, file_path=None, avg_time_train=None, std_time_train=None,
                 avg_time_test=None, std_time_test=None, avg_time=None, std_time=None,opts=None):
    text = ""

    if opts is not None:
        text += "---\n"
        text += "Arguments:\n"
        for arg in vars(opts):
            text += f"{arg}: {getattr(opts, arg)}\n"
        text += "---\n"

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        aa = [r['AA'] for r in results]
        kappas = [r["Kappa"] for r in results]
        class_acc = [r["class acc"] for r in results]

        class_acc_mean = np.mean(class_acc, axis=0)
        class_acc_std = np.std(class_acc, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        aa = results['AA']
        classacc = results["class acc"]
        kappa = results["Kappa"]

    text += "Confusion matrix :\n"
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.02f}±{:.02f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
    else:
        text += "Accuracy : {:.02f}%\n".format(accuracy)
    text += "---\n"

    text += "class acc :\n"
    if agregated:
        for label, score, std in zip(label_values, class_acc_mean,
                                     class_acc_std):
            text += "\t{}: {:.02f}±{:.02f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, classacc):
            text += "\t{}: {:.02f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("AA: {:.02f}±{:.02f}\n".format(np.mean(aa),
                                                   np.std(aa)))
        text += ("Kappa: {:.02f}±{:.02f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "AA: {:.02f}%\n".format(aa)
        text += "Kappa: {:.02f}\n".format(kappa)

    if avg_time_train is not None and std_time_train is not None:
        text += "---\n"
        text += ("train_runtime: {:.02f}±{:.02f}\n".format(avg_time_train,
                                                std_time_train))

    if avg_time_test is not None and std_time_test is not None:
        text += "---\n"
        text += ("test_runtime: {:.02f}±{:.02f}\n".format(avg_time_test,
                                                std_time_test))

    if avg_time is not None and std_time is not None:
        text += "---\n"
        text += ("runtime: {:.02f}±{:.02f}\n".format(avg_time,
                                                std_time))


    # os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if agregated and file_path:
        with open(file_path, 'w') as file:
            file.write(text)

    print(text)


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights


def applyPCA(X, numComponents):
    """
    Apply Principal Component Analysis (PCA) to the input 3D array.

    Parameters:
    - X: numpy array
        The input 3D array with dimensions (num_samples, num_time_steps, num_features).
    - numComponents: int
        The number of principal components to retain after PCA.

    Returns:
    - newX: numpy array
        The transformed 3D array after applying PCA, with dimensions (num_samples, num_time_steps, numComponents).
    """
    # Reshape the input array to 2D for PCA
    newX = np.reshape(X, (-1, X.shape[2]))

    # Initialize PCA with the specified number of components and enable whitening
    pca = PCA(n_components=numComponents, whiten=True)

    # Apply PCA transformation to the reshaped array
    newX = pca.fit_transform(newX)

    # Reshape the transformed array back to its original 3D shape
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    # Return the transformed array after PCA
    return newX


def padWithZeros(X, margin=2):
    """
    Pad the input 3D array with zeros around its edges.

    Parameters:
    - X: numpy array
        The input 3D array with dimensions (num_samples, num_time_steps, num_features).
    - margin: int, optional (default=2)
        The number of zero-padding rows and columns to add around the edges of the array.

    Returns:
    - newX: numpy array
        The padded 3D array with dimensions (num_samples + 2 * margin, num_time_steps + 2 * margin, num_features).
    """
    # Create a new array filled with zeros, larger than the original array with padding
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))

    # Set the offsets for placing the original array within the zero-padded array
    x_offset = margin
    y_offset = margin

    # Copy the original array to the center of the zero-padded array
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    # Return the resulting zero-padded array
    return newX


def DrawResult(height, width, num_class, labels):
    num_class = num_class
    row = height
    col = width
    # palette = np.array([[37, 58, 150],
    #                     [47, 78, 161],
    #                     [56, 87, 166],
    #                     [56, 116, 186],
    #                     [51, 181, 232],
    #                     [112, 204, 216],
    #                     [119, 201, 168],
    #                     [148, 204, 120],
    #                     [188, 215, 78],
    #                     [238, 234, 63],
    #                     [246, 187, 31],
    #                     [244, 127, 33],
    #                     [239, 71, 34],
    #                     [238, 33, 35],
    #                     [180, 31, 35],
    #                     [123, 18, 20]])
    palette = np.array([[37, 58, 150],
                        # [47, 78, 161],
                        # [56, 87, 166],
                        # [56, 116, 186],
                        [51, 181, 232],
                        [112, 204, 216],
                        # [119, 201, 168],
                        [148, 204, 120],
                        [188, 215, 78],
                        [238, 234, 63],
                        # [246, 187, 31],
                        [244, 127, 33],
                        [239, 71, 34],
                        [238, 33, 35],
                        # [180, 31, 35],
                        [123, 18, 20]])
    palette = palette[:num_class]
    palette = palette * 1.0 / 255
    X_result = np.zeros((labels.shape[0], 3))
    for i in range(1, num_class + 1):
        X_result[np.where(labels == i), 0] = palette[i - 1, 0]
        X_result[np.where(labels == i), 1] = palette[i - 1, 1]
        X_result[np.where(labels == i), 2] = palette[i - 1, 2]
    X_result = np.reshape(X_result, (row, col, 3))
    plt.axis("off")
    plt.imshow(X_result)
    return X_result


def Draw(model,image,gt,patch_size,dataset_name,model_name,num_classes,device):
    height = gt.shape[0]
    width = gt.shape[1]
    # device = torch.device("cuda")
    image=padWithZeros(image,patch_size//2)
    outputs = np.zeros((height,width))
    for i in range(height):
        for j in range(width):
            if int(gt[i, j]) == -1:
                continue
            else:
                image_patch = image[i:i + patch_size, j:j + patch_size, :]
                image_patch = image_patch.reshape(1, image_patch.shape[0], image_patch.shape[1], image_patch.shape[2],
                                                  1)
                X_test_image = torch.FloatTensor(
                    image_patch.transpose(0, 4, 3, 1, 2)).to(device)
                prediction = model(X_test_image)
                prediction = np.argmax(prediction.detach().cpu().numpy(), axis=1)
                outputs[i][j] = prediction + 1
        if i % 20 == 0:
            print('... ... row ', i, ' handling ... ...')
    predict_labels = np.array(outputs).flatten()
    img = DrawResult(height,width,num_classes,np.reshape(predict_labels,-1))
    plt.imsave( dataset_name+ model_name+'.png', img)
