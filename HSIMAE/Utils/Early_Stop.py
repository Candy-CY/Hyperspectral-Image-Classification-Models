import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.metric_max = None
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_metric, value, model, save_path):
        if self.best_score is None:
            self.metric = val_metric
            self.best_score = value
            self.save_checkpoint(val_metric, value, model, save_path)
        elif val_metric < self.metric_max + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.metric_max = val_metric
            self.best_score = value
            self.save_checkpoint(val_metric, value, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_metric, value, model, save_path):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(f'Validation metric increased ({self.val_metric_max:.6f} --> {val_metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), save_path)  # 这里会存储迄今最优模型的参数
        self.metric_max = val_metric
        self.best_score = value
