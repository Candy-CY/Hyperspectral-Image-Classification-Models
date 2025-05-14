import numpy as np
import spectral as spy
from spectral import spy_colors


def visualize_predict(gt,predict_label,save_predict_path,save_gt_path,only_vis_label=False):
    row, col = gt.shape[0], gt.shape[1]
    predict = np.reshape(predict_label,(row,col)) + 1
    if only_vis_label:
        vis_predict = np.where(gt==0,gt,predict)
    else:
        vis_predict = predict
    spy.save_rgb(save_predict_path, vis_predict, colors=spy_colors)
    spy.save_rgb(save_gt_path, gt, colors=spy_colors)