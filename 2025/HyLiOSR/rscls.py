# -*- coding: utf-8 -*-


import numpy as np
import copy
import scipy.stats as stats
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


class rscls:
    def __init__(self, im, gt, cls):
        if cls == 0:
            print('num of class not specified !!')
        self.im = copy.deepcopy(im)
        if gt.max() != cls:
            self.gt = copy.deepcopy(gt - 1)
        else:
            self.gt = copy.deepcopy(gt - 1)
        self.gt_b = copy.deepcopy(gt)
        self.cls = cls
        self.patch = 1
        self.imx, self.imy, self.imz = self.im.shape
        self.record = []
        self.sample = {}

    def padding(self, patch):
        self.patch = patch
        pad = self.patch // 2
        r1 = np.repeat([self.im[0, :, :]], pad, axis=0)
        r2 = np.repeat([self.im[-1, :, :]], pad, axis=0)
        self.im = np.concatenate((r1, self.im, r2))
        r1 = np.reshape(self.im[:, 0, :], [self.imx + 2 * pad, 1, self.imz])
        r2 = np.reshape(self.im[:, -1, :], [self.imx + 2 * pad, 1, self.imz])
        r1 = np.repeat(r1, pad, axis=1)
        r2 = np.repeat(r2, pad, axis=1)
        self.im = np.concatenate((r1, self.im, r2), axis=1)
        self.im = self.im.astype('float32')

    def normalize(self, style='01'):
        im = self.im
        for i in range(im.shape[-1]):
            im[:, :, i] = (im[:, :, i] - im[:, :, i].min()) / (im[:, :, i].max() - im[:, :, i].min())
        if style == '-11':
            im = im * 2 - 1

    def locate_sample(self):
        sam = []
        for i in range(self.cls):
            _xy = np.array(np.where(self.gt == i)).T
            _sam = np.concatenate([_xy, i * np.ones([_xy.shape[0], 1])], axis=-1)
            try:
                sam = np.concatenate([sam, _sam], axis=0)
            except:
                sam = _sam
        self.sample = sam.astype(int)

    def get_patch(self, xy):
        d = self.patch // 2
        x = xy[0]
        y = xy[1]
        try:
            self.im[x][y]
        except IndexError:
            return []
        x += d
        y += d
        sam = self.im[(x - d):(x + d + 1), (y - d):(y + d + 1)]
        return np.array(sam)

    def train_sample(self, pn):
        x_train, y_train = [], []
        self.locate_sample()
        _samp = self.sample
        for _cls in range(self.cls):
            _xy = _samp[_samp[:, 2] == _cls]
            np.random.shuffle(_xy)
            _xy = _xy[:pn, :]
            for xy in _xy:
                self.gt[xy[0], xy[1]] = 255  # !!
                #
                x_train.append(self.get_patch(xy[:-1]))
                y_train.append(xy[-1])
            # print(_xy)
        x_train, y_train = np.array(x_train), np.array(y_train)
        idx = np.random.permutation(x_train.shape[0])
        x_train = x_train[idx]
        y_train = y_train[idx]
        return x_train, y_train.astype(int)

    def test_sample(self):
        x_test, y_test = [], []
        self.locate_sample()
        _samp = self.sample
        for _cls in range(self.cls):
            _xy = _samp[_samp[:, 2] == _cls]
            np.random.shuffle(_xy)
            for xy in _xy:
                x_test.append(self.get_patch(xy[:-1]))
                y_test.append(xy[-1])
        return np.array(x_test), np.array(y_test)

    def all_sample(self):
        imx, imy = self.gt.shape
        
        print("x轴为：",imx)
        print("y轴为：",imy)
        sample = []
        
        for i in range(imx):
            for j in range(imy):
                sample.append(self.get_patch(np.array([i, j])))
        
        gt1 = self.gt.reshape(-1,1)
        
        return np.array(sample),np.array(gt1)

    def all_sample_light(self, clip=0, bs=10):
        imx, imy = self.gt.shape
        imz = self.im.shape[-1]
        patch = self.patch
        # fp = np.memmap('allsample' + str(clip) + '.h5', dtype='float32', mode='w+', shape=(imgx*self.IMGY,5,5,bs))
        fp = np.zeros([imx * imy, patch, patch, imz])
        countnum = 0
        for i in range(imx * clip, imx * (clip + 1)):
            for j in range(imy):
                xy = np.array([i, j])
                fp[countnum, :, :, :] = self.get_patch(xy)
                countnum += 1
        return fp

    def all_sample_row_hd(self, sub=0):
        imx, imy = self.gt.shape
        imz = self.im.shape[-1]
        patch = self.patch
        # fp = np.memmap('allsample' + str(clip) + '.h5', dtype='float32', mode='w+', shape=(imgx*self.IMGY,5,5,bs))
        fp = np.zeros([imx * imy, patch, patch, imz])
        countnum = 0
        for i in range(sub):
            for j in range(imy):
                xy = np.array([i, j])
                fp[countnum, :, :, :] = self.get_patch(xy)
                countnum += 1
        return fp

    def all_sample_row(self, sub=0):
        imx, imy = self.gt.shape
        fp = []
        for j in range(imy):
            xy = np.array([sub, j])
            fp.append(self.get_patch(xy))
        return np.array(fp)

    def all_sample_heavy(self, name, clip=0, bs=10):
        imx, imy = self.gt.shape
        imz = self.im.shape[-1]
        patch = self.patch
        try:
            fp = np.memmap(name, dtype='float32', mode='w+', shape=(imx * imy, patch, patch, imz))
        except:
            fp = np.memmap(name, dtype='float32', mode='r', shape=(imx * imy, patch, patch, imz))
        # fp = np.zeros([imx*imy,patch,patch,imz])
        countnum = 0
        for i in range(imx * clip, imx * (clip + 1)):
            for j in range(imy):
                xy = np.array([i, j])
                fp[countnum, :, :, :] = self.get_patch(xy)
                countnum += 1
        return fp

    def read_all_sample(self, name, clip=0, bs=10):
        imx, imy = self.gt.shape
        imz = self.im.shape[-1]
        patch = self.patch
        fp = np.memmap(name, dtype='float32', mode='r', shape=(imx * imy, patch, patch, imz))
        return fp

    def locate_obj(self, seg):
        obj = {}
        for i in range(seg.min(), seg.max() + 1):
            obj[str(i)] = np.where(seg == i)
        self.obj = obj
        self.seg = seg


def obpc(seg, cmap, obj):
    pcmap = copy.deepcopy(cmap)
    for (k, v) in obj.items():
        tmplabel = stats.mode(cmap[v])[0]
        pcmap[v] = tmplabel
    return pcmap


def cfm(pre, ref, ncl=9):
    if ref.min() != 0:
        print('warning: label should begin with 0 !!')
        return

    nsize = ref.shape[0]
    cf = np.zeros((ncl, ncl))
    for i in range(nsize):
        cf[pre[i], ref[i]] += 1

    tmp1 = 0
    for j in range(ncl):
        tmp1 = tmp1 + (cf[j, :].sum() / nsize) * (cf[:, j].sum() / nsize)
    cfm = np.zeros((ncl + 2, ncl + 1))
    cfm[:-2, :-1] = cf
    oa = 0
    for i in range(ncl):
        if cf[i, :].sum():
            cfm[i, ncl] = cf[i, i] / cf[i, :].sum()
        if cf[:, i].sum():
            cfm[ncl, i] = cf[i, i] / cf[:, i].sum()
        oa += cf[i, i]
    cfm[-1, 0] = oa / nsize
    cfm[-1, 1] = (cfm[-1, 0] - tmp1) / (1 - tmp1)
    cfm[-1, 2] = cfm[ncl, :-1].mean()
    print('oa: ', format(cfm[-1, 0], '.5'), ' kappa: ', format(cfm[-1, 1], '.5'),
          ' mean: ', format(cfm[-1, 2], '.5'))
    return cfm


def gtcfm(pre, gt, ncl):
    pre = np.uint8(pre)
    gt = np.uint8(gt)
    if gt.max() == 255:
        print('warning: max 255 !!')
    cf = np.zeros([ncl, ncl])
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j]:
                cf[pre[i, j] - 1, gt[i, j] - 1] += 1
    tmp1 = 0
    nsize = np.sum(gt != 0)
    for j in range(ncl):
        tmp1 = tmp1 + (cf[j, :].sum() / nsize) * (cf[:, j].sum() / nsize)
    cfm = np.zeros((ncl + 2, ncl + 1))
    cfm[:-2, :-1] = cf
    oa = 0
    for i in range(ncl):
        if cf[i, :].sum():
            cfm[i, ncl] = cf[i, i] / cf[i, :].sum()
        if cf[:, i].sum():
            cfm[ncl, i] = cf[i, i] / cf[:, i].sum()
        oa += cf[i, i]
    cfm[-1, 0] = oa / nsize
    cfm[-1, 1] = (cfm[-1, 0] - tmp1) / (1 - tmp1)
    cfm[-1, 2] = cfm[ncl, :-1].mean()
    cfm[-1, 3] = cfm[:-2, ncl].mean()
    oa = cfm[-1, 0]
    aa = cfm[-1, 2]
    kappa = cfm[-1, 1]
    print('oa: ', format(oa, '.5'), ' kappa: ', format(kappa, '.5'),
          ' aa/pa: ', format(aa, '.5'), ' ua: ', format(cfm[-1, 3], '.5'))
    print('AA is :')
    osr = 0
    for acc in cfm[ncl, :-1]:
        print(acc)
        osr = acc
    

    return cfm,oa,aa,kappa,osr


def gtcfm0(pre, gt, ncl):
    if gt.max() == 255:
        print('warning: max 255 !!')
    cf = np.zeros([ncl, ncl])
    for i in range(gt.shape[0]):
        for j in range(gt.shape[1]):
            if gt[i, j]:
                cf[pre[i, j], gt[i, j]] += 1
    tmp1 = 0
    nsize = np.sum(gt != 0)
    for j in range(ncl):
        tmp1 = tmp1 + (cf[j, :].sum() / nsize) * (cf[:, j].sum() / nsize)
    cfm = np.zeros((ncl + 2, ncl + 1))
    cfm[:-2, :-1] = cf
    oa = 0
    for i in range(ncl):
        if cf[i, :].sum():
            cfm[i, ncl] = cf[i, i] / cf[i, :].sum()
        if cf[:, i].sum():
            cfm[ncl, i] = cf[i, i] / cf[:, i].sum()
        oa += cf[i, i]
    cfm[-1, 0] = oa / nsize
    cfm[-1, 1] = (cfm[-1, 0] - tmp1) / (1 - tmp1)
    cfm[-1, 2] = cfm[ncl, :-1].mean()
    cfm[-1, 3] = cfm[:-2, ncl].mean()
    print('oa: ', format(cfm[-1, 0], '.5'), ' kappa: ', format(cfm[-1, 1], '.5'),
          ' aa/pa: ', format(cfm[-1, 2], '.5'), ' ua: ', format(cfm[-1, 3], '.5'))
    return cf, cfm


def svm(trainx, trainy):
    cost = []
    gamma = []
    for i in range(-5, 16, 2):
        cost.append(np.power(2.0, i))
    for i in range(-15, 4, 2):
        gamma.append(np.power(2.0, i))

    parameters = {'C': cost, 'gamma': gamma}
    svm = SVC(verbose=0, kernel='rbf')
    clf = GridSearchCV(svm, parameters, cv=3)
    p = clf.fit(trainx, trainy)

    print(clf.best_params_)
    bestc = clf.best_params_['C']
    bestg = clf.best_params_['gamma']
    tmpc = [-1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.0,
            0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    cost = []
    gamma = []
    for i in tmpc:
        cost.append(bestc * np.power(2.0, i))
        gamma.append(bestg * np.power(2.0, i))
    parameters = {'C': cost, 'gamma': gamma}
    svm = SVC(verbose=0, kernel='rbf')
    clf = GridSearchCV(svm, parameters, cv=3)
    p = clf.fit(trainx, trainy)
    print(clf.best_params_)
    p2 = clf.best_estimator_
    return p2


def svm_rbf(trainx, trainy):
    cost = []
    gamma = []
    for i in range(-3, 10, 2):
        cost.append(np.power(2.0, i))
    for i in range(-5, 4, 2):
        gamma.append(np.power(2.0, i))

    parameters = {'C': cost, 'gamma': gamma}
    svm = SVC(verbose=0, kernel='rbf')
    clf = GridSearchCV(svm, parameters, cv=3)
    clf.fit(trainx, trainy)

    # print(clf.best_params_)
    bestc = clf.best_params_['C']
    bestg = clf.best_params_['gamma']
    tmpc = [-1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.0,
            0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    cost = []
    gamma = []
    for i in tmpc:
        cost.append(bestc * np.power(2.0, i))
        gamma.append(bestg * np.power(2.0, i))
    parameters = {'C': cost, 'gamma': gamma}
    svm = SVC(verbose=0, kernel='rbf')
    clf = GridSearchCV(svm, parameters, cv=3)
    clf.fit(trainx, trainy)
    # print(clf.best_params_)
    p = clf.best_estimator_
    return p


def rf(trainx, trainy, sim=1, nj=1):
    nest = []
    nfea = []
    for i in range(20, 201, 20):
        nest.append(i)
    if sim:
        for i in range(1, int(trainx.shape[-1])):
            nfea.append(i)
        parameters = {'n_estimators': nest, 'max_features': nfea}
    else:
        parameters = {'n_estimators': nest}
    rf = RandomForestClassifier(n_jobs=nj, verbose=0, oob_score=False)
    clf = GridSearchCV(rf, parameters, cv=3)
    p = clf.fit(trainx, trainy)
    p2 = clf.best_estimator_
    return p2


def GNB(trainx, trainy):
    clf = GaussianNB()
    p = clf.fit(trainx, trainy)
    return p


def svm_linear(trainx, trainy):
    cost = []
    for i in range(-3, 10, 2):
        cost.append(np.power(2.0, i))

    parameters = {'C': cost}
    svm = SVC(verbose=0, kernel='linear')
    clf = GridSearchCV(svm, parameters, cv=3)
    clf.fit(trainx, trainy)

    # print(clf.best_params_)
    bestc = clf.best_params_['C']
    tmpc = [-1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0.0,
            0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    cost = []
    for i in tmpc:
        cost.append(bestc * np.power(2.0, i))
    parameters = {'C': cost}
    svm = SVC(verbose=0, kernel='linear')
    clf = GridSearchCV(svm, parameters, cv=3)
    clf.fit(trainx, trainy)
    p = clf.best_estimator_
    return p


def make_sample(sample, label):
    a = np.flip(sample, 1)   
    b = np.flip(sample, 2)   
    c = np.flip(b, 1)
    newsample = np.concatenate((a, b, c, sample), axis=0)
    newlabel = np.concatenate((label, label, label, label), axis=0)
    return newsample, newlabel


def save_cmap(img, cmap, fname):
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()


def strimg255(im, perc=0.5):
    maxx = np.percentile(im, 100 - perc)
    minn = np.percentile(im, perc)
    im[im > maxx] = maxx
    im[im < minn] = minn
    im_new = np.fix((im - minn) / (maxx - minn) * 255).astype(np.uint8)
    return im_new


