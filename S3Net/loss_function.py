import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.25):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y, x0_, x1_):
        self.check_type_forward((x0, x1, y))
        # euclidian distance
        # x0_ = F.normalize(x0_)
        # x1_ = F.normalize(x1_)
        cos = torch.cosine_similarity(x0_, x1_, dim=1)
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)
        mdist = self.margin - dist
        dist = torch.clamp(mdist, min=0.0)
        loss = y * (1-cos) * dist_sq + (1 - y) * cos * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss
#
class TripletLoss(torch.nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
    def forward(self, x0, x1, y, x0_, x1_):
        x0 = F.normalize(x0)
        x1 = F.normalize(x1)
        x1 = x1.T
        res = 2.0- 2 * torch.mm(x0, x1)
        pos = torch.diag(res)
        pos = pos.unsqueeze(1)
        res = self.margin+ pos - res
        # loss, _ = torch.topk(res, 3, dim=1, largest=True)
        # loss = torch.log(1 + torch.exp(res))
        loss = torch.mean(res, 1)
        loss = torch.mean(loss)
        return loss

#Softmarginloss
class CircleLoss(nn.Module):
    def __init__(self, scale=32, margin=0.25, similarity='cos', **kwargs):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            "feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"

        m = labels.size(0)
        mask = labels.expand(m, m).t().eq(labels.expand(m, m)).float()
        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss

#es-cnnloss
class ESCNNLoss(torch.nn.Module):
    """
    ESCNNLoss function.
    Based on:
    """

    def __init__(self, margin=1.25):
        super(ESCNNLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        self.check_type_forward((x0, x1, y))
        # euclidian distance
        y_prep = torch.softmax(x0+x1,1)
        first_part = torch.sum(torch.add(torch.multiply(y.cuda().unsqueeze(-1), torch.log(y_prep)), torch.multiply((1-y).cuda().unsqueeze(-1), torch.log(1.0-y_prep))),1)
        t = torch.sum(torch.multiply(y.cuda().unsqueeze(-1), y_prep), 1)
        second_part_1 = y_prep + 0.2 - torch.sum(torch.multiply(y.cuda().unsqueeze(-1), y_prep), 1).unsqueeze(-1)
        compare_matrix = torch.zeros_like(second_part_1)
        second_part_1 = torch.max(second_part_1, compare_matrix)
        second_part = torch.mean(torch.multiply(y.cuda().unsqueeze(-1), second_part_1), 1)
        loss = 0.0 - torch.mean(first_part + 0.05*second_part)
        return loss

class FocalLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.25):
        super(FocalLoss, self).__init__()
        self.margin = margin

    def check_type_forward(self, in_types):
        assert len(in_types) == 3

        x0_type, x1_type, y_type = in_types
        assert x0_type.size() == x1_type.shape
        assert x1_type.size()[0] == y_type.shape[0]
        assert x1_type.size()[0] > 0
        assert x0_type.dim() == 2
        assert x1_type.dim() == 2
        assert y_type.dim() == 1

    # def forward(self, x0, x1, y):
    #     self.check_type_forward((x0, x1, y))
    #
    #     # euclidian distance
    #     diff = x0 - x1
    #     dist_sq = torch.sum(torch.pow(diff, 2), 1)
    #     dist = torch.sqrt(dist_sq)
    #     mdist = self.margin - dist
    #     dist = torch.clamp(mdist, min=0.0)
    #     loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
    #     loss = torch.sum(loss) / 2.0 / x0.size()[0]

    def forward(self, x0, x1, targets):
        diff = x0-x1
        dist_sq = torch.pow(diff, 2)
        inputs = diff
        N = inputs.size(0)
        C = inputs.size(1)
        P = torch.softmax(inputs, 1)
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class CosLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, mar_gin=0.1, margin=1.0):
        super(CosLoss, self).__init__()
        self.margin = margin
        self.mar_gin = mar_gin
        self.idx = 0
    # def check_type_forward(self, in_types):
    #     assert len(in_types) == 3
    #
    #     x0_type, x1_type, y_type = in_types
    #     assert x0_type.size() == x1_type.shape
    #     assert x1_type.size()[0] == y_type.shape[0]
    #     assert x1_type.size()[0] > 0
    #     assert x0_type.dim() == 2
    #     assert x1_type.dim() == 2
    #     assert y_type.dim() == 1

    def forward(self, x0, x1, y):
        #方向,点乘小丸子
        #欧式，最后的分类的
        self.idx+=1
        x0_soft = F.softmax(x0,1)
        x1_soft = F.softmax(x1,1)
        # cosdis = torch.sum(x0_soft*x0_soft) + torch.sum(x1_soft*x1_soft) - 2* torch.mm(x0_soft, torch.transpose(x1_soft,0,1))
        cosdis = torch.matmul(x0_soft, x1_soft.T)
        coslabel = torch.diag(cosdis)
        coslabel = torch.div(coslabel, torch.linalg.norm(x0_soft, 1) * torch.linalg.norm(x1_soft, 1))

        if self.idx==60:
            coslabel = torch.div(coslabel, torch.linalg.norm(x0_soft, 1) * torch.linalg.norm(x1_soft, 1))
        e_diff = x0 - x1
        e_dist_sq = torch.sum(torch.pow(e_diff, 2), 1)
        e_dist = torch.sqrt(e_dist_sq)
        e_mdist = self.margin - e_dist
        e_dist = torch.clamp(e_mdist, min=0.0)
        # e_loss = y*coslabel*e_dist_sq + (1-y)*coslabel* torch.pow(e_dist, 2)
        e_loss = 2*y*abs(y-0.5)*e_dist_sq + 2*(1 - y)* abs(y-0.5) * torch.pow(e_dist, 2) + 400*y*(1-y)*e_dist_sq* coslabel
        # e_loss = y*e_dist_sq + (1 - y)*torch.pow(e_dist, 2)

        loss_1 = torch.sum(e_loss) / 2.0 / x0.size()[0]

        return loss_1


class CosLoss2(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self):
        super(CosLoss2, self).__init__()

    def forward(self, x0, x1, y):
        loss = torch.nn.CrossEntropyLoss()(x0+x1, y)

        return loss

class SofmarginLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.5):
        super(SofmarginLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # 方向
        x0 = F.normalize(x0, dim=-1)
        x1 = F.normalize(x1, dim=-1)
        x1 = torch.t(x1)
        dist_array = 2 - 2 * torch.matmul(x0, x1)
        pos_dist = torch.diagonal(dist_array)
        pair_n = x0.shape[0] * (x0.shape[0] - 1.0)

        # x0-to-x1
        triplet_dist_g2s = pos_dist - dist_array
        # loss_g2s = torch.sum(torch.clamp(triplet_dist_g2s + self.margin, min=0)) / pair_n
        loss_g2s = torch.sum(pos_dist + torch.log(1 + torch.clamp(triplet_dist_g2s + self.margin, min=0))) / pair_n
        # satellite to ground
        triplet_dist_s2g = torch.unsqueeze(pos_dist, 1) - dist_array
        # loss_s2g = torch.sum(torch.clamp(triplet_dist_s2g + self.margin, min=0)) / pair_n
        loss_s2g = torch.sum(pos_dist + torch.log(1 + torch.clamp(triplet_dist_s2g + self.margin, min=0))) / pair_n
        loss = (loss_g2s + loss_s2g)/2
        # else:
        #     # ground to satellite
        #     triplet_dist_g2s = pos_dist - dist_array
        #     triplet_dist_g2s = tf.log(1 + tf.exp(triplet_dist_g2s * loss_weight))
        #     top_k_g2s, _ = tf.nn.top_k(tf.transpose(triplet_dist_g2s), batch_hard_count)
        #     loss_g2s = tf.reduce_mean(top_k_g2s)
        #
        #     # satellite to ground
        #     triplet_dist_s2g = tf.expand_dims(pos_dist, 1) - dist_array
        #     triplet_dist_s2g = tf.log(1 + tf.exp(triplet_dist_s2g * loss_weight))
        #     top_k_s2g, _ = tf.nn.top_k(triplet_dist_s2g, batch_hard_count)
        #     loss_s2g = tf.reduce_mean(top_k_s2g)
        #
        #     loss = (loss_g2s + loss_s2g) / 2.0
        return loss
class direction_Loss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.2):
        super(direction_Loss, self).__init__()
        self.margin = margin

    # def check_type_forward(self, in_types):
    #     assert len(in_types) == 3
    #
    #     x0_type, x1_type, y_type = in_types
    #     assert x0_type.size() == x1_type.shape
    #     assert x1_type.size()[0] == y_type.shape[0]
    #     assert x1_type.size()[0] > 0
    #     assert x0_type.dim() == 2
    #     assert x1_type.dim() == 2
    #     assert y_type.dim() == 1

    def forward(self, x0, x1, x_c_0, x_c_1, y):
        #方向
        a = F.normalize(x0, dim=-1)
        b = F.normalize(x1, dim=-1)
        b = torch.t(b)
        cose = torch.mm(a,b)
        dig_cose = torch.diagonal(cose)
        dig_cose = 0.5 + 0.5 * dig_cose
        cos_loss = y * (1 - dig_cose) + (1-y) * dig_cose
        sine = torch.sqrt(1.0-dig_cose*dig_cose)
        #大小
        x0_f = F.normalize(x0, dim=-1)
        x1_f = F.normalize(x1, dim=-1)
        x1_0 = x1_f * dig_cose
        x1_1 = x1_f * sine
        l_diff = torch.sqrt(torch.sqrt(torch.pow((torch.pow(x0_f, 2) - torch.pow(x1_0, 2)), 2)))
        l_diff = torch.sum(l_diff, 1)
        dist = torch.sum(torch.sqrt(torch.pow(x1_1,2)) ,1)
        # loss = l_diff
        # loss_n = 1 - loss
        # loss_n = torch.clamp(loss_n, min=0.0)
        loss = y * (l_diff-dist) + (1 - y) * (dist-l_diff)
        e_diff = x0 - x1
        e_dist_sq = torch.sum(torch.pow(e_diff, 2), 1)
        e_dist = torch.sqrt(e_dist_sq)
        e_mdist = self.margin - e_dist
        e_dist = torch.clamp(e_mdist, min=0.0)
        e_loss = y * e_dist_sq + (1 - y) * e_dist
        loss = cos_loss + e_loss
        loss = torch.sum(e_loss) / 2.0 / x0.size()[0]
        return loss