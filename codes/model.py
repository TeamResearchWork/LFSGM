import torch
import torch.nn as nn
import torch.nn.init
import sys
import time
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict

# key codes will be uploaded soon after acception

def l2norm(X, dim=1):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class ImageEncoder(nn.Module):
    def __init__(self, opt, img_dim, embed_size, use_abs=False, no_img_norm=False):
        super(ImageEncoder, self).__init__()
        self.embed_size = embed_size
        self.no_img_norm = no_img_norm
        self.use_abs = use_abs
        self.dataset = opt.dataset
        self.opt = opt

        self.fc = nn.Linear(img_dim, embed_size)
        self.fc_attn = nn.Linear(embed_size * 2, 1)
        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)

        self.init_weights()

        if not self.opt.no_fragment_attn:
            self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        if self.dataset == 'flickr30k':
            self.bn = nn.BatchNorm1d(embed_size)

    def init_weights(self):
        r1 = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r1, r1)
        self.fc.bias.data.fill_(0)

        r2 = np.sqrt(6.) / np.sqrt(self.embed_size * 2)
        self.fc_attn.weight.data.uniform_(-r2, r2)
        self.fc_attn.bias.data.fill_(0)

    def fragment_attn(self, global_features, local_features):
        input_cat = torch.cat([global_features, local_features], 2)
        weights = self.fc_attn(input_cat)
        weights = torch.sigmoid(weights)
        weights = nn.functional.softmax(input=weights, dim=1)
        weights = weights.squeeze(2).contiguous()

        return weights

    def forward(self, images):
        fc_img_emd = self.fc(images)

        if not (self.dataset == 'flickr30k' or self.dataset == 'AI2D#'):
            fc_img_emd = l2norm(fc_img_emd, dim=2)

        gcn_img_emd = fc_img_emd.permute(0, 2, 1)
        gcn_img_emd = self.Rs_GCN_1(gcn_img_emd)
        gcn_img_emd = self.Rs_GCN_2(gcn_img_emd)
        gcn_img_emd = self.Rs_GCN_3(gcn_img_emd)
        gcn_img_emd = self.Rs_GCN_4(gcn_img_emd)

        gcn_img_emd = gcn_img_emd.permute(0, 2, 1)
        gcn_img_emd = l2norm(gcn_img_emd, dim=2)

        if not self.opt.no_fragment_attn:
            rnn_img, hidden_state = self.img_rnn(gcn_img_emd)
            global_features = hidden_state[0]

            if self.dataset == 'flickr30k':
                global_features = self.bn(global_features)

            if not self.no_img_norm:
                global_features = l2norm(global_features)

            # calculate the weight for each fragment in loss
            features = global_features.unsqueeze(1).contiguous()
            features = features.expand_as(gcn_img_emd)
            frag_weights = self.fragment_attn(features, gcn_img_emd)

            return gcn_img_emd, global_features, frag_weights

        return gcn_img_emd, fc_img_emd

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(ImageEncoder, self).load_state_dict(new_state)


class TextEncoder(nn.Module):

    def __init__(self, opt, txt_dim, embed_size, num_layers, use_gru=False, bi_gru=False, no_txt_norm=False):
        super(TextEncoder, self).__init__()
        self.opt = opt
        self.embed_size = embed_size
        self.bi_gru = bi_gru
        self.use_gru = use_gru
        self.no_txt_norm = no_txt_norm

        self.rnn = nn.GRU(txt_dim, embed_size, num_layers, batch_first=True, bidirectional=bi_gru)
        self.fc_attn = nn.Linear(embed_size * 2, 1)

        self.init_weights()

    def init_weights(self):
        r2 = np.sqrt(6.) / np.sqrt(self.embed_size * 2)
        self.fc_attn.weight.data.uniform_(-r2, r2)
        self.fc_attn.bias.data.fill_(0)

    def fragment_attn(self, global_features, local_features, cap_lengths):
        input_cat = torch.cat([global_features, local_features], 2)
        weights_raw = self.fc_attn(input_cat)
        weights = torch.zeros(weights_raw.shape)

        for i in range(weights_raw.shape[0]):
            weight = torch.sigmoid(weights_raw[i][0:cap_lengths[i]])
            weight = nn.functional.softmax(input=weight, dim=0)
            weights[i][0:cap_lengths[i]] = weight
        weights = weights.squeeze(2).contiguous()

        return weights

    def forward(self, captions, lengths):

        if not self.use_gru:
            feature = self.fc(captions)
            lengths = torch.IntTensor(lengths)
            lengths = lengths.cuda()
            if not self.no_txt_norm:
                feature = nn.functional.normalize(feature, p=2, dim=-1, eps=1e-8)

            return feature, lengths

        packed_feature = pack_padded_sequence(captions, lengths.data.tolist(), batch_first=True)
        out, hidden_state = self.rnn(packed_feature)
        padded_feature = pad_packed_sequence(out, batch_first=True)
        cap_feature, cap_length = padded_feature

        if self.use_gru and self.bi_gru:
            cap_feature = (cap_feature[:, :, :int(cap_feature.size(2) / 2)] + cap_feature[:, :, int(cap_feature.size(2) / 2):]) / 2.

        for i in range(1, len(cap_feature)):
            cap_feature[i][lengths[i]:][:] = 1.

        if not self.no_txt_norm:
           cap_feature = l2norm(cap_feature, dim=2)

        for i in range(1, len(cap_feature)):
            cap_feature[i][lengths[i]:][:] = 0.

        if not self.opt.no_fragment_attn:
            global_features = hidden_state[0]

            if not self.no_txt_norm:
                global_features = l2norm(global_features)

            # calculate the weight for each fragment in loss --- attention mechanism
            features = global_features.unsqueeze(1).contiguous()
            features = features.expand_as(cap_feature)
            frag_weights = self.fragment_attn(features, cap_feature, cap_length)

            return cap_feature, global_features, cap_length, frag_weights

        return cap_feature, cap_feature, cap_length


def func_attention(query, context, opt, smooth, eps=1e-8, weight=None):
    """
    We changed the codes from open codes offered by
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    # Get attention
    queryT = torch.transpose(query, 1, 2)

    attn = torch.bmm(context, queryT)

    if opt.raw_feature_norm == "softmax":

        attn = attn.view(batch_size * sourceL, queryL)
        attn = nn.functional.softmax(attn, dim=1)

        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)

    if weight is not None:
        attn = attn + weight

    attn_out = attn.clone()
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    attn = nn.functional.softmax(attn * smooth, dim=1)
    attn = attn.view(batch_size, queryL, sourceL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    contextT = torch.transpose(context, 1, 2)
    weightedContext = torch.bmm(contextT, attnT)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attn_out


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return w12 / (w1 * w2).clamp(min=eps)


class Rs_GCN(nn.Module):
    def __init__(self, in_channels, inter_channels, bn_layer=True):
        """
        We changed the codes from open codes offered by
        """
        super(Rs_GCN, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv1d
        max_pool = nn.MaxPool1d
        bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant(self.W[1].weight, 0)
            nn.init.constant(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant(self.W.weight, 0)
            nn.init.constant(self.W.bias, 0)

        self.theta = None
        self.phi = None

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, v):
        batch_size = v.size(0)

        g_v = self.g(v).view(batch_size, self.inter_channels, -1)
        g_v = g_v.permute(0, 2, 1)

        theta_v = self.theta(v).view(batch_size, self.inter_channels, -1)
        theta_v = theta_v.permute(0, 2, 1)
        phi_v = self.phi(v).view(batch_size, self.inter_channels, -1)
        R = torch.matmul(theta_v, phi_v)
        N = R.size(-1)
        R_div_C = R / N

        y = torch.matmul(R_div_C, g_v)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *v.size()[2:])
        W_y = self.W(y)
        v_star = W_y + v

        return v_star


class ContrastiveLoss(nn.Module):

    def __init__(self, opt, cross_attn, loss_detail=False, margin=0, max_violation=False):
        """
        We changed the codes from open codes offered by
        """
        super(ContrastiveLoss, self).__init__()
        self.cross_attn = cross_attn
        self.margin = margin
        self.loss_detail = loss_detail
        self.max_violation = max_violation
        self.opt = opt

    def forward(self, scores):
        diagonal = scores.diag().view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_s = (self.margin + scores - d1).clamp(min=0)
        cost_im = (self.margin + scores - d2).clamp(min=0)

        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        loss_cap = cost_s.masked_fill_(I, 0)
        loss_img = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            loss_cap = loss_cap.max(1)[0]
            loss_img = loss_img.max(0)[0]
        if self.loss_detail:
            print("loss_cap: {} loss_img: {}".format(loss_cap.sum(), loss_img.sum()))

        return self.opt.lambda_txt_weight * loss_cap.sum() + loss_img.sum()


class CMMemory(nn.Module):
    def __init__(self, opt, memory_size, memory_dim, init_v_memory=None, init_t_memory=None):
        """
        The memory module memorizes the uncommon characters
        and enhances both the visual and textual feature representations.
        :param memory_size: the number of memory slots
        :param memory_dim: same as the visual and textual feature dimension in common embedding space
        """
        super(CMMemory, self).__init__()
        self.opt = opt
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.feature_dim = memory_dim
        self.v_memory = self.init_memory() if init_v_memory is None else Variable(torch.from_numpy(init_v_memory))
        self.t_memory = self.init_memory() if init_t_memory is None else Variable(torch.from_numpy(init_t_memory))
        self.memory_changed = False
        self.v_memory_new = self.v_memory
        self.t_memory_new = self.t_memory

        self.fc_ve = nn.Linear(self.feature_dim, self.memory_dim)
        self.fc_te = nn.Linear(self.feature_dim, self.memory_dim)
        self.fc_va = nn.Linear(self.feature_dim, self.memory_dim)
        self.fc_ta = nn.Linear(self.feature_dim, self.memory_dim)

        self.linear_t2i = nn.Linear(opt.embed_size * 2, opt.embed_size)
        self.gate_t2i = nn.Linear(opt.embed_size * 2, opt.embed_size)
        self.linear_i2t = nn.Linear(opt.embed_size * 2, opt.embed_size)
        self.gate_i2t = nn.Linear(opt.embed_size * 2, opt.embed_size)

        if torch.cuda.is_available():
            self.v_memory = self.v_memory.cuda()
            self.t_memory = self.t_memory.cuda()
        self.v_memory.requires_grad = True
        self.t_memory.requires_grad = True

        self.init_weights()

    def init_memory(self):
        memory = Variable(torch.randn(self.memory_size, self.memory_dim))

        return memory

    def init_weights(self):
        self.fc_ve.weight.data.uniform_(-0.1, 0.1)
        self.fc_va.weight.data.uniform_(-0.1, 0.1)
        self.fc_te.weight.data.uniform_(-0.1, 0.1)
        self.fc_ta.weight.data.uniform_(-0.1, 0.1)

    def enhance_feature(self, feature, read_content, lengths=None, vision=True):

        if vision:
            enhanced_feature = self.gated_memory_i2t(feature, read_content)
        else:
            enhanced_feature = self.gated_memory_t2i(feature, read_content, lengths)

        return enhanced_feature

    def calc_weight(self, feature, memory):
        similarity = nn.functional.linear(input=feature, weight=memory)
        wr_weight = nn.functional.softmax(input=similarity, dim=2)

        return wr_weight  # (batch_size, o_num, memory_size)

    def read_memory(self, feature, memory, read_weight=None):
        if read_weight is None:
            read_weight = self.calc_weight(feature=feature, memory=memory)

        read_content = nn.functional.linear(input=read_weight, weight=memory.transpose(0, 1).contiguous(), bias=None)
        read_content = l2norm(read_content, dim=2)

        return read_content

    def write_memory(self, feature, memory, vision, write_weight=None, write_strength=None, cap_lengths=None):
        if write_weight is None:
            print("There is no write_weight!")
            write_weight = self.calc_weight(feature=feature, memory=memory)

        if vision:
            # generate visual memory erase and add operation vectors
            erase_vector = self.fc_ve(feature)
            add_vector = self.fc_va(feature)
        else:
            # generate textual memory erase and add operation vectors
            erase_vector = self.fc_te(feature)
            add_vector = self.fc_ta(feature)

        erase_vector = torch.sigmoid(erase_vector)
        add_vector = l2norm(add_vector)

        if vision:
            num_all = write_strength.size(0) * write_strength.size(1)
        else:
            num_all = torch.sum(cap_lengths)

        if torch.cuda.is_available():
            write_strength.cuda()
            write_weight.cuda()

        erase_info = torch.mul(erase_vector, write_strength)
        erase_info = torch.bmm(write_weight.transpose(1, 2).contiguous().detach(), erase_info)
        erase_info = torch.sum(erase_info, 0) / num_all  # (memory_size, memory_dim)

        add_info = torch.mul(add_vector, write_strength)
        add_info = torch.bmm(write_weight.transpose(1, 2).contiguous().detach(), add_info)
        add_info = torch.sum(add_info, 0) / num_all

        new_memory = memory - (memory * erase_info) + add_info

        return new_memory

    def cal_scores(self, img_fc, v_feature, ht, t_feature, cap_lengths, frag_weights_img=None, frag_weights_cap=None, **kwargs):
        if self.opt.cross_attn == 'both':
            scores_ti, strengths_ti = self.xattn_score_Text_IMRAM(v_feature, v_feature, t_feature, t_feature, cap_lengths, self.opt, frag_weights_cap)
            scores_it, strengths_it = self.xattn_score_Image_IMRAM(v_feature, v_feature, t_feature, t_feature, cap_lengths, self.opt, frag_weights_img)
            scores_ti = torch.stack(scores_ti, 0).sum(0)
            scores_it = torch.stack(scores_it, 0).sum(0)
            scores = scores_it + scores_ti

            return scores, strengths_it, strengths_ti
        else:
            raise ValueError("unknown first norm type:")


    def forward(self, v_feature, img_feature, t_feature, cap_lengths, v_strengths=None, t_strengths=None,
                v_weight=None, t_weight=None, frag_weights_img=None, frag_weights_cap=None, write_signal=False):
        if write_signal:
            # write features back to memory
            self.v_memory_new = self.write_memory(feature=v_feature.detach(), memory=self.v_memory, vision=True,
                                                  write_weight=v_weight.detach(), write_strength=v_strengths.detach())
            self.t_memory_new = self.write_memory(feature=t_feature.detach(), memory=self.t_memory, vision=False,
                                                  write_weight=t_weight.detach(), write_strength=t_strengths.detach(),
                                                  cap_lengths=cap_lengths)

            return "Successfully update memory"

        # calc the read/write_weight
        v_weight = self.calc_weight(feature=v_feature, memory=self.v_memory)
        t_weight = self.calc_weight(feature=t_feature, memory=self.t_memory)

        # read from the memory and enhance the feature representations
        v_content = self.read_memory(feature=v_feature, memory=self.v_memory, read_weight=v_weight)
        t_content = self.read_memory(feature=t_feature, memory=self.t_memory, read_weight=t_weight)
        v_enhanced = self.enhance_feature(feature=v_feature, read_content=v_content, vision=True)
        t_enhanced = self.enhance_feature(feature=t_feature, read_content=t_content, lengths=cap_lengths, vision=False)

        if not self.opt.no_fragment_attn:
            # with local attention mechanism
            scores, strengths_it, strengths_ti = self.cal_scores(v_feature, v_enhanced, t_feature, t_enhanced,
                                                                 cap_lengths, frag_weights_img.cuda(), frag_weights_cap.cuda())
        else:
            # without local attention mechanism
            scores, strengths_it, strengths_ti = self.cal_scores(v_feature, v_enhanced, t_feature, t_enhanced, cap_lengths)

        return scores, v_enhanced, t_enhanced, v_weight, t_weight, strengths_it, strengths_ti

    def xattn_score_Text_IMRAM(self, images_fc, images, caption_ht, captions_all, cap_lens, opt, fragment_attn=None, get_weight=True):
        """
        We changed the codes from open codes offered by
        """
        similarities = [[] for _ in range(opt.iteration_step)]
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        images = images.float()
        captions_all = captions_all.float()
        caption_ht = caption_ht.float()
        images_fc = images.mean(1, keepdim=True)
        write_strengths = []
        max_num = int(torch.LongTensor(cap_lens).max(0)[0].item())

        for i in range(n_caption):
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            query = cap_i_expand
            context = images
            weight = 0
            for j in range(opt.iteration_step):
                attn_feat, _ = func_attention(query, context, opt, smooth=opt.lambda_softmax)
                row_sim = cosine_similarity(cap_i_expand, attn_feat, dim=2)

                if get_weight:
                    row_strength = torch.zeros(1, max_num)
                    if opt.no_self_regulating:
                        row_strength[0, :n_word] = torch.ones(row_sim[i].shape)
                    else:
                        if opt.regulate_way == 'sigmoid':
                            row_strength[0, :n_word] = 1. - torch.sigmoid(row_sim[i, :] * 3)
                        elif opt.regulate_way == 'clamp':
                            row_strength[0, :n_word] = (1. - row_sim[i, :].clamp(max=1.))
                        elif opt.regulate_way == 'softmax':
                            row_strength[0, :n_word] = 1. - nn.functional.softmax(input=row_sim, dim=1)[i, :]
                    write_strengths.append(row_strength)
                if (not opt.no_fragment_attn) and (fragment_attn is not None):
                    row_sim = row_sim * fragment_attn[i][:n_word].repeat(n_image, 1) * n_word
                    row_sim = row_sim.mean(dim=1, keepdim=True)
                else:
                    row_sim = row_sim.mean(dim=1, keepdim=True)
                similarities[j].append(row_sim)

        new_similarities = []
        for j in range(opt.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1).double()
            if self.training:
                similarities_one = similarities_one.transpose(0, 1)
            new_similarities.append(similarities_one)
        if get_weight:
            write_strengths = torch.cat(write_strengths, 0).unsqueeze(-1).contiguous()

            return new_similarities, write_strengths

        return new_similarities

    def xattn_score_Image_IMRAM(self, images_fc, images, caption_ht, captions_all, cap_lens, opt, fragment_attn=None, get_weight=True):
        """
        We changed the codes from open codes offered by
        """
        similarities = [[] for _ in range(opt.iteration_step)]
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        n_region = images.size(1)
        images = images.float()
        captions_all = captions_all.float()
        caption_ht = caption_ht.float()
        images_fc = images.mean(1, keepdim=True)
        write_strengths = []
        for i in range(n_caption):
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            query = images
            context = cap_i_expand
            weight = 0
            for j in range(opt.iteration_step):
                attn_feat, _ = func_attention(query, context, opt, smooth=opt.lambda_softmax)
                row_sim = cosine_similarity(images, attn_feat, dim=2)

                if get_weight:
                    if opt.no_self_regulating:
                        row_strength = torch.ones(row_sim[i].shape)
                    else:
                        if opt.regulate_way == 'sigmoid':
                            row_strength = 1. - torch.sigmoid(row_sim[i, :] * 3)
                        elif opt.regulate_way == 'clamp':
                            row_strength = (1. - row_sim[i, :].clamp(max=1.))
                        elif opt.regulate_way == 'softmax':
                            row_strength = 1. - nn.functional.softmax(input=row_sim, dim=1)[i, :]

                    row_strength = row_strength.unsqueeze(0).contiguous()
                    write_strengths.append(row_strength)
                if (not opt.no_fragment_attn) and (fragment_attn is not None):
                    row_sim = row_sim * fragment_attn * 36
                    row_sim = row_sim.mean(dim=1, keepdim=True)
                else:
                    row_sim = row_sim.mean(dim=1, keepdim=True)
                similarities[j].append(row_sim)

        new_similarities = []
        for j in range(opt.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1).double()
            if self.training:
                similarities_one = similarities_one.transpose(0, 1)
            new_similarities.append(similarities_one)
        if get_weight:
            write_strengths = torch.cat(write_strengths, 0).unsqueeze(-1).contiguous()

            return new_similarities, write_strengths

        return new_similarities

    def gated_memory_t2i(self, input_0, input_1, lengths):
        input_cat = torch.cat([input_0, input_1], 2)
        gate = nn.functional.tanh(nn.functional.relu(self.linear_i2t(input_cat)))
        output = input_0 + input_1 * gate
        output = l2norm(output, dim=2)

        for i in range(1, len(lengths)):
            output[i][lengths[i]:][:] = 0.

        return output

    def gated_memory_i2t(self, input_0, input_1):
        input_cat = torch.cat([input_0, input_1], 2)
        gate = nn.functional.tanh(nn.functional.relu(self.linear_i2t(input_cat)))
        output = input_0 + input_1 * gate
        output = l2norm(output, dim=2)

        return output


class FSCMM(object):
    """
    Few-Shot Cross-Modal Matching model ---- LFSRM
    """

    def __init__(self, opt, v_memory=None, t_memory=None):
        # Build Models
        self.opt = opt
        self.loss_detail = opt.loss_detail
        self.grad_clip = opt.grad_clip
        self.img_encoder = ImageEncoder(opt, opt.img_dim, opt.embed_size, use_abs=opt.use_abs, no_imgnorm=opt.no_img_norm)
        self.txt_encoder = TextEncoder(opt, opt.word_dim, opt.embed_size, opt.num_layers, use_gru=opt.use_gru,
                                       bi_gru=opt.bi_gru, no_txt_norm=opt.no_txt_norm)
        self.cm_memory = CMMemory(self.opt, opt.memory_size, opt.memory_dim, v_memory, t_memory)
        self.criterion = ContrastiveLoss(self.opt, opt.cross_attn, opt.loss_detail, margin=opt.margin,
                                         max_violation=opt.max_violation)

        if torch.cuda.is_available():
            self.img_encoder.cuda()
            self.txt_encoder.cuda()
            self.cm_memory.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        params = list(self.img_encoder.parameters())
        params += list(self.txt_encoder.parameters())
        if opt.finetune:
            params += list(self.img_encoder.cnn.parameters())
        memory_params = list(self.cm_memory.parameters())

        self.params = params
        self.memory_params = memory_params

        self.optimizer = torch.optim.Adam(self.params, lr=opt.init_lr)
        self.optimizer_memory = torch.optim.Adam(self.memory_params, lr=opt.init_lr)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_encoder.state_dict(), self.txt_encoder.state_dict(), self.cm_memory.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_encoder.load_state_dict(state_dict[0])
        self.txt_encoder.load_state_dict(state_dict[1])
        self.cm_memory.load_state_dict(state_dict[2])

    def train_mode(self):
        """
        switch to train mode
        """
        self.img_encoder.train()
        self.txt_encoder.train()
        self.cm_memory.train()
        self.cm_memory.v_memory.requires_grad = True
        self.cm_memory.t_memory.requires_grad = True

    def eval_mode(self):
        """
        switch to evaluate mode
        """
        self.img_encoder.eval()
        self.txt_encoder.eval()
        self.cm_memory.eval()
        self.cm_memory.v_memory.requires_grad = False
        self.cm_memory.t_memory.requires_grad = False

    def forward_mem(self, images, captions, lengths, volatile=False):
        """
        Process on image features and sentence features; Calculate the similarity scores
        :return: similarity scores and corresponding enhanced features after memory module
        """

        # set mini-batch
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)

        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        if not self.opt.no_fragment_attn:
            # with local fragment attention mechanism
            GCN_img_features, img_features, frag_weights_img = self.img_encoder(images)
            cap_feature, ht_embeddings, cap_lengths, frag_weights_cap = self.txt_encoder(captions, lengths)
            scores, v_enhanced, t_enhanced, v_weight, t_weight, strengths_it, strengths_ti = self.cm_memory(
                GCN_img_features, img_features, cap_feature, cap_lengths, frag_weights_img=frag_weights_img,
                frag_weights_cap=frag_weights_cap, write_signal=False)

            return scores, v_enhanced, GCN_img_features, t_enhanced, cap_feature, cap_lengths, v_weight, t_weight, strengths_it.cuda(), strengths_ti.cuda(), frag_weights_img.cuda(), frag_weights_cap.cuda()
        else:
            # without local fragment attention mechanism
            GCN_img_features, img_features = self.img_encoder(images)
            cap_feature, ht_embeddings, cap_lengths = self.txt_encoder(captions, lengths)
            scores, v_enhanced, t_enhanced, v_weight, t_weight, strengths_it, strengths_ti = self.cm_memory(
                GCN_img_features, img_features, cap_feature, cap_lengths, write_signal=False)

            return scores, v_enhanced, GCN_img_features, t_enhanced, cap_feature, cap_lengths, v_weight, t_weight, strengths_it.cuda(), strengths_ti.cuda()

    def write_mem(self, img_feature, cap_feature, cap_lengths, v_strengths, t_strengths, v_weight, t_weight,
                  volatile=False):
        """
        Memorize uncommon features which the model is unfamiliar with at now state
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        v_strengths.to(device)
        t_strengths.to(device)
        v_weight.to(device)
        t_weight.to(device)
        response = self.cm_memory(img_feature, img_feature, cap_feature, cap_lengths, v_strengths, t_strengths,
                                  v_weight, t_weight, write_signal=True)

        return response

    def forward_loss(self, scores):
        """
        Compute loss from scores
        """
        loss = self.criterion(scores)
        self.logger.update('Le', loss.item())

        return loss

    def train_step(self, images, captions, lengths, ids=None, *args):
        """
        One training step given images and captions (Or stored pretrained features)
        """
        self.Eiters += 1
        self.logger.update('Ei', self.Eiters)
        self.logger.update('Lr', self.optimizer.param_groups[0]['lr'])

        self.optimizer.zero_grad()
        self.optimizer_memory.zero_grad()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cm_memory.v_memory.grad = torch.zeros(self.cm_memory.memory_size, self.cm_memory.memory_dim).to(device)
        self.cm_memory.t_memory.grad = torch.zeros(self.cm_memory.memory_size, self.cm_memory.memory_dim).to(device)

        start_time = time.time()
        if not self.opt.no_fragment_attn:
            scores, v_enhanced, img_features, t_enhanced, cap_feature, cap_lengths, v_weight, t_weight, strengths_it, strengths_ti, fragment_attn_img, fragment_attn_cap = self.forward_mem(images, captions, lengths)
        else:
            scores, v_enhanced, img_features, t_enhanced, cap_feature, cap_lengths, v_weight, t_weight, strengths_it, strengths_ti = self.forward_mem(images, captions, lengths)

        forward_mem_time = time.time()
        torch.autograd.set_detect_anomaly(True)

        loss = self.forward_loss(scores)
        forward_loss_time = time.time()

        # write features back to memory
        response = self.write_mem(img_features, cap_feature, cap_lengths, strengths_it, strengths_ti,
                                  v_weight, t_weight)
        write_time = time.time()

        self.cm_memory.v_memory.data = self.cm_memory.v_memory_new.data
        self.cm_memory.t_memory.data = self.cm_memory.t_memory_new.data
        self.cm_memory.v_memory_new.grad = self.cm_memory.v_memory.grad
        self.cm_memory.t_memory_new.grad = self.cm_memory.t_memory.grad

        if self.loss_detail:
            sys.stdout.write("\rEiters: {}  Loss: {}".format(self.Eiters, loss.item()))

        loss.backward(retain_graph=True)
        self.cm_memory.v_memory_new.backward(self.cm_memory.v_memory_new.grad, retain_graph=True)
        self.cm_memory.t_memory_new.backward(self.cm_memory.t_memory_new.grad, retain_graph=True)
        loss_bk_time = time.time()

        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
            clip_grad_norm(self.memory_params, self.grad_clip)
        # grad_clip_time = time.time()

        self.optimizer.step()
        self.optimizer_memory.step()

        sys.stdout.write("\rLoss: %.8s  forward_mem: %.8s   forward_loss: %.8s   loss_bk: %.8s" % (loss.item(), forward_mem_time - start_time,
                    forward_loss_time - forward_mem_time, loss_bk_time - write_time))
