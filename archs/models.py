import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune

from .layers import layer_dict, ThrInPrune, LayerNumLogger, rewind


kwargs_default = {
    'gcn': {
        'cached': False,
        'add_self_loops': False,
        'improved': False,
        'normalize': False,
        'rnorm': 0.5,
        'diag': 1.,
        'depth_inv': False,
    },
    'gin': {
        'eps': 0.0,
        'train_eps': False,
        'rnorm': None,
        'diag': 1.,
        'depth_inv': False,
    },
    'gat': {
        'heads': 8,
        'concat': True,
        'share_weights': False,
        'add_self_loops': False,
        'rnorm': None,
        'diag': 1.,
        'depth_inv': False,
    },
    'gcn2': {
        'alpha': 0.1,
        'theta': 0.5,
        'shared_weights': True,
        'cached': False,
        'add_self_loops': False,
        'normalize': False,
        'rnorm': 0.5,
        'diag': 1.,
        'depth_inv': True,
    },
    'gsage': {
        'aggr': 'mean',
        'improved': False,
        'normalize': False,
        'root_weight': True,
        'project': False,
        'bias': True,
        'rnorm': 0.5,
        'diag': 1.,
        'depth_inv': False,
    },
}


def state2module(model, param_name):
    parts = param_name.split('.')
    module = model
    for part in parts[:-1]:
        module = getattr(module, part)
    return module


def set_attr(module, k, v):
    if hasattr(module, k):
        setattr(module, k, v)


class GNNThr(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass,
                 thr_a=0.0, thr_w=0.0, dropout: float = 0.0, layer: str = 'gcn',
                 **kwargs,):
        super(GNNThr, self).__init__()
        self.apply_thr = ('_' in layer)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()
        self.use_bn = True
        self.kwargs = kwargs

        # Select conv layer
        Conv = layer_dict[layer]
        # Set layer args
        for k, v in kwargs_default[layer.split('_')[0]].items():
            self.kwargs.setdefault(k, v)
        norm_kwargs = {'affine': True, 'track_running_stats': True, 'momentum': 0.9}
        if not isinstance(thr_a, list):
            if layer.endswith('_rnd'):
                thr_a = [float(thr_a)] + [0.0] * (nlayer - 1)
            else:
                thr_a = [float(thr_a)] * nlayer
        thr_w = thr_w if isinstance(thr_w, list) else [float(thr_w)] * nlayer

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        depths = list(range(nlayer))
        if not self.kwargs.pop('depth_inv'):
            depths = list(reversed(depths))
        self.convs.append(Conv(nfeat, nhidden,
                               depth=depths[0], thr_a=thr_a[0], thr_w=thr_w[0],
                               **self.kwargs))
        self.norms.append(nn.BatchNorm1d(nhidden, **norm_kwargs))
        for layer in range(1, nlayer - 1):
            self.convs.append(Conv(nhidden, nhidden,
                                   depth=depths[layer], thr_a=thr_a[layer], thr_w=thr_w[layer],
                                   **self.kwargs))
            self.norms.append(nn.BatchNorm1d(nhidden, **norm_kwargs))
        self.convs.append(Conv(nhidden, nclass,
                               depth=depths[nlayer-1], thr_a=thr_a[nlayer-1], thr_w=thr_w[nlayer-1],
                               **self.kwargs))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for norm in self.norms:
            norm.reset_parameters()

    def forward(self, x, edge_idx, node_lock=torch.Tensor([]), verbose=False):
        if self.apply_thr:
            # Layer inheritence of edge_idx
            for i, conv in enumerate(self.convs[:-1]):
                x, edge_idx = conv(x, edge_idx, node_lock=node_lock, verbose=verbose)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x, _ = self.convs[-1](x, edge_idx, node_lock=node_lock, verbose=verbose)
            return x
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, edge_idx)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x = self.convs[-1](x, edge_idx)
            return x

    def get_repre(self, x, edge_idx, layer=None, node_lock=torch.Tensor([]), verbose=False):
        layer = layer if layer is not None else len(self.convs)-1
        if self.apply_thr:
            for i, conv in enumerate(self.convs[:layer]):
                x, edge_idx = conv(x, edge_idx, node_lock=node_lock, verbose=verbose)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x, _ = self.convs[layer](x, edge_idx, node_lock=node_lock, verbose=verbose)
            return x
        else:
            for i, conv in enumerate(self.convs[:layer]):
                x = conv(x, edge_idx)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x = self.convs[layer](x, edge_idx)
            return x

    def set_scheme(self, scheme_a, scheme_w):
        self.apply(lambda module: set_attr(module, 'scheme_a', scheme_a))
        self.apply(lambda module: set_attr(module, 'scheme_w', scheme_w))

    def remove(self):
        for conv in self.convs:
            if hasattr(conv, 'prune_lst'):
                for module in conv.prune_lst:
                    if prune.is_pruned(module):
                        prune.remove(module, 'weight')

    def get_numel(self):
        numel_a = sum([conv.logger_a.numel_after for conv in self.convs])
        numel_w = sum([conv.logger_w.numel_after for conv in self.convs])
        return numel_a/1e3, numel_w/1e3

    @classmethod
    def batch_counter_hook(cls, module, input, output):
        if not hasattr(module, '__batch_counter__'):
            module.__batch_counter__ = 0
        module.__batch_counter__ += 1


class GNNLPThr(GNNThr):
    def __init__(self, nlayer, nfeat, nhidden, nclass,
                 thr_a=0.0, thr_w=0.0, dropout: float = 0.0, layer: str = 'gcn',
                 **kwargs,):
        super(GNNLPThr, self).__init__(nlayer, nfeat, nhidden, nhidden,
                                       thr_a=thr_a, thr_w=thr_w, dropout=dropout, layer=layer,
                                       **kwargs)
        self.lin_out = nn.ModuleList([
            nn.Linear(nhidden, nhidden),
            nn.Linear(nhidden, nhidden),
            nn.Linear(nhidden, 1),
        ])

    def decode(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lin_out[:-1]:
            x = lin(x)
            x = self.act(x)
            x = self.dropout(x)
        x = self.lin_out[-1](x)
        return x


class SandwitchThr(GNNThr):
    def __init__(self, nlayer, nfeat, nhidden, nclass,
                 thr_a=0.0, thr_w=0.0, dropout: float = 0.0, layer: str = 'gcn',
                 **kwargs,):
        super(SandwitchThr, self).__init__(nlayer, nhidden, nhidden, nhidden,
                                           thr_a=thr_a, thr_w=thr_w, dropout=dropout, layer=layer,
                                           **kwargs)
        self.lin_in = nn.Linear(nfeat, nhidden)
        self.lin_out = nn.Linear(nhidden, nclass)
        norm_kwargs = {'affine': True, 'track_running_stats': True, 'momentum': 0.9}
        self.norms.append(nn.BatchNorm1d(nhidden, **norm_kwargs))

    def forward(self, x, edge_idx, node_lock=torch.Tensor([]), verbose=False):
        x = self.lin_in(x)
        if self.use_bn:
            x = self.norms[-1](x)
        x = x_0 = self.act(x)
        x = self.dropout(x)

        if self.apply_thr:
            # Layer inheritence of edge_idx
            for i, conv in enumerate(self.convs[:-1]):
                x, edge_idx = conv(x, x_0, edge_idx, node_lock=node_lock, verbose=verbose)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x, _ = self.convs[-1](x, x_0, edge_idx, node_lock=node_lock, verbose=verbose)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, x_0, edge_idx)
                if self.use_bn:
                    x = self.norms[i](x)
                x = self.act(x)
                x = self.dropout(x)
            x = self.convs[-1](x, x_0, edge_idx)

        x = self.lin_out(x)
        return x


class MLP(nn.Module):
    def __init__(self, nlayer, nfeat, nhidden, nclass, dropout, thr_w=0.0, layer: str = 'sgc'):
        super(MLP, self).__init__()
        fbias = True
        self.fbn = True
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU(inplace=True)
        self.nfeat = nfeat
        self.nhidden = nhidden
        self.nclass = nclass
        self.algo = layer
        self.threshold_w = thr_w
        self.scheme_w = 'full'
        self.kwargs = {'rnorm': 0.5, 'diag': 1.}

        self.fcs = nn.ModuleList()
        if self.fbn: self.bns = nn.ModuleList()

        if nlayer == 1:
            self.fcs.append(nn.Linear(nfeat, nclass, bias=fbias))
        else:
            self.fcs.append(nn.Linear(nfeat, nhidden, bias=fbias))
            if self.fbn: self.bns.append(nn.BatchNorm1d(nhidden))
            for _ in range(nlayer - 2):
                self.fcs.append(nn.Linear(nhidden, nhidden, bias=fbias))
                if self.fbn: self.bns.append(nn.BatchNorm1d(nhidden))
            self.fcs.append(nn.Linear(nhidden, nclass, bias=fbias))
        for fc in self.fcs:
            fc.logger_w = LayerNumLogger(layer)

    def reset_parameters(self):
        for lin in self.fcs:
            lin.reset_parameters()
        if self.fbn:
            for bn in self.bns:
                bn.reset_parameters()

    def apply_prune(self, lin, x):
        logger_w = lin.logger_w
        if '_' in self.algo:
            if self.scheme_w in ['pruneall', 'pruneinc']:
                if self.scheme_w == 'pruneall':
                    if prune.is_pruned(lin):
                        rewind(lin, 'weight')
                if self.algo.endswith('_thr'):
                    norm_node_in = torch.norm(x, dim=0)
                    norm_all_in = torch.norm(norm_node_in, dim=None)/x.shape[1]
                    if norm_all_in > 1e-8:
                        threshold_wi = self.threshold_w * norm_all_in / norm_node_in
                        ThrInPrune.apply(lin, 'weight', threshold_wi)
                else:
                    pass
            elif self.scheme_w == 'keep':
                pass
            elif self.scheme_w == 'full':
                raise NotImplementedError()
            logger_w.numel_before = lin.weight.numel()
            logger_w.numel_after = torch.sum(lin.weight != 0).item()
        else:
            logger_w.numel_before = lin.weight.numel()
            logger_w.numel_after = lin.weight.numel()

    def forward(self, x, edge_idx, *args, **kwargs):
        # import torch_geometric
        # deg_inv = 1./ torch_geometric.utils.degree(edge_idx[0][0], x.shape[0])
        for i, fc in enumerate(self.fcs[:-1]):
            self.apply_prune(fc, x)
            x = fc(x)
            # x = x * deg_inv.unsqueeze(1)
            x = self.act(x)
            if self.fbn: x = self.bns[i](x)
            x = self.dropout(x)
        fc = self.fcs[-1]
        self.apply_prune(fc, x)
        x = fc(x)
        return x

    def set_scheme(self, scheme_a, scheme_w):
        self.scheme_w = scheme_w

    def get_numel(self):
        numel_w = sum([fc.logger_w.numel_after for fc in self.fcs])
        return 0, numel_w/1e3
