import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticExtractor(nn.Module):
    """
    Base class for semantic extractors
    """

    def __init__(self, n_class, dims, layers, type="", **kwargs):
        """
        Args:
          n_class : The number of semantic categories.
          dims : The dimension (depth) of each feature map.
          layers : The layer indice of the generator.
        """
        super().__init__()

        self.type = type
        self.n_class = n_class
        self.dims = dims
        self.layers = layers
        self.segments = [0] + list(np.cumsum(self.dims))
        self.build()

    def _index_feature(self, features, i):
        """"""
        l1 = len(self.layers)
        l2 = len(features)
        if l1 < l2:
            return features[self.layers[i]]
        elif l1 == l2:
            return features[i]
        else:
            print(f"!> Error: The length of layers ({l1}) != features ({l2})")

    def predict(self, stage):
        """Return a numpy array."""
        res = self.forward(stage, True)[0].argmax(1)
        return res.detach().cpu().numpy().astype("int32")

    def arch_info(self):
        """Return the architecture information dict."""
        return dict(
            n_class=self.n_class,
            dims=self.dims,
            layers=self.layers)

    def build(self):
        pass


class LSE(SemanticExtractor):
    """Extract the semantics from generator's feature maps using 1x1 convolution.
    """

    def __init__(self, lw_type="softplus", use_bias=False, **kwargs):
        """
        Args:
          lw_type : The layer weight type. Candidates are softplus, sigmoid, none.
          use_bias : default is not to use bias.
        """
        self.lw_type = lw_type
        self.use_bias = use_bias
        super().__init__(**kwargs)
        self.build()

    def build(self):
        """Build the architecture of LSE."""

        def conv_block(in_dim, out_dim):
            return nn.Conv2d(in_dim, out_dim, 1, bias=self.use_bias)

        self.extractor = nn.ModuleList([
            conv_block(dim, self.n_class) for dim in self.dims])

        self.layer_weight = nn.Parameter(torch.ones((len(self.layers),)))

    def arch_info(self):
        base_dic = SemanticExtractor.arch_info(self)
        base_dic["lw_type"] = self.lw_type
        base_dic["use_bias"] = self.use_bias
        base_dic["type"] = "LSE"
        return base_dic

    def _calc_layer_weight(self):
        if self.lw_type == "none" or self.lw_type == "direct":
            return self.layer_weight
        if self.lw_type == "softplus":
            weight = torch.nn.functional.softplus(self.layer_weight)
        elif self.lw_type == "sigmoid":
            weight = torch.sigmoid(self.layer_weight)
        return weight / weight.sum()

    def forward(self, features, size=None):
        """Given a set of features, return the segmentation.
        Args:
          features : A list of feature maps. When len(features) > len(self.layers)
                     , it is assumed that the features if taken from all the layers
                     from the generator and will be selected here. Otherwise, it is
                     assumed that the features correspond to self.layers.
          size : The target output size. Bilinear resize will be used if this
                 argument is specified.
        Returns:
          A list of segmentations corresponding to each layer, with the last one
          being the final segmentation integrating all layers.
        """

        outputs = []
        for i in range(len(self.layers)):
            feat = self._index_feature(features, i)
            x = self.extractor[i](feat)
            outputs.append(x)

        # detect final output size, if not specified
        size = size if size else outputs[-1].shape[2:]
        layers = BU(outputs, size)

        weight = self._calc_layer_weight()
        if self.lw_type == "none":
            final = sum(layers)
        else:
            final = sum([r * w for r, w in zip(layers, weight)])
        outputs.append(final)
        return outputs


class NSE1(LSE):
    """A direct nonlinear generalization from LSE.
    """

    def __init__(self, lw_type="softplus", use_bias=True,
                 ksize=1, n_layers=3, **kwargs):
        """
        Args:
          ksize : The convolution kernel size.
        """
        self.lw_type = lw_type
        self.use_bias = use_bias
        self.ksize = ksize
        self.n_layers = n_layers
        SemanticExtractor.__init__(self, **kwargs)
        self.build()

    def arch_info(self):
        base_dic = SemanticExtractor.arch_info(self)
        base_dic["ksize"] = self.ksize
        base_dic["n_layers"] = self.n_layers
        base_dic["lw_type"] = self.lw_type
        base_dic["use_bias"] = self.use_bias
        base_dic["type"] = "NSE-1"
        return base_dic

    def build(self):
        def conv_block(in_dim, out_dim):
            midim = (in_dim + out_dim) // 2
            padding = {1: 0, 3: 1}[self.ksize]
            _m = []
            _m.append(nn.Conv2d(in_dim, midim, self.ksize, padding=padding))
            _m.append(nn.ReLU(inplace=True))

            for _ in range(self.n_layers - 2):
                _m.append(nn.Conv2d(midim, midim, self.ksize, padding=padding))
                _m.append(nn.ReLU(inplace=True))

            _m.append(nn.Conv2d(midim, out_dim, self.ksize, padding=padding))
            return nn.Sequential(*_m)

        self.extractor = nn.ModuleList([
            conv_block(dim, self.n_class)
            for dim in self.dims])

        # combining result from each layer
        self.layer_weight = nn.Parameter(torch.ones((len(self.dims),)))

    def forward(self, features, size=None):
        return super().forward(features, size)


class NSE2(SemanticExtractor):
    """A generator-like semantic extractor."""

    def __init__(self, ksize=3, **kwargs):
        """
          Args:
            ksize: kernel size of convolution
        """
        self.type = "NSE-2"
        self.ksize = ksize
        super().__init__(**kwargs)

    def arch_info(self):
        base_dic = SemanticExtractor.arch_info(self)
        base_dic["ksize"] = self.ksize
        base_dic["type"] = "NSE-2"
        return base_dic

    def build(self):
        def conv_block(in_dim, out_dim):
            _m = [
                nn.Conv2d(in_dim, out_dim, self.ksize, 1, self.ksize // 2),
                nn.ReLU(inplace=True)]
            return nn.Sequential(*_m)

        # transform generative representation to semantic embedding
        self.extractor = nn.ModuleList([
            conv_block(dim, dim) for dim in self.dims])
        # learning residue between different layers
        self.reviser = nn.ModuleList([conv_block(prev, cur) \
                                      for prev, cur in zip(self.dims[:-1], self.dims[1:])])
        # transform semantic embedding to label
        self.visualizer = nn.Conv2d(self.dims[-1], self.n_class, self.ksize,
                                    padding=self.ksize // 2)

    def forward(self, features, size=None):
        for i in range(len(self.layers)):
            feat = self._index_feature(features, i)
            if i == 0:
                hidden = self.extractor[i](feat)
            else:
                if hidden.size(2) * 2 == feat.size(2):
                    hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
                hidden = self.reviser[i - 1](hidden)
                hidden = hidden + self.extractor[i](feat)
        x = self.visualizer(hidden)
        if size is not None and size != x.size(3):
            x = BU(x, size)
        return [x]


EXTRACTOR_POOL = {
    "LSE": LSE,
    "NSE-1": NSE1,
    "NSE-2": NSE2
}


def BU(img, size, align_corners=True):
    """Bilinear interpolation with Pytorch.
    Args:
      img : a list of tensors or a tensor.
    """
    if type(img) is list:
        return [F.interpolate(i,
                              size=size, mode='bilinear', align_corners=align_corners)
                for i in img]
    return F.interpolate(img,
                         size=size, mode='bilinear', align_corners=align_corners)
