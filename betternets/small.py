from math import sqrt

import torch

from score_sde.models import up_or_down_sampling
from score_sde.models.dense_layer import dense
from score_sde.models.discriminator import conv2d
from score_sde.models.layerspp import AdaptiveGroupNorm
from score_sde.models.ncsnpp_generator_adagn import PixelNorm
from util import count_parameters


class LinearOnChannel(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        self.lin = dense(in_features, out_features, bias=bias)

    def forward(self, x):
        """
        :param x:   (batsize, numchannels, height, width)
        :return:
        """
        x = x.permute(0, 2, 3, 1)
        y = self.lin(x)
        return y.permute(0, 3, 1, 2)


class MHSelfAttn(torch.nn.Module):
    def __init__(self, dim=64, numheads=2, skip_rescale=True,
                 add_pos=False, height=-1, width=-1, **kw):
        super().__init__(**kw)
        self.dim, self.numheads, self.skip_rescale = dim, numheads, skip_rescale

        self.groupnorm = torch.nn.GroupNorm(num_groups=min(self.dim // 4, 32), num_channels=self.dim, eps=1e-6)
        self.keymap = LinearOnChannel(self.dim, self.dim)
        self.querymap = LinearOnChannel(self.dim, self.dim)
        self.valuemap = LinearOnChannel(self.dim, self.dim)
        self.postmap = LinearOnChannel(self.dim, self.dim)

        self.add_pos = add_pos
        if self.add_pos:
            self.posemb = LearnedPositionalEmbeddings2D(height, width, self.dim)

    def forward(self, x):
        batsize, numchannels, height, width = x.shape

        _x = self.groupnorm(x)
        __x = self.posemb(_x) if self.add_pos else _x

        q = self.querymap(__x)   # (batsize, dim, height, width)
        k = self.keymap(__x)         # (batsize, dim, height, width)
        v = self.valuemap(_x)       # (batsize, dim, height, width)

        q = q.view(batsize, self.numheads, self.dim // self.numheads, height, width)
        k = k.view(batsize, self.numheads, self.dim // self.numheads, height, width)
        v = v.view(batsize, self.numheads, self.dim // self.numheads, height, width)

        attn_scores = torch.einsum("bhdxy,bhdij->bhxyij", q, k) / sqrt(self.dim)
        attn_scores = attn_scores.view(batsize, self.numheads, height, width, -1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.view(batsize, self.numheads, height, width, height, width)

        h = torch.einsum("bhxyij,bhdij->bhdxy", attn_weights, v)
        h = h.reshape(batsize, self.dim, height, width)
        h = self.postmap(h)

        ret = x + h
        if self.skip_rescale:
            ret /= sqrt(2.)
        return ret


class LearnedPositionalEmbeddings2D(torch.nn.Module):
    def __init__(self, height, width, dim):
        super().__init__()
        self.height, self.width, self.dim = height, width, dim
        self.xemb = torch.nn.Embedding(self.height, dim)
        self.yemb = torch.nn.Embedding(self.width, dim)

    def forward(self, x):
        posemb = self.xemb.weight[:, None, :] + self.yemb.weight[None, :, :]
        ret = posemb.permute(2, 0, 1)[None, :, :, :] + x
        return ret


class LearnedPositionalEmbeddings2DIndep(torch.nn.Module):
    def __init__(self, height, width, dim):
        super().__init__()
        self.height, self.width, self.dim = height, width, dim
        self.size = height * width
        self.emb = torch.nn.Embedding(self.size, dim)

    def forward(self, x):
        posemb = self.emb.weight.view(self.height, self.width, self.dim).permute(2, 0, 1)
        ret = posemb[None, :, :, :] + x
        return ret


class TransformerLayerSelfAttn(torch.nn.Module):
    def __init__(self, dim=64, numheads=2, skip_rescale=True, act=torch.nn.GELU()):
        super().__init__()
        self.dim, self.numheads, self.skip_rescale = dim, numheads, skip_rescale
        self.dim_ff = self.dim * 2

        self.groupnorm = torch.nn.GroupNorm(num_groups=min(dim // 4, 32), num_channels=dim, eps=1e-6)
        self.linA = dense(self.dim, self.dim_ff)
        self.linB = dense(self.dim_ff, self.dim)
        self.act = act

        self.mhsa = MHSelfAttn(dim=self.dim, numheads=self.numheads, skip_rescale=skip_rescale)

    def forward(self, x):
        """
        :param x:  (batsize, numchannel, height, width)
        :return:
        """
        h = self.mhsa(x)

        _h = self.groupnorm(h)
        g = self.linA(_h)
        g = self.act(g)
        g = self.linB(g)

        ret = g + h
        if self.skip_rescale:
            ret /= sqrt(2)
        return ret


class MHCrossAttn(torch.nn.Module):
    """ Pulls info from input A to input B using attention """
    def __init__(self, dimA=64, dimB=64, dim=64, numheads=2, skip_rescale=True,
                 add_pos=False, Asize=-1, Bsize=-1, **kw):
        super().__init__(**kw)
        self.dim, self.dimA, self.dimB, self.numheads, self.skip_rescale \
            = dim, dimA, dimB, numheads, skip_rescale

        self.querymap = LinearOnChannel(self.dimB, self.dim)
        self.keymap = LinearOnChannel(self.dimA, self.dim)
        self.valuemap = LinearOnChannel(self.dimA, self.dim)
        self.postmap = LinearOnChannel(self.dim, self.dimB)

        self.groupnorm = torch.nn.GroupNorm(num_groups=min(self.dimA // 4, 32), num_channels=self.dimA, eps=1e-6)
        self.groupnorm2 = torch.nn.GroupNorm(num_groups=min(self.dimB // 4, 32), num_channels=self.dimB, eps=1e-6)

        self.add_pos = add_pos
        if self.add_pos:
            assert False, "don't use this code"
            self.posA = LearnedPositionalEmbeddings2D(Asize, Asize, self.dimA)
            self.posB = LearnedPositionalEmbeddings2D(Bsize, Bsize, self.dimB)

    def forward(self, a, b):
        batsize, numchannels, height, width = a.shape
        _, _, height2, width2 = b.shape

        skip_b = b

        a = self.groupnorm(a)
        b = self.groupnorm2(b)

        _a = self.posA(a) if self.add_pos else a
        _b = self.posB(b) if self.add_pos else b

        q = self.querymap(_b)   # (batsize, dim, height2, width2)
        k = self.keymap(_a)         # (batsize, dim, height, width)
        v = self.valuemap(a)       # (batsize, dim, height, width)

        q = q.view(batsize, self.numheads, self.dim // self.numheads, height2, width2)
        k = k.view(batsize, self.numheads, self.dim // self.numheads, height, width)
        v = v.view(batsize, self.numheads, self.dim // self.numheads, height, width)

        attn_scores = torch.einsum("bhdxy,bhdij->bhxyij", q, k) / sqrt(self.dim)
        attn_scores = attn_scores.view(batsize, self.numheads, height2, width2, -1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = attn_weights.view(batsize, self.numheads, height2, width2, height, width)

        h = torch.einsum("bhxyij,bhdij->bhdxy", attn_weights, v)
        h = h.reshape(batsize, self.dim, height2, width2)
        h = self.postmap(h)

        ret = skip_b + h
        if self.skip_rescale:
            ret /= sqrt(2.)
        return ret


class TransformerLayerCrossAttn(torch.nn.Module):
    def __init__(self, dim=64, dim_cells=64, numheads=2, skip_rescale=True,
                 add_pos=False, imgsize=-1, memsize=-1,
                 act=torch.nn.GELU()):
        super().__init__()
        self.dim, self.dim_cells, self.numheads, self.skip_rescale \
            = dim, dim_cells, numheads, skip_rescale
        self.dim_ff = self.dim * 2
        self.dim_ff_cells = self.dim_cells * 2

        self.groupnorm_x = torch.nn.GroupNorm(num_groups=min(dim // 4, 32), num_channels=dim, eps=1e-6)
        self.linA_x = torch.nn.Linear(self.dim, self.dim_ff)
        self.linB_x = torch.nn.Linear(self.dim_ff, self.dim)
        self.groupnorm_cells = torch.nn.GroupNorm(num_groups=min(dim_cells // 4, 32), num_channels=dim_cells, eps=1e-6)
        self.linA_cells = torch.nn.Linear(self.dim, self.dim_ff)
        self.linB_cells = torch.nn.Linear(self.dim_ff, self.dim_cells)
        self.act = act

        self.mhca = MHCrossAttn(dimA=self.dim, dimB=dim_cells, numheads=self.numheads, skip_rescale=skip_rescale,
                                add_pos=add_pos, Asize=imgsize, Bsize=memsize)
        self.mhca2 = MHCrossAttn(dimA=self.dim_cells, dimB=dim, numheads=self.numheads, skip_rescale=skip_rescale,
                                add_pos=add_pos, Asize=memsize, Bsize=imgsize)

    def forward(self, x, cells):
        """
        :param x:  (batsize, numchannel, height, width)
        :return:
        """
        skip_x, skip_cells = x, cells

        cells = self.mhca(skip_x, skip_cells)
        x = self.mhca2(skip_cells, skip_x)
        skip_x, skip_cells = x, cells

        x = self.groupnorm(x)
        _x = self.linA(x)
        _x = self.act(_x)
        _x = self.linB(_x)

        newx = _x + skip_x
        if self.skip_rescale:
            newx /= sqrt(2)

        cells = self.groupnorm_cells(cells)
        _cells = self.linA_cells(cells)
        _cells = self.act(_cells)
        _cells = self.linB_cells(_cells)

        newcells = _cells + skip_cells
        if self.skip_rescale:
            newcells /= sqrt(2)

        return newx, newcells


class SpecialTransformerLayer(torch.nn.Module):
    """
    Assumes input is image and a (smaller) "memory" image.
    This layer has three phases:
    1. pull information from real image into the memory image using multi-head attention
    2. do multi-head self-attention over the memory image
    3. perform an update using two FF layers
    4. pull information from memory image into image using multi-head attention
    """
    def __init__(self, dim=64, memdim=64, numheads=2, skip_rescale=True,
                 add_pos=False, imgsize=-1, memsize=-1, step_emb_dim=-1,
                 z_emb_dim=-1, adaptdim=-1,
                 act=torch.nn.GELU()):
        super().__init__()
        self.dim, self.memdim, self.numheads, self.skip_rescale, self.step_emb_dim, self.z_emb_dim, self.adaptdim \
            = dim, memdim, numheads, skip_rescale, step_emb_dim, z_emb_dim, adaptdim
        self.dim_ff = self.dim * 2
        self.dim_ff_cells = self.memdim * 2

        adaptive = self.step_emb_dim != -1 or self.z_emb_dim != -1

        if adaptive:
            self.adaptdim = self.dim if self.adaptdim == -1 else self.adaptdim
            self.groupnorm = AdaptiveGroupNorm(min(dim // 4, 32), dim, self.adaptdim)
            self.groupnorm_cells = AdaptiveGroupNorm(min(memdim // 4, 32), memdim, self.adaptdim)
        else:
            self.groupnorm = torch.nn.GroupNorm(num_groups=min(dim // 4, 32), num_channels=dim, eps=1e-6)
            self.groupnorm_cells = torch.nn.GroupNorm(num_groups=min(memdim // 4, 32), num_channels=memdim, eps=1e-6)

        self.linA = LinearOnChannel(self.dim, self.dim_ff)
        self.linB = LinearOnChannel(self.dim_ff, self.dim)

        self.linA_cells = LinearOnChannel(self.memdim, self.dim_ff_cells)
        self.linB_cells = LinearOnChannel(self.dim_ff_cells, self.memdim)
        self.act = act

        self.mhsa = MHSelfAttn(dim=self.memdim, numheads=self.numheads, skip_rescale=skip_rescale)
        self.mhca = MHCrossAttn(dimA=self.dim, dimB=self.memdim, numheads=self.numheads, skip_rescale=skip_rescale,
                                add_pos=add_pos, Asize=imgsize, Bsize=memsize)
        self.mhca2 = MHCrossAttn(dimA=self.memdim, dimB=self.dim, numheads=self.numheads, skip_rescale=skip_rescale,
                                add_pos=add_pos, Asize=memsize, Bsize=imgsize)

        self.step_linear_gn = torch.nn.Sequential(dense(self.step_emb_dim, self.adaptdim), self.act)
        self.zemb_linear_gn = None
        if self.z_emb_dim != -1:
            self.zemb_linear_gn = torch.nn.Sequential(dense(self.z_emb_dim, self.adaptdim), self.act)

    def forward(self, x, cells, stepemb=None, zemb=None):
        """
        :param x:  (batsize, numchannel, height, width)
        :return:
        """
        skip_cells = cells

        # pull from image to memory
        cells = self.mhca(x, cells)
        # do self-attention on memory
        cells = self.mhsa(cells)

        # compute adaptive state values
        adaptive_h = None
        if stepemb is not None:
            _adaptive_h = self.step_linear_gn(stepemb)[..., None, None]
            adaptive_h = adaptive_h + _adaptive_h if adaptive_h is not None else _adaptive_h
        if zemb is not None:
            _adaptive_h = self.zemb_linear_gn(zemb)[..., None, None]
            adaptive_h = adaptive_h + _adaptive_h if adaptive_h is not None else _adaptive_h

        # update memory
        if adaptive_h is not None:
            cells = self.groupnorm_cells(cells, adaptive_h)
        else:
            cells = self.groupnorm_cells(cells)
        _cells = self.linA_cells(cells)
        _cells = self.act(_cells)
        _cells = self.linB_cells(_cells)

        cells = _cells + skip_cells
        if self.skip_rescale:
            cells /= sqrt(2)

        # pull from memory to image
        x = self.mhca2(cells, x)
        skip_x = x
        # update image
        if adaptive_h is not None:
            x = self.groupnorm(x, adaptive_h)
        else:
            x = self.groupnorm(x)
        _x = self.linA(x)
        _x = self.act(_x)
        _x = self.linB(_x)

        x = _x + skip_x
        if self.skip_rescale:
            x /= sqrt(2)
        return x, cells


class ResidualBlock(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            t_emb_dim=-1,
            z_emb_dim=-1,
            downsample=False,
            upsample=False,
            act=torch.nn.LeakyReLU(0.2),
            fir_kernel=(1, 3, 3, 1),
            skip_rescale=True,
    ):
        super().__init__()
        assert not (downsample and upsample)
        self.fir_kernel = fir_kernel
        self.upsample = upsample
        self.downsample = downsample
        self.in_channels, self.out_channels, self.skip_rescale = in_channels, out_channels, skip_rescale

        self.conv1 = conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = conv2d(out_channels, out_channels, kernel_size, padding=padding, init_scale=0.)

        self.dense_time = None
        if t_emb_dim != -1:
            self.dense_time = torch.nn.Sequential(dense(t_emb_dim, z_emb_dim), act)

        self.act = act
        self.z_emb_dim = z_emb_dim
        self.adaptive = self.z_emb_dim != -1 or t_emb_dim != -1
        if self.adaptive:
            self.groupnorm1 = AdaptiveGroupNorm(min(in_channels // 4, 32), in_channels, self.z_emb_dim)
            self.groupnorm2 = AdaptiveGroupNorm(min(out_channels // 4, 32), out_channels, self.z_emb_dim)
        else:
            self.groupnorm1 = torch.nn.GroupNorm(num_groups=min(in_channels // 4, 32), num_channels=in_channels, eps=1e-6)
            self.groupnorm2 = torch.nn.GroupNorm(num_groups=min(out_channels // 4, 32), num_channels=out_channels, eps=1e-6)

        self.skip = None
        if self.in_channels != self.out_channels or self.upsample or self.downsample:
            self.skip = LinearOnChannel(in_channels, out_channels, bias=False)

    def forward(self, input, t_emb=None, z_emb=None):

        modulator = None
        if t_emb is not None:
            _modulator = self.dense_time(t_emb)
            modulator = modulator + _modulator if modulator is not None else _modulator
        if z_emb is not None:
            _modulator = z_emb
            modulator = modulator + _modulator if modulator is not None else _modulator

        if modulator is not None:
            out = self.act(self.groupnorm1(input, modulator))
        else:
            out = self.act(self.groupnorm1(input))

        if self.downsample:
            out = up_or_down_sampling.downsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.downsample_2d(input, self.fir_kernel, factor=2)
        if self.upsample:
            out = up_or_down_sampling.upsample_2d(out, self.fir_kernel, factor=2)
            input = up_or_down_sampling.upsample_2d(input, self.fir_kernel, factor=2)

        out = self.conv1(out)

        if modulator is not None:
            out = self.act(self.groupnorm2(out, modulator))
        else:
            out = self.act(self.groupnorm2(out))

        out = self.conv2(out)

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input

        out = out + skip
        if self.skip_rescale:
            out /= sqrt(2)
        return out


class ContinuousTimestepEmbedding(torch.nn.Module):
    def __init__(self, outdim, embdim=1000, act=torch.nn.GELU(), maxsteps=-1):
        """
        :param hdim:
        :param outdim:
        :param act:
        :param maxsteps:  if maxsteps is provided (not -1), we normalize, otherwise we expect input to this layer to be float between 0 and 1
        """
        super().__init__()
        self.embdim, self.hdim, self.outdim, self.maxsteps \
            = embdim, outdim, outdim, maxsteps
        self.net = torch.nn.Sequential(
            dense(1, self.embdim),
            act,
            dense(self.embdim, self.hdim),
            act,
            dense(self.hdim, self.outdim)
        )

    def forward(self, t, maxsteps=None):
        # "t" are discrete steps between 0 and self.maxsteps
        maxsteps = maxsteps if maxsteps is not None else self.maxsteps
        if maxsteps != -1:      # if maxsteps specified, we transform discrete time into continuous
            t = t.float() / self.maxsteps
        temb = self.net(t[:, None])
        return temb


class Discriminator_32x32(torch.nn.Module):
    def __init__(self, num_inp_channels=3, imgsize=32, num_features=64, num_heads=2, step_emb_dim=128, act=torch.nn.LeakyReLU(0.2),
                 memsize=10, memdim=256, maxsteps=-1):
        super().__init__()
        self.num_inp_channels, self.imgsize, self.num_features, self.step_emb_dim, self.memsize, self.memdim, self.numheads, self.maxsteps \
            = num_inp_channels, imgsize, num_features, step_emb_dim, memsize, memdim, num_heads, maxsteps

        self.act = act
        self.time_embed = ContinuousTimestepEmbedding(step_emb_dim, act=act, maxsteps=self.maxsteps)
        self.x_posembed = LearnedPositionalEmbeddings2D(self.imgsize, self.imgsize, self.num_features)
        self.cell_posembed = LearnedPositionalEmbeddings2DIndep(self.memsize, self.memsize, self.memdim)

        self.initmap = LinearOnChannel(self.num_inp_channels*2, self.num_features)  # 32x32

        self.conv1 = ResidualBlock(self.num_features, self.num_features, t_emb_dim=step_emb_dim, act=act)  # 32x32
        self.tm1 = SpecialTransformerLayer(self.num_features, self.memdim, numheads=self.numheads, step_emb_dim=step_emb_dim)
        self.conv2 = ResidualBlock(self.num_features, self.num_features*2, t_emb_dim=step_emb_dim, downsample=True, act=act)  # 16x16
        self.tm2 = SpecialTransformerLayer(self.num_features*2, self.memdim, numheads=self.numheads, step_emb_dim=step_emb_dim)
        self.conv3 = ResidualBlock(self.num_features*2, self.num_features*4, t_emb_dim=step_emb_dim, downsample=True, act=act)  # 8x8
        self.tm3 = SpecialTransformerLayer(self.num_features*4, self.memdim, numheads=self.numheads, step_emb_dim=step_emb_dim)

        self.tm4 = SpecialTransformerLayer(self.num_features*4, self.memdim, numheads=self.numheads, step_emb_dim=step_emb_dim)

        self.final_lin_A = dense(self.num_features*8, self.num_features*4)
        self.final_nonlin = torch.nn.GELU()
        self.final_lin_B = dense(self.num_features*4, 1)

    def forward(self, x, t, x_t):
        batsize = x.size(0)
        # embed time step
        step_emb = self.act(self.time_embed(t))

        # create input x
        inp = torch.cat((x, x_t), dim=1)  # (batsize, 6, height, width)

        # initialize memory
        cells = torch.zeros(batsize, self.memdim, self.memsize, self.memsize).to(x.device)
        cells = self.cell_posembed(cells)

        # run layers
        hs = [inp]
        cellses = [cells]
        hs.append(self.initmap(hs[-1]))
        hs.append(self.conv1(hs[-1], step_emb))
        h, cells = self.tm1(hs[-1], cellses[-1], step_emb)
        hs.append(h), cellses.append(cells)
        hs.append(self.conv2(hs[-1], step_emb))
        h, cells = self.tm2(hs[-1], cellses[-1], step_emb)
        hs.append(h), cellses.append(cells)
        hs.append(self.conv3(hs[-1], step_emb))
        h, cells = self.tm3(hs[-1], cellses[-1], step_emb)
        hs.append(h), cellses.append(cells)

        h, cells = self.tm4(hs[-1], cellses[-1], step_emb)
        hs.append(h), cellses.append(cells)

        cls = cellses[-1][:, :, 0, 0]
        maxpooled = cellses[-1].max(-1)[0].max(-1)[0]

        out = torch.cat((cls, maxpooled), -1)
        out = self.final_lin_B(self.final_nonlin(self.final_lin_A(out)))

        return out


class ZMapper(torch.nn.Module):
    def __init__(self, zdim, zembdim, act=torch.nn.GELU(), numlayers=2):
        super().__init__()
        self.zdim, self.zembdim, self.act, self.numlayers \
            = zdim, zembdim, act, numlayers

        self.layers = torch.nn.Sequential(
            PixelNorm(),
            dense(self.zdim, self.zembdim),
            self.act
        )
        for _ in range(numlayers-1):
            self.layers.append(dense(self.zembdim, self.zembdim))
            self.layers.append(self.act)

    def forward(self, x):
        return self.layers(x)


class Generator_32x32(torch.nn.Module):
    def __init__(self, num_inp_channels=3, imgsize=32, num_features=64, num_heads=2, step_emb_dim=128, zdim=128, z_emb_dim=128, act=torch.nn.LeakyReLU(0.2),
                 memsize=10, memdim=256, maxsteps=-1, num_z_map_layers=4):
        super().__init__()
        self.num_inp_channels, self.imgsize, self.num_features, self.step_emb_dim, self.memsize, self.memdim, self.numheads, self.maxsteps, self.z_emb_dim, self.zdim, self.num_z_map_layers \
            = num_inp_channels, imgsize, num_features, step_emb_dim, memsize, memdim, num_heads, maxsteps, z_emb_dim, zdim, num_z_map_layers

        self.act = act
        self.time_embed = ContinuousTimestepEmbedding(step_emb_dim, act=act, maxsteps=self.maxsteps)
        self.x_posembed = LearnedPositionalEmbeddings2D(self.imgsize, self.imgsize, self.num_features)
        self.cell_posembed = LearnedPositionalEmbeddings2DIndep(self.memsize, self.memsize, self.memdim)

        self.initmap = LinearOnChannel(self.num_inp_channels, self.num_features)  # 32x32

        # down
        self.conv1 = ResidualBlock(self.num_features, self.num_features, t_emb_dim=step_emb_dim, act=act)  # 32x32
        self.tm1 = SpecialTransformerLayer(self.num_features, self.memdim, numheads=self.numheads, step_emb_dim=step_emb_dim)
        self.conv2 = ResidualBlock(self.num_features, self.num_features*2, t_emb_dim=step_emb_dim, downsample=True, act=act)  # 16x16
        self.tm2 = SpecialTransformerLayer(self.num_features*2, self.memdim, numheads=self.numheads, step_emb_dim=step_emb_dim)
        self.conv3 = ResidualBlock(self.num_features*2, self.num_features*4, t_emb_dim=step_emb_dim, downsample=True, act=act)  # 8x8
        self.tm3 = SpecialTransformerLayer(self.num_features*4, self.memdim, numheads=self.numheads, step_emb_dim=step_emb_dim)

        self.tm4 = SpecialTransformerLayer(self.num_features*4, self.memdim, numheads=self.numheads, step_emb_dim=step_emb_dim)

        self.z_mapper = ZMapper(self.zdim, self.z_emb_dim, act=act, numlayers=self.num_z_map_layers)

        # up
        self.conv5 = ResidualBlock(self.num_features * 4, self.num_features * 2, t_emb_dim=step_emb_dim, act=act, upsample=True)  # 8x8 -> 16x16
        self.tm5 = SpecialTransformerLayer(self.num_features * 2, self.memdim, numheads=self.numheads, step_emb_dim=step_emb_dim)
        self.conv6 = ResidualBlock(self.num_features * 4, self.num_features, t_emb_dim=step_emb_dim, act=act, upsample=True)  # 16x16x8 -> 32x32
        self.tm6 = SpecialTransformerLayer(self.num_features, self.memdim, numheads=self.numheads, step_emb_dim=step_emb_dim)

        self.conv6 = ResidualBlock(self.num_features * 2, self.num_features, t_emb_dim=step_emb_dim, act=act)  # 32x32
        self.outmap = LinearOnChannel(self.num_features, self.num_inp_channels)

    def forward(self, x, t, z):
        batsize = x.size(0)
        # embed time step
        step_emb = self.act(self.time_embed(t))

        # embed z
        z_emb = self.z_mapper(z)

        # create input x
        inp = x

        # initialize memory
        cells = torch.zeros(batsize, self.memdim, self.memsize, self.memsize)
        cells = self.cell_posembed(cells)

        # run layers
        # down
        hs = [inp]
        cellses = [cells]
        hs.append(self.initmap(hs[-1]))
        hs.append(self.conv1(hs[-1], step_emb, z_emb))             # 32x32
        h, cells = self.tm1(hs[-1], cellses[-1], step_emb, z_emb)
        hs.append(h), cellses.append(cells)
        hs.append(self.conv2(hs[-1], step_emb, z_emb))             # 16x16
        h, cells = self.tm2(hs[-1], cellses[-1], step_emb, z_emb)
        hs.append(h), cellses.append(cells)
        hs.append(self.conv3(hs[-1], step_emb, z_emb))             # 8x8
        h, cells = self.tm3(hs[-1], cellses[-1], step_emb, z_emb)
        hs.append(h), cellses.append(cells)

        h, cells = self.tm4(hs[-1], cellses[-1], step_emb, z_emb)
        hs.append(h), cellses.append(cells)

        # up
        hsm1, *hs = hs
        cells = cellses[-1]
        h = self.conv5(hsm1, step_emb, z_emb)     # 16x16
        hs = hs[:-2]
        h, cells = self.tm5(h, cells, step_emb, z_emb)
        h = torch.cat((hs.pop(), h), 1)
        h = self.conv6(h, step_emb, z_emb)        # 32x32
        h, cells = self.tm6(h, cells, step_emb, z_emb)
        h = torch.cat([hs.pop(), h], 1)
        h = self.conv7(h, step_emb, z_emb)        # 32x32

        out = self.outmap(h)

        return out


if __name__ == '__main__':
    m = Discriminator_32x32(maxsteps=32)
    print(m)
    print(count_parameters(m))

    x = torch.rand(5, 3, 32, 32)
    x_t = torch.rand(5, 3, 32, 32)
    t = torch.randint(0, 32, (5,))

    y = m(x, t, x_t)
    print(y)

    m = Generator_32x32(maxsteps=32)
    print(m)
    print(count_parameters(m))
    y = m(x, t)
    print(y)
