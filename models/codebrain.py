import torch
import torch.nn as nn

import math
from einops import rearrange, repeat


import torch.nn.functional as F
from einops import rearrange, repeat
import opt_einsum as oe
from functools import partial
optimized = True

if optimized:
    contract = oe.contract
else:
    contract = torch.einsum

def get_initializer(name, activation=None):
    if activation in [ None, 'id', 'identity', 'linear', 'modrelu' ]:
        nonlinearity = 'linear'
    elif activation in ['relu', 'tanh', 'sigmoid']:
        nonlinearity = activation
    elif activation in ['gelu', 'swish', 'silu']:
        nonlinearity = 'relu' # Close to ReLU so approximate with ReLU's gain
    else:
        raise NotImplementedError(f"get_initializer: activation {activation} not supported")

    if name == 'uniform':
        initializer = partial(torch.nn.init.kaiming_uniform_, nonlinearity=nonlinearity)
    elif name == 'normal':
        initializer = partial(torch.nn.init.kaiming_normal_, nonlinearity=nonlinearity)
    elif name == 'xavier':
        initializer = torch.nn.init.xavier_normal_
    elif name == 'zero':
        initializer = partial(torch.nn.init.constant_, val=0)
    elif name == 'one':
        initializer = partial(torch.nn.init.constant_, val=1)
    else:
        raise NotImplementedError(f"get_initializer: initializer type {name} not supported")

    return initializer

class modrelu(nn.Module):
    def __init__(self, features):
        super(modrelu, self).__init__()
        self.features = features
        self.b = nn.Parameter(torch.Tensor(self.features))
        self.reset_parameters()

    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

    def forward(self, inputs):
        norm = torch.abs(inputs)
        biased_norm = norm + self.b
        magnitude = nn.functional.relu(biased_norm)
        phase = torch.sign(inputs)

        return phase * magnitude

class Modrelu(modrelu):
    def reset_parameters(self):
        self.b.data.uniform_(-0.01, 0.01)

class TransposedLinear(nn.Module):

    def __init__(self, d_input, d_output, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d_output, d_input))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(d_output))
            bound = 1 / math.sqrt(d_input)
            nn.init.uniform_(self.bias, -bound, bound)
            setattr(self.bias, "_optim", {"weight_decay": 0.0})
        else:
            self.bias = 0.0

    def forward(self, x):
        num_axis = len(x.shape[2:])
        y = contract('b u ..., v u -> b v ...', x, self.weight) + \
            self.bias.view(-1, *[1]*num_axis)
        return y


class TransposedLN(nn.Module):
    def __init__(self, d, scalar=True):
        super().__init__()
        self.scalar = scalar
        if self.scalar:
            self.m = nn.Parameter(torch.zeros(1))
            self.s = nn.Parameter(torch.ones(1))
            setattr(self.m, "_optim", {"weight_decay": 0.0})
            setattr(self.s, "_optim", {"weight_decay": 0.0})
        else:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        if self.scalar:
            s, m = torch.std_mean(x, dim=1, unbiased=False, keepdim=True)
            y = (self.s/s) * (x-m+self.m)
        else:
            _x = self.ln(rearrange(x, 'b d ... -> b ... d'))
            y = rearrange(_x, 'b ... d -> b d ...')
        return y


def Activation(activation=None, size=None, dim=-1):
    if activation in [None, 'id', 'identity', 'linear']:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'modrelu':
        return Modrelu(size)
    elif activation == 'ln':
        return TransposedLN(dim)
    else:
        raise NotImplementedError(
            "hidden activation '{}' is not implemented".format(activation))


def LinearActivation(
    d_input, d_output, bias=True,
    zero_bias_init=False,
    transposed=False,
    initializer=None,
    activation=None,
    activate=False,
    weight_norm=False,
    **kwargs,
):

    linear_cls = TransposedLinear if transposed else nn.Linear
    if activation == 'glu':
        d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if initializer is not None:
        get_initializer(initializer, activation)(linear.weight)

    if bias and zero_bias_init:
        nn.init.zeros_(linear.bias)

    if weight_norm:
        linear = nn.utils.weight_norm(linear)

    if activate and activation is not None:
        activation = Activation(activation, d_output,
                                dim=1 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear


class Normalization(nn.Module):
    def __init__(
        self,
        d,
        transposed=False,
        _name_='layer',
        **kwargs
    ):
        super().__init__()
        self.transposed = transposed
        self._name_ = _name_

        if _name_ == 'layer':
            self.channel = True
            if self.transposed:
                self.norm = TransposedLN(d, **kwargs)
            else:
                self.norm = nn.LayerNorm(d, **kwargs)
        elif _name_ == 'instance':
            self.channel = False
            norm_args = {'affine': False, 'track_running_stats': False}
            norm_args.update(kwargs)
            self.norm = nn.InstanceNorm1d(d, **norm_args)
        elif _name_ == 'batch':
            self.channel = False
            norm_args = {'affine': True, 'track_running_stats': True}
            norm_args.update(kwargs)
            self.norm = nn.BatchNorm1d(d, **norm_args)
        elif _name_ == 'group':
            self.channel = False
            self.norm = nn.GroupNorm(1, d, *kwargs)
        elif _name_ == 'none':
            self.channel = True
            self.norm = nn.Identity()
        else: raise NotImplementedError

    def forward(self, x):
        shape = x.shape
        if self.transposed:
            x = rearrange(x, 'b d ... -> b d (...)')
        else:
            x = rearrange(x, 'b ... d -> b (...)d ')

        if self.channel or self.transposed:
            x = self.norm(x)
        else:
            x = x.transpose(-1, -2)
            x = self.norm(x)
            x = x.transpose(-1, -2)

        x = x.view(shape)
        return x

    def step(self, x, **kwargs):
        assert self._name_ in ["layer", "none"]
        if self.transposed: x = x.unsqueeze(-1)
        x = self.forward(x)
        if self.transposed: x = x.squeeze(-1)
        return x

class KernelModule(nn.Module):
    def __init__(self, kernel):
        super(KernelModule, self).__init__()
        self.kernel = nn.Parameter(kernel)

class GConv(nn.Module):
    requires_length = True

    def __init__(
        self,
        d_model,
        d_state=64,
        l_max=1,
        channels=1,
        bidirectional=False,
        activation='gelu',
        ln=False,
        postact=None,
        initializer=None,
        weight_norm=False,
        hyper_act=None,
        dropout=0.0,
        transposed=True,
        verbose=False,
        shift=False,
        linear=False,
        mode="cat_randn",
        **kernel_args,
    ):

        super().__init__()
        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.ln = ln
        self.channels = channels
        self.transposed = transposed
        self.shift = shift
        self.linear = linear
        self.mode = mode
        self.l_max = l_max

        self.hyper = hyper_act is not None
        if self.hyper:
            channels *= 2
            self.hyper_activation = Activation(hyper_act)

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2


        if not self.linear:
            self.activation = Activation(activation)
            dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
            self.dropout = dropout_fn(
                dropout) if dropout > 0.0 else nn.Identity()
            if self.ln:
                self.norm = Normalization(
                    self.h*self.channels, transposed=transposed)
            else:
                self.norm = nn.Identity()


        if not self.linear:
            self.output_linear = LinearActivation(
                self.h*self.channels,
                self.h,
                transposed=self.transposed,
                initializer=initializer,
                activation=postact,
                activate=True,
                weight_norm=weight_norm,
            )

        self.init_scale = kernel_args.get('init_scale', 0)
        self.kernel_dim = kernel_args.get('kernel_dim', 2)
        self.num_scales = kernel_args.get(
            'n_scales', 1+math.ceil(math.log2(l_max/self.kernel_dim))-self.init_scale)
        if self.num_scales is None:
            self.num_scales = 1 + \
                math.ceil(math.log2(l_max/self.kernel_dim)) - self.init_scale
        self.kernel_list = nn.ModuleList()

        decay_min = kernel_args.get('decay_min', 2)
        decay_max = kernel_args.get('decay_max', 2)
        for _ in range(self.num_scales):
            if 'randn' in mode:
                kernel = torch.randn(channels, self.h, self.kernel_dim)
            elif 'cos' in mode:
                kernel = nn.Parameter(torch.cat([torch.cos(torch.linspace(0, 2*i*math.pi, self.kernel_dim)).expand(
                    channels, 1, self.kernel_dim) for i in range(self.h)], dim=1)[:, torch.randperm(self.h), :])
            else:
                raise ValueError(f"Unknown mode {mode}")

            kernel_module = KernelModule(kernel)
            self.kernel_list.append(kernel_module)

        if 'learnable' in mode:
            self.decay = nn.Parameter(torch.rand(
                self.h) * (decay_max - decay_min) + decay_min)
            if 'fixed' in mode:
                self.decay.requires_grad = False
            else:
                self.decay._optim = {
                    'lr': kernel_args.get('lr', 0.001),
                }
            self.register_buffer('multiplier', torch.tensor(1.0))
        else:
            self.register_buffer('multiplier', torch.linspace(
                decay_min, decay_max, self.h).view(1, -1, 1))

        self.register_buffer('kernel_norm', torch.ones(channels, self.h, 1))
        self.register_buffer('kernel_norm_initialized',
                             torch.tensor(0, dtype=torch.bool))


    def forward(self, u, return_kernel=False):
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        kernel_list = []
        interpolate_mode = 'nearest' if 'nearest' in self.mode else 'linear'
        multiplier = self.multiplier
        if 'sum' in self.mode:
            for i in range(self.num_scales):
                kernel = F.pad(
                    F.interpolate(
                        self.kernel_list[i].kernel,
                        scale_factor=2**(i+self.init_scale),
                        mode=interpolate_mode,
                    ),
                    (0, self.kernel_dim*2**(self.num_scales-1+self.init_scale) -
                     self.kernel_dim*2**(i+self.init_scale)),
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = sum(kernel_list)
        elif 'cat' in self.mode:
            for i in range(self.num_scales):
                kernel = F.interpolate(
                    self.kernel_list[i].kernel,
                    scale_factor=2**(max(0, i-1)+self.init_scale),
                    mode=interpolate_mode,
                ) * multiplier ** (self.num_scales - i - 1)
                kernel_list.append(kernel)
            k = torch.cat(kernel_list, dim=-1)
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        if 'learnable' in self.mode:
            k = k * torch.exp(-self.decay.view(1, -1, 1)*torch.log(
                torch.arange(k.size(-1), device=k.device)+1).view(1, 1, -1))

        if not self.kernel_norm_initialized:
            self.kernel_norm = k.norm(dim=-1, keepdim=True).detach()
            self.kernel_norm_initialized = torch.tensor(
                1, dtype=torch.bool, device=k.device)

        if k.size(-1) > L:
            k = k[..., :L]
        elif k.size(-1) < L:
            k = F.pad(k, (0, L - k.size(-1)))

        k = k / self.kernel_norm

        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                + F.pad(k1.flip(-1), (L, 0)) \

        k_f = torch.fft.rfft(k.float(), n=2*L)
        u_f = torch.fft.rfft(u.float(), n=2*L)
        y_f = contract('bhl,chl->bchl', u_f, k_f)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L]

        y = y + contract('bhl,ch->bchl', u, self.D)

        y = rearrange(y, '... c h l -> ... (c h) l')

        if not self.linear:
            y = self.dropout(self.activation(y))

        if not self.transposed:
            y = y.transpose(-1, -2)

        if not self.linear:
            y = self.norm(y)
            y = self.output_linear(y)

        if return_kernel:
            return y, k
        return y, None

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)

def swish(x):
    return x * torch.sigmoid(x)


class SNorm(nn.Module):
    def __init__(self, channels):
        super(SNorm, self).__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        x_norm = (x - x.mean(2, keepdims=True)) / (x.var(2, keepdims=True, unbiased=True) + 0.00001) ** 0.5

        out = x_norm * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
        return out


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        norm = x.norm(dim=1, keepdim=True)
        rms = norm / (x.shape[1] ** 0.5)
        x_normed = x / (rms + self.eps)
        return self.scale * x_normed


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(Conv, self).__init__()
        self.padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.padding)
        self.conv = nn.utils.weight_norm(self.conv)
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        out = self.conv(x)
        return out


class ZeroConv1d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ZeroConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.conv(x)
        return out


class Residual_block(nn.Module):
    def __init__(self, res_channels, skip_channels,
                 diffusion_step_embed_dim_out, in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_block, self).__init__()
        self.res_channels = res_channels

        self.sn = RMSNorm(res_channels)

        self.S41 = GConv(d_model=2 * self.res_channels,
                         channels=4,
                         l_max=s4_lmax,
                         d_state=s4_d_state,
                         dropout=s4_dropout,
                         bidirectional=s4_bidirectional,
                         layer_norm=s4_layernorm)

        self.conv_layer = Conv(self.res_channels, 2 * self.res_channels, kernel_size=3)

        self.attention = nn.MultiheadAttention(embed_dim=2 * self.res_channels, num_heads=4, dropout=s4_dropout,
                                               bias=True, batch_first=True)

        self.gelu = nn.GELU()

        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1)
        self.res_conv = nn.utils.weight_norm(self.res_conv)
        nn.init.kaiming_normal_(self.res_conv.weight)


        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1)
        self.skip_conv = nn.utils.weight_norm(self.skip_conv)
        nn.init.kaiming_normal_(self.skip_conv.weight)


    def generate_local_window_mask(self, seq_len, window_size):
        assert window_size % 2 == 1, "window_size shoule be odd number, like 7, 9, 11"

        half_window = window_size // 2

        mask = torch.full((seq_len, seq_len), float('-inf'))

        for i in range(seq_len):
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            mask[i, start:end] = 0

        return mask

    def forward(self, input_data):
        x, original = input_data
        h = x
        B, C, L = x.shape
        x = self.sn(x)
        assert C == self.res_channels

        part_t = original.view([B, self.res_channels, L])
        h = h + part_t

        h = self.conv_layer(h)

        h = self.gelu(h)
        h_t, _ = self.S41(h)

        h_s = rearrange(h_t, "b c l -> b l c")
        SWA_mask = self.generate_local_window_mask(L, 1).cuda()
        h_s, _ = self.attention(h_s, h_s, h_s, attn_mask=SWA_mask)
        h_s = rearrange(h_s, "b l c -> b c l")

        h = h_t + h_s

        out = torch.tanh(h[:, :self.res_channels, :]) * torch.sigmoid(h[:, self.res_channels:, :])

        res = self.res_conv(out)
        assert x.shape == res.shape
        skip = self.skip_conv(out)

        return (x + res) * math.sqrt(0.5), skip


class Residual_group(nn.Module):
    def __init__(self, res_channels, skip_channels, num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 in_channels,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm):
        super(Residual_group, self).__init__()
        self.num_res_layers = num_res_layers

        self.residual_blocks = nn.ModuleList()
        for n in range(self.num_res_layers):
            self.residual_blocks.append(Residual_block(res_channels, skip_channels,
                                                       diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                                       in_channels=in_channels,
                                                       s4_lmax=s4_lmax,
                                                       s4_d_state=s4_d_state,
                                                       s4_dropout=s4_dropout,
                                                       s4_bidirectional=s4_bidirectional,
                                                       s4_layernorm=s4_layernorm))

    def forward(self, input_data):
        noise = input_data
        h = noise
        skip = 0
        for n in range(self.num_res_layers):
            h, skip_n = self.residual_blocks[n]((h, noise))
            skip = skip_n + skip

        return skip * math.sqrt(1.0 / self.num_res_layers)


class SSSM(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels,
                 num_res_layers,
                 diffusion_step_embed_dim_in,
                 diffusion_step_embed_dim_mid,
                 diffusion_step_embed_dim_out,
                 s4_lmax,
                 s4_d_state,
                 s4_dropout,
                 s4_bidirectional,
                 s4_layernorm,
                 codebook_size_t,
                 codebook_size_f,
                 if_codebook=True):
        super(SSSM, self).__init__()

        self.patch_embedding = PatchEmbedding(in_channels, out_channels, res_channels, s4_lmax // 19, 200)

        self.init_conv = nn.Sequential(Conv(in_channels, res_channels, kernel_size=1), nn.ReLU())

        self.residual_layer = Residual_group(res_channels=res_channels,
                                             skip_channels=skip_channels,
                                             num_res_layers=num_res_layers,
                                             diffusion_step_embed_dim_in=diffusion_step_embed_dim_in,
                                             diffusion_step_embed_dim_mid=diffusion_step_embed_dim_mid,
                                             diffusion_step_embed_dim_out=diffusion_step_embed_dim_out,
                                             in_channels=in_channels,
                                             s4_lmax=s4_lmax,
                                             s4_d_state=s4_d_state,
                                             s4_dropout=s4_dropout,
                                             s4_bidirectional=s4_bidirectional,
                                             s4_layernorm=s4_layernorm)

        self.final_conv = nn.Sequential(Conv(skip_channels, skip_channels, kernel_size=1),
                                        nn.ReLU(),
                                        ZeroConv1d(skip_channels, out_channels))
        self.lm_head_t = nn.Linear(out_channels, codebook_size_t, bias=False)
        self.lm_head_f = nn.Linear(out_channels, codebook_size_f, bias=False)
        self.if_codebook = if_codebook
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, inputs, mask=None):
        bz, ch_num, seq_len, patch_size = inputs.shape
        inputs = self.patch_embedding(inputs, mask=mask)
        x = rearrange(inputs, 'b c s p -> b p (c s)')
        x = self.init_conv(x)
        x = self.residual_layer(x)
        x = self.final_conv(x)
        x = rearrange(x, 'b p (c s)  -> b c s p', p=patch_size, s=seq_len, c=ch_num)
        x = self.norm(x)
        if self.if_codebook:
            x = x.squeeze()[mask == 1]
            x_t = self.lm_head_t(x)
            x_f = self.lm_head_f(x)
            return (x_t, x_f)
        else:
            return x.squeeze()


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len, p_seq_len):
        super().__init__()
        self.d_model = p_seq_len
        self.p_seq_len = p_seq_len
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=(19, 7), stride=(1, 1),
                      padding=(9, 3),
                      groups=self.d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(p_seq_len), requires_grad=False)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, self.d_model),
            nn.Dropout(0.1),
        )

    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.p_seq_len)

        mask_x = mask_x.contiguous().view(bz * ch_num * patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=3750):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(2), :].permute(1, 2, 0)
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, param):
        super(Model, self).__init__()
        self.backbone = SSSM(
            in_channels = 200, res_channels = 200,
            skip_channels = 200, out_channels = 200,
            num_res_layers = param.n_layer,
            diffusion_step_embed_dim_in = 200,
            diffusion_step_embed_dim_mid = 200,
            diffusion_step_embed_dim_out = 200,
            s4_lmax = 570,
            s4_d_state = 64,
            s4_dropout = param.dropout,
            s4_bidirectional = True,
            s4_layernorm = True,
            codebook_size_t = param.codebook_size_t,
            codebook_size_f = param.codebook_size_f,
            if_codebook = False)
        if param.use_pretrained_weights:
            print("LOADING WEIGHTS......")
            map_location = torch.device(f'cuda:{param.cuda}')
            state_dict = torch.load(param.foundation_dir, map_location=map_location)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            self.backbone.load_state_dict(new_state_dict)
            if param.frozen:
                for p in self.backbone.parameters():
                    p.requires_grad = False
        self.backbone.proj_out = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(param.enc_in*200, 5*200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(5*200, 200),
            nn.ELU(),
            nn.Dropout(param.dropout),
            nn.Linear(200, param.num_of_classes)
        )
        self.frozen = param.frozen

    def forward(self, x):
        if self.frozen:
            self.backbone.eval()
        bs, channel, seq_len = x.shape
        x = x[:,:,:seq_len//200*200]
        x = x.contiguous().view(bs, channel, seq_len//200, 200)
        bz, ch_num, seq_len, patch_size = x.shape
        feats = self.backbone(x)
        feats = feats.mean(dim=2)
        feats = feats.contiguous().view(bz, ch_num*patch_size)
        out = self.classifier(feats)
        return out