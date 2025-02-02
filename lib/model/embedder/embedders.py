import torch
import numpy as np

class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        min_freq = self.kwargs['min_freq']
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(min_freq, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**min_freq, 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class Embedder_hannw:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        min_freq = self.kwargs['min_freq']
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        kick_in_iter = self.kwargs['kick_in_iter']
        full_band_iter = self.kwargs['full_band_iter']

        freq_bands = 2. ** torch.linspace(min_freq, max_freq, steps=N_freqs)

        # get hann window weights
        kick_in_iter = torch.tensor(kick_in_iter,
                                    dtype=torch.float32)
        t = torch.clamp(self.kwargs['iter_val'] - kick_in_iter, min=0.)
        N = full_band_iter - kick_in_iter
        m = N_freqs
        alpha = m * t / N

        for freq_idx, freq in enumerate(freq_bands):
            w = (1. - torch.cos(np.pi * torch.clamp(alpha - freq_idx,
                                                    min=0., max=1.))) / 2.
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq, w=w: w * p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(opt, current_step=-1):

    if opt.mode == 'fourier':
        embed_kwargs = {
            'include_input': opt.include_input,
            'input_dims': opt.d_in,
            'min_freq': opt.min_freq,
            'max_freq_log2': opt.min_freq + opt.multires - 1,
            'num_freqs': opt.multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }
        embedder_obj = Embedder(**embed_kwargs)

    elif opt.mode == 'hannw_fourier':
        embed_kwargs = {
            'include_input': opt.include_input,
            'input_dims': opt.d_in,
            'min_freq': opt.min_freq,
            'max_freq_log2': opt.min_freq + opt.multires - 1,
            'num_freqs': opt.multires,
            'periodic_fns': [torch.sin, torch.cos],
            'iter_val': current_step,
            'kick_in_iter': opt.kick_in_iter,
            'full_band_iter': opt.full_band_iter
        }
        embedder_obj = Embedder_hannw(**embed_kwargs)


    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim