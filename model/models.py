from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch import nn
import torchaudio.functional as AF

from s4.src.models.sequence.backbones.sashimi import Sashimi as SashmiBackbone
from s4.src.tasks.decoders import SequenceDecoder


class NCP(nn.Module):
    def __init__(self, args):
        super().__init__()
        wiring = AutoNCP(args.ncp_neurons, 1)
        self.cfc = CfC(1, wiring)

    def forward(self, x):
        x = x.squeeze(0).unsqueeze(-1)
        return self.cfc(x)[0].squeeze(-1).unsqueeze(0).unsqueeze(0)


class Packer(nn.Module):
    def __init__(self, model, pre_shape=None, post_shape=None):
        super().__init__()
        self.model = model
        self.pre = pre_shape
        self.post = post_shape

    def forward(self, x):
        if self.pre is not None:
            x = x.reshape(self.pre)
        x = self.model(x)
        if self.post is not None:
            x = x.reshape(self.post)
        return x


class Sashimi(nn.Module):
    # default configs from the paper SaShiMi experiment
    sashimi_conf = {'_name_': 'sashimi',
         'act_pool': None,
         'd_model': 64,
         'dropout': 0.0,
         'dropres': 0.0,
         'expand': 2,
         'ff': 2,
         'initializer': None,
         'interp': 0,
         'layer': {'_name_': 's4',
                   'activation': 'gelu',
                   'bidirectional': False,
                   'bottleneck': None,
                   'channels': 1,
                   'd_state': 64,
                   'deterministic': False,
                   'drop_kernel': 0.0,
                   'dt_max': 0.1,
                   'dt_min': 0.001,
                   'dt_transform': 'softplus',
                   'final_act': 'glu',
                   'gate': None,
                   'gate_act': 'id',
                   'init': 'legs',
                   'initializer': None,
                   'l_max': None,
                   'layer': 'fftconv',
                   'lr': {'A': 0.001, 'B': 0.001, 'dt': 0.001},
                   'measure': None,
                   'mode': 'nplr',
                   'mult_act': None,
                   'n_ssm': 1,
                   'postact': None,
                   'rank': 1,
                   'tie_dropout': None,
                   'verbose': True,
                   'wd': 0.0,
                   'weight_norm': False},
         'n_layers': 8,
         'norm': 'layer',
         'pool': [4, 4],
         'prenorm': True,
         'residual': 'R',
         'transposed': True}
    del sashimi_conf['_name_']

    def __init__(self, args):
        super().__init__()
        self.encoder = nn.Embedding(256, 64)
        self.model = SashmiBackbone(**self.sashimi_conf)
        self.added_decoder = False
        self.device = args.device

    def forward(self, x):
        if not self.added_decoder:
            self.add_module('decoder', SequenceDecoder(64, 1, x.shape[1],
                    mode='last').to(self.device))
            self.added_decoder = True
        # print(f'Encoder input shape: {x.shape}')
        x = self.encoder(x)
        # print(f'Sashimi input shape: {x.shape}')
        # Sashimi Backbone returns tuple of x, None
        x, _ = self.model(x)
        # print(f'Decoder input shape: {x.shape}')
        x = self.decoder(x)
        # print(f'Decoder output shape: {x.shape}')
        return x


class MuLaw(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = AF.mu_law_encoding(x, 256)
        x = self.model(x)
        x = AF.mu_law_decoding(x, 256)
        return x


def get_ncp(args):
    ncp = NCP(args)
    return ncp

def get_sashimi(args):
    sashimi = Packer(MuLaw(Sashimi(args)), pre_shape=[1,-1], post_shape=[1,1,1,-1])
    return sashimi
