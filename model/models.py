from ncps.torch import CfC
from ncps.wirings import AutoNCP
from torch import nn

class NCP(nn.Module):
    def __init__(self, args):
        super().__init__()
        wiring = AutoNCP(args.ncp_neurons, 1)
        self.cfc = CfC(1, wiring)

    def forward(self, x):
        x = x.squeeze(0).unsqueeze(-1)
        return self.cfc(x)[0].squeeze(-1).unsqueeze(0).unsqueeze(0)

def get_ncp(args):
    ncp = NCP(args)
    return ncp
