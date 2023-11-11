import torch
_eps = 1e-10

class StiefelManifold : 

    def __init__(self, args, logger, eps = 1e-3, norm_clip = 1, max_norm = 1e3) : 
        self.args = args
        self.logger = logger
        self.eps = eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm

    def normalize(self, w) : 
        return w

    def init_embed(self, embed, irange = 1e-2) : 
        embed.weight.data.uniform_(-irange, irange)
        embed.weight.data.copy_(self.normalize(embed.weight.data))

    def symmetric(self, A) : 
        return 0.5 * (A + A.t())

    def rgrad(self, A, B) : 
        out = B - A.mm(self.symmetric(A.transpose(0, 1).mm(B)))
        return out

    def exp_map_x(self, A, ref) : 
        data = A + ref
        Q, R = torch.linalg.qr(data)
        # To avoid (any possible) negative values in the output matrix, we multiply the negative values by -1
        sign = (R.diag().sign() + 0.5).sign().diag()
        out = Q.mm(sign)
        return out
