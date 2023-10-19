"""Lorentz manifold."""
import torch
from torch.autograd import Function, Variable
from utils import *
from utils.pre_utils import *
from manifolds import *
from utils.math_utils import arcosh

_eps = 1e-10

class LorentzManifold : 

    def __init__(self, args, eps = 1e-3, norm_clip = 1, max_norm = 1e3) : 
        self.args = args
        self.eps = eps
        self.norm_clip = norm_clip
        self.max_norm = max_norm

    def minkowski_dot(self, x, y, keepdim = True) : 
        res = torch.sum(x * y, dim = -1) - 2 * x[..., 0] * y[..., 0]
        if keepdim : 
            res = res.view(res.shape + (1,))
        return res


    def sqdist(self, x, y, c) : 
        K = 1. / c
        prod = self.minkowski_dot(x, y)
        eps = {torch.float32 :  1e-7, torch.float64 :  1e-15}
        theta = torch.clamp(-prod / K, min = 1.0 + eps[x.dtype])
        sqdist = K * arcosh(theta) ** 2
        return torch.clamp(sqdist, max = 50.0)


    @staticmethod
    def ldot(u, v, keepdim = False) : 
        """
        Lorentzian Scalar Product
        Args : 
            u :  [batch_size, d + 1]
            v :  [batch_size, d + 1]
        Return : 
            keepdim :  False [batch_size]
            keepdim :  True  [batch_size, 1]
        """
        d = u.size(1) - 1
        uv = u * v
        uv = torch.cat((-uv.narrow(1, 0, 1), uv.narrow(1, 1, d)), dim = 1) 
        return torch.sum(uv, dim = 1, keepdim = keepdim)

    def from_lorentz_to_poincare(self, x) : 
        """
        Args : 
            u :  [batch_size, d + 1]
        """
        d = x.size(-1) - 1
        return x.narrow(-1, 1, d) / (x.narrow(-1, 0, 1) + 1)

    def from_poincare_to_lorentz(self, x) : 
        """
        Args : 
            u :  [batch_size, d]
        """
        x_norm_square = th_dot(x, x)
        return torch.cat((1 + x_norm_square, 2 * x), dim = 1) / (1 - x_norm_square + self.eps)

    def distance(self, u, v) : 
        d = -LorentzDot.apply(u, v)
        dis = Acosh.apply(d, self.eps)
        return dis

    def normalize(self, w) : 
        """
        Normalize vector such that it is located on the Lorentz
        Args : 
            w :  [batch_size, d + 1]
        """
        d = w.size(-1) - 1
        narrowed = w.narrow(-1, 1, d)
        if self.max_norm : 
            narrowed = torch.renorm(narrowed.view(-1, d), 2, 0, self.max_norm)
        first = 1 + torch.sum(torch.pow(narrowed, 2), dim = -1, keepdim = True)
        first = torch.sqrt(first)
        tmp = torch.cat((first, narrowed), dim = 1)
        return tmp

    def init_embed(self, embed, irange = 1e-2) : 
        embed.weight.data.uniform_(-irange, irange)
        embed.weight.data.copy_(self.normalize(embed.weight.data))

    def rgrad(self, p, d_p) : 
        """Riemannian gradient for Lorentz"""
        u = d_p
        x = p
        u.narrow(-1, 0, 1).mul_(-1)
        u.addcmul_(self.ldot(x, u, keepdim = True).expand_as(x), x)
        return d_p

    def exp_map_zero(self, v) : 
        zeros = torch.zeros_like(v)
        zeros[ : , 0] = 1
        return self.exp_map_x(zeros, v)

    def exp_map_x(self, p, d_p, d_p_normalize = True, p_normalize = True) : 
        if d_p_normalize : 
            d_p = self.normalize_tan(p, d_p)

        ldv = self.ldot(d_p, d_p, keepdim = True)
        nd_p = torch.sqrt(torch.clamp(ldv + self.eps, _eps))

        t = torch.clamp(nd_p, max = self.norm_clip)
        newp = (torch.cosh(t) * p) + (torch.sinh(t) * d_p / nd_p)

        if p_normalize : 
            newp = self.normalize(newp)
        return newp

    def normalize_tan(self, x_all, v_all) : 
        d = v_all.size(1) - 1
        x = x_all.narrow(1, 1, d)
        xv = torch.sum(x * v_all.narrow(1, 1, d), dim = 1, keepdim = True)
        tmp = 1 + torch.sum(torch.pow(x_all.narrow(1, 1, d), 2), dim = 1, keepdim = True)
        tmp = torch.sqrt(tmp)
        return torch.cat((xv / tmp, v_all.narrow(1, 1, d)), dim = 1)

    def log_map_zero(self, y, i = -1) : 
        zeros = torch.zeros_like(y)
        zeros[ : , 0] = 1
        return self.log_map_x(zeros, y)

    def log_map_x(self, x, y, normalize = False) : 
        """Logarithmic map on the Lorentz Manifold"""
        xy = self.ldot(x, y).unsqueeze(-1)
        tmp = torch.sqrt(torch.clamp(xy * xy - 1 + self.eps, _eps))
        v = Acosh.apply(-xy, self.eps) / (
            tmp
        ) * torch.addcmul(y, xy, x)
        if normalize : 
            result = self.normalize_tan(x, v)
        else : 
            result = v
        return result

    def parallel_transport(self, x, y, v) : 
        """Parallel transport for Lorentz"""
        v_ = v
        x_ = x
        y_ = y

        xy = self.ldot(x_, y_, keepdim = True).expand_as(x_)
        vy = self.ldot(v_, y_, keepdim = True).expand_as(x_)
        vnew = v_ + vy / (1 - xy) * (x_ + y_)
        return vnew

    def metric_tensor(self, x, u, v) : 
        return self.ldot(u, v, keepdim = True)



class LorentzDot(Function) : 
    @staticmethod
    def forward(ctx, u, v) : 
        ctx.save_for_backward(u, v)
        return LorentzManifold.ldot(u, v)

    @staticmethod
    def backward(ctx, g) : 
        u, v = ctx.saved_tensors
        g = g.unsqueeze(-1).expand_as(u).clone()
        g.narrow(-1, 0, 1).mul_(-1)
        return g * v, g * u

class Acosh(Function) : 
    @staticmethod
    def forward(ctx, x, eps) :  
        z = torch.sqrt(torch.clamp(x * x - 1 + eps, _eps))
        ctx.save_for_backward(z)
        ctx.eps = eps
        xz = x + z
        tmp = torch.log(xz)
        return tmp

    @staticmethod
    def backward(ctx, g) : 
        z, = ctx.saved_tensors
        z = torch.clamp(z, min = ctx.eps)
        z = g / z
        return z, None