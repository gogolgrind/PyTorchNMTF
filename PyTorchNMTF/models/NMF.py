import torch
from torch import nn
from ipypb import ipb
from collections import  defaultdict
from .. import objectives

class NMF(nn.Module):
    
    def __init__(self,X,
                 k = 10, solver = 'mu', n_iter = 10, eps = 1e-7, 
                 alpha = 0.99,
                 loss = 'l2',
                 lr = 1e-2, verbose = False):
        super(NMF, self).__init__()
        self.n_iter = n_iter
        self.k = k
        self.loss = loss
        self.lr = lr
        self.alpha = alpha
        self.verbose = verbose
        self.solver = solver
        self.eps = eps
        self.decomposed = False
        self.is_cuda = False
        self.report = defaultdict(list)
        self.__initfact__(X)
        
    def __initfact__(self,X):
        self.n,self.m = X.shape
        self.X = torch.from_numpy(X)
        self.scale = torch.sqrt(torch.mean(self.X) / self.k)
        W = torch.abs(torch.rand([self.n,self.k])*self.scale)
        H = torch.abs(torch.rand([self.k,self.m])*self.scale)
        self.W = torch.nn.Parameter(W)
        self.H = torch.nn.Parameter(H)
        # for autograd solver
        self.opt = torch.optim.RMSprop([self.W,self.H], alpha=self.alpha, lr=self.lr,weight_decay=1e-6)
        if self.loss == 'l2':
            self.loss_fn = objectives.l2
        elif self.loss == 'kl':
            self.loss_fn = objectives.kl_dev
        
    def to(self,device):
        self.is_cuda = (device == 'cuda')
        if self.is_cuda:
            self.X = self.X.to('cuda')
        return super(NMF, self).to(device)
    
    def plus(self,X):
        X[X < 0] = self.eps
        return X
    
    def __mu__(self,epoch):
        """
            multiplicative update, explisit form.
        """
        W,H = self.W,self.H
        WT = torch.transpose(W,0,1)
        HT = torch.transpose(H,0,1)
        XHT = self.X @ HT
        WHHT = W @ H @ HT
        W = W * (XHT)/(WHHT+self.eps)
        WTX = WT @ self.X
        WTWH = WT @ W @ H
        H = H * (WTX)/(WTWH+self.eps)
        self.W = torch.nn.Parameter(W)
        self.H = torch.nn.Parameter(H)
        l = self.loss_fn(self.X,self.W @ self.H)
        return l.item()
    
    def __autograd__(self,epoch):
        """
           autograd update, with gradient projection
        """
        self.opt.zero_grad()
        l = self.loss_fn(self.X,self.W @ self.H)
        l.backward()
        self.opt.step()
        ## grad projection
        for p in self.parameters():
            p.data = self.plus(p.data)
        return l.item()

        
    def __update__(self,epoch):
        if self.solver == 'mu':
            l = self.__mu__(epoch)
        elif self.solver == 'autograd':
            l = self.__autograd__(epoch)
        else:
            raise NotImplementedError
        self.report['epoch'].append(epoch)
        self.report['loss'].append(l)
        if self.verbose and epoch % 500 == 0:
            print("%d\tloss: %.4f"%(epoch,l))
        assert self.is_nonneg()
            
    def fit(self):
        it = range(self.n_iter)
        if self.verbose:
            it = ipb(it)
        for e in it:
            self.__update__(e)
        self.decomposed = True
        return self
    
    def show_report(self):
        return pd.DataFrame(self.report)
    
    def is_nonneg(self):
        return bool(torch.all(self.W >= 0) and torch.all(self.H >= 0))
    
    def forward(self, H):
        return self.W @ H
    
    def fit_transform(self):
        if not self.decomposed:
            self.fit(X)
        return [self.W, self.H]
    
    def mse(self):
        err = ( (self.X - self.W @ self.H) ** 2 ).mean().detach().cpu().numpy()
        return float(err)
    
    def inverse_trainsform(self):
        return (self.W @ self.H).detach()