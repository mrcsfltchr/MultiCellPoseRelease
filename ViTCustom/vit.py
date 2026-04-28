import torch
from torch  import nn
import torch.nn.Functional as F

class MHA(nn.Module):
    
    def __init__(self,hidden,num_heads,embed_dim, batch = 8):
        
        super(MHA,self).__init__()
        
        #query
        self.query = nn.Linear(batch,num_heads,hidden,embed_dim)
        self.key = nn.Linear(batch,num_heads,hidden,embed_dim)
        self.values  = nn.Linear(batch,num_heads,hidden,embed_dim)
        
    def forward(self,x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attention = torch.exp(torch.matmul(q,k.t()))
        
        return attention*v