import torch, math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor

class OneHotEncode(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes=num_classes
    def forward(self, x: Tensor) -> Tensor:
        return F.one_hot(x, self.num_classes)

class RefandReadEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding_depth=0
        
        self.read_emb=OneHotEncode(4)
        self.embedding_depth+=4

        self.ref_emb=OneHotEncode(5)
        self.embedding_depth+=5
            
            
    def forward(self, batch_base_seq, batch_ref_seq):
        batch_base_seq_emb=self.read_emb(batch_base_seq)
        batch_ref_seq_emb=self.ref_emb(batch_ref_seq)
        
        return torch.cat((batch_base_seq_emb, batch_ref_seq_emb), 2)
    
class SeqEmbed(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding_depth=0
        
        self.seq_emb=RefandReadEmbed()
        
        self.embedding_depth=self.seq_emb.embedding_depth
        
    def forward(self, batch_base_seq, batch_ref_seq):
        return self.seq_emb(batch_base_seq, batch_ref_seq)
            
class PositionalEncoding(nn.Module):
    def __init__(self, pe_dim: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, pe_dim, 2) * (-math.log(pe_dim) / (pe_dim)))
        pe = torch.zeros(1, max_len, pe_dim)
        pe[0,:, 0::2] = torch.sin(position * div_term)
        pe[0,:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x_pos=torch.Tensor.repeat(self.pe,(x.size(0),1,1)) 
        x = torch.cat((x, x_pos),2)
        return x
        
class ClassifierAll(nn.Module):
    def __init__(self, in_dim: int, num_fc: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_fc)
        self.out = nn.Linear(num_fc,1)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        x=self.out(x)
        return x
    
class BiLSTM(nn.Module):
    def __init__(self, model_dims, num_layers, dim_feedforward, num_fc):
        super(BiLSTM, self).__init__()
        
        self.emb=SeqEmbed()
        self.model_len=model_dims[0]
        self.model_depth=model_dims[1]+self.emb.embedding_depth
        
        self.bilstm = nn.LSTM(input_size=self.model_depth, hidden_size=dim_feedforward, num_layers=num_layers, bidirectional=True, batch_first = True)
        
        self.classifier=ClassifierAll(in_dim=self.model_len*dim_feedforward*2, num_fc=num_fc)

    def forward(self, batch_x, batch_base_seq, batch_ref_seq):
        seq_emb=self.emb(batch_base_seq, batch_ref_seq)
        x=torch.cat((batch_x, seq_emb), 2)
        x, _=self.bilstm(x)
        x = self.classifier(x)
        x=torch.nn.functional.sigmoid(x) 
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, model_dims, num_layers, dim_feedforward, num_fc, pe_dim, nhead):
        super(TransformerModel, self).__init__()
        
        self.emb=SeqEmbed()
        self.model_len=model_dims[0]
        
        self.pe_block=PositionalEncoding(pe_dim=pe_dim, max_len=self.model_len)
        
        self.model_depth=model_dims[1]+self.emb.embedding_depth+pe_dim
        self.pad_length=math.ceil(self.model_depth/nhead)*nhead-self.model_depth        
        pad=torch.zeros(1,self.model_len, self.pad_length)
        self.register_buffer('pad', pad)
        self.model_depth+=self.pad_length
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_depth, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier=ClassifierAll(in_dim=self.model_len*self.model_depth, num_fc=num_fc)

    def forward(self, batch_x, batch_base_seq, batch_ref_seq):
        seq_emb=self.emb(batch_base_seq, batch_ref_seq)
        x=torch.cat((batch_x, seq_emb), 2)
        x=self.pe_block(x)        
        x_pad=torch.Tensor.repeat(self.pad,(x.size(0),1,1)) 
        x = torch.cat((x, x_pad),2)
        
        x=self.transformer_encoder(x)
        x = self.classifier(x)
        x=torch.nn.functional.sigmoid(x) 
        return x
