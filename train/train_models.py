import math, time, argparse, re
from torch.utils.data import Dataset,DataLoader
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import numpy as np
import multiprocessing as mp
import os, sys
from torch import optim
from torch import nn, Tensor
import pathlib
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import queue
import pickle
from pathlib import Path

class OneHotEncode(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes=num_classes
    def forward(self, x: Tensor) -> Tensor:
        return F.one_hot(x, self.num_classes)

class RefandReadEmbed(nn.Module):
    def __init__(self, embedding_dim, embedding_type):
        super().__init__()
        
        self.embedding_depth=0
        
        if embedding_type=='one_hot':
            self.read_emb=OneHotEncode(4)
            self.embedding_depth+=4
            
            self.ref_emb=OneHotEncode(5)
            self.embedding_depth+=5
            
        elif embedding_type=='learnable':
            self.read_emb=nn.Embedding(4, embedding_dim)
            self.embedding_depth+=embedding_dim
            
            self.ref_emb=nn.Embedding(5, embedding_dim)
            self.embedding_depth+=embedding_dim
            
    def forward(self, batch_base_seq, batch_ref_seq):
        batch_base_seq_emb=self.read_emb(batch_base_seq)
        batch_ref_seq_emb=self.ref_emb(batch_ref_seq)
        
        return torch.cat((batch_base_seq_emb, batch_ref_seq_emb), 2)

class ReadEmbed(nn.Module):
    def __init__(self, embedding_dim, embedding_type):
        super().__init__()
        
        self.embedding_depth=0
        
        if embedding_type=='one_hot':
            self.read_emb=OneHotEncode(4)
            self.embedding_depth+=4

        elif embedding_type=='learnable':
            self.read_emb=nn.Embedding(4, embedding_dim)
            self.embedding_depth+=embedding_dim
            

    def forward(self, batch_base_seq, batch_ref_seq):
        batch_base_seq_emb=self.read_emb(batch_base_seq)
        
        return batch_base_seq_emb
    
class SeqEmbed(nn.Module):
    def __init__(self, embedding_dim, embedding_type, include_ref):
        super().__init__()
        
        self.embedding_depth=0
        
        if include_ref:
            self.seq_emb=RefandReadEmbed(embedding_dim, embedding_type)
            
        else:
            self.seq_emb=ReadEmbed(embedding_dim, embedding_type)
        
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

class PositionalEmbedding(nn.Module):
    def __init__(self, pe_dim: int, max_len: int):
        super().__init__()
        pos=torch.arange(max_len)
        self.register_buffer('pos', pos)
        self.pe=nn.Embedding(max_len, pe_dim)

    def forward(self, x: Tensor) -> Tensor:
        x_pos=self.pe(self.pos)
        x_pos=torch.Tensor.repeat(x_pos,(x.size(0),1,1)) 
        x = torch.cat((x, x_pos),2)
        return x

class PositionalParameter(nn.Module):
    def __init__(self, pe_dim: int, max_len: int):
        super().__init__()
        
        self.pe=torch.nn.Parameter(torch.randn(max_len, pe_dim)) 

    def forward(self, x: Tensor) -> Tensor:
        x_pos=torch.Tensor.repeat(self.pe,(x.size(0),1,1)) 
        x = torch.cat((x, x_pos),2)
        return x

class ClassifierMiddle(nn.Module):
    def __init__(self, in_dim: int, num_fc: int, model_len: int):
        super().__init__()
        self.mid = model_len//2
        self.fc = nn.Linear(in_dim, num_fc)
        self.out = nn.Linear(num_fc,1)
        
    def forward(self, x):
        x = F.relu(self.fc(x[:,self.mid, :]))
        x=self.out(x)
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
    def __init__(self, model_dims, num_layers, dim_feedforward, num_fc, embedding_dim, embedding_type, include_ref, fc_type):
        super(BiLSTM, self).__init__()
        
        self.emb=SeqEmbed(embedding_dim, embedding_type, include_ref)
        self.model_len=model_dims[0]
        self.model_depth=model_dims[1]+self.emb.embedding_depth
        
        self.bilstm = nn.LSTM(input_size=self.model_depth, hidden_size=dim_feedforward, num_layers=num_layers, bidirectional=True, batch_first = True)
        
        if fc_type=='middle':
            self.classifier=ClassifierMiddle(in_dim=dim_feedforward*2, num_fc=num_fc, model_len=self.model_len)
        
        else:
            self.classifier=ClassifierAll(in_dim=self.model_len*dim_feedforward*2, num_fc=num_fc)

    def forward(self, batch_x, batch_base_seq, batch_ref_seq):
        seq_emb=self.emb(batch_base_seq, batch_ref_seq)
        x=torch.cat((batch_x, seq_emb), 2)
        x, _=self.bilstm(x)
        x = self.classifier(x)
        
        return x
    
class TransformerModel(nn.Module):
    def __init__(self, model_dims, num_layers, dim_feedforward, num_fc, embedding_dim, embedding_type, include_ref, pe_dim, nhead, pe_type, fc_type):
        super(TransformerModel, self).__init__()
        
        self.emb=SeqEmbed(embedding_dim, embedding_type, include_ref)
        self.model_len=model_dims[0]
        
        if pe_type=='fixed':
            self.pe_block=PositionalEncoding(pe_dim=pe_dim, max_len=self.model_len)
        
        elif pe_type=='embedding':
            self.pe_block=PositionalEmbedding(pe_dim=pe_dim, max_len=self.model_len)
            
        elif pe_type=='parameter':                
            self.pe_block=PositionalParameter(pe_dim=pe_dim, max_len=self.model_len)

        self.model_depth=model_dims[1]+self.emb.embedding_depth+pe_dim
        self.pad_length=math.ceil(self.model_depth/nhead)*nhead-self.model_depth        
        pad=torch.zeros(1,self.model_len, self.pad_length)
        self.register_buffer('pad', pad)
        self.model_depth+=self.pad_length
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_depth, nhead=nhead, dim_feedforward=dim_feedforward, dropout=0, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        if fc_type=='middle':
            self.classifier=ClassifierMiddle(in_dim=self.model_depth, num_fc=num_fc, model_len=self.model_len)
        
        else:
            self.classifier=ClassifierAll(in_dim=self.model_len*self.model_depth, num_fc=num_fc)

    def forward(self, batch_x, batch_base_seq, batch_ref_seq):
        seq_emb=self.emb(batch_base_seq, batch_ref_seq)
        x=torch.cat((batch_x, seq_emb), 2)
        x=self.pe_block(x)        
        x_pad=torch.Tensor.repeat(self.pad,(x.size(0),1,1)) 
        x = torch.cat((x, x_pad),2)
        
        x=self.transformer_encoder(x)
        x = self.classifier(x)
        
        return x
    
@functools.lru_cache(maxsize=128)
def read_from_cache(file_name):
    return read_from_file(file_name)

def read_from_file(file_name):
    data=np.load(file_name)
    mat=data['mat']
    base_qual=data['base_qual']
    
    features=np.dstack((mat, base_qual[:,:,np.newaxis]))
    features=torch.Tensor(features)
    
    base_seq=torch.Tensor(data['base_seq']).type(torch.LongTensor)
    ref_seq=torch.Tensor(data['ref_seq']).type(torch.LongTensor)

    labels=torch.Tensor(data['label'][:,np.newaxis])
    
    return features, base_seq, ref_seq, labels

def generate_batches(files,  data_type='train', batch_size=512):
    counter = 0
    
    print_freq=max(1, len(files)//10)
    pattern = r'/(HG[^/]+)/'
    
    while counter<len(files):
        file_name = files[counter]

        counter +=1
        
        if data_type=='test':
            data=read_from_cache(file_name)
            genome = re.findall(pattern, str(file_name))[0]

        else:
            data=read_from_file(file_name)
            genome = re.findall(pattern, str(file_name))[0]

        features, base_seq, ref_seq, labels=data
        
        for local_index in range(0, labels.shape[0], batch_size):
            batch_x=features[local_index:(local_index + batch_size)]
            batch_base_seq=base_seq[local_index:(local_index + batch_size)]
            batch_ref_seq=ref_seq[local_index:(local_index + batch_size)]
            batch_y=labels[local_index:(local_index + batch_size)]
            
            

            yield batch_x, batch_base_seq, batch_ref_seq, batch_y, genome
        
        if counter%print_freq==0:
            print('.', end='',flush=True)

def train(training_input, testing_input, model_config, epochs, prefix, retrain, batch_size, args_str):
    
    print('Starting training.' , flush=True)
    
    model_type, model_save_path = model_config['model_type'], model_config['model_save_path']
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:
        dev = "cpu"

    
    loss_list={'train':[],'test':[]}
    acc_list={'train':[],'test':[]}

    model_dims=(21,10)
    
    if model_type=='bilstm':
        net = BiLSTM(model_dims=model_config['model_dims'], num_layers=model_config['num_layers'], \
                     dim_feedforward=model_config['dim_feedforward'], \
                     num_fc=model_config['num_fc'], embedding_dim=model_config['embedding_dim'], \
                     embedding_type=model_config['embedding_type'], include_ref=model_config['include_ref'], \
                     fc_type=model_config['fc_type']);
    
    elif model_type=='transformer':
        net = TransformerModel(model_dims=model_config['model_dims'], num_layers=model_config['num_layers'], \
                     dim_feedforward=model_config['dim_feedforward'], \
                     num_fc=model_config['num_fc'], embedding_dim=model_config['embedding_dim'], \
                     embedding_type=model_config['embedding_type'], include_ref=model_config['include_ref'],\
                     pe_dim=model_config['pe_dim'], nhead=model_config['nhead'], \
                               pe_type=model_config['pe_type'], fc_type=model_config['fc_type']);
        
    net.to(dev);
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr= model_config['lr'], weight_decay=model_config['l2_coef'])
    
    if retrain:
        checkpoint = torch.load(retrain)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    model_details=str(net.to(torch.device(dev)))
    print(model_details, flush=True)
    num_params=sum(p.numel() for p in net.parameters())
    print('# Parameters=', num_params, flush=True)
    
    log_file_path=os.path.join(model_save_path, '%s.log.%s' %(prefix, os.environ['SLURM_JOB_ID']))
    pattern = r'/(HG[^/]+)/'
    list_of_genomes=list(set(re.findall(pattern, str(file_name))[0] for file_name in testing_input+training_input))
    
    train_w_wo_ref=model_config['train_w_wo_ref']
    include_ref=model_config['include_ref']
    
    if include_ref and train_w_wo_ref:
        list_of_genomes=list_of_genomes+[x+'_wo_ref' for x in list_of_genomes]
    
    dummy_ref_seq=(4+torch.zeros(batch_size, 21)).type(torch.LongTensor).to(dev)
    
    with open(log_file_path,'w') as log_file:
        log_file.write(args_str)
        log_file.write('\n# Parameters=%d\n' %num_params)
        log_file.write(model_details)
        
        for j in range(epochs):
            net.train()
            acc_train={g:0 for g in list_of_genomes}
            loss_train={g:0 for g in list_of_genomes}
            len_train={g:1 for g in list_of_genomes}
            
            acc_test={g:0 for g in list_of_genomes}
            loss_test={g:0 for g in list_of_genomes}
            len_test={g:1 for g in list_of_genomes}

            t=time.time() 
            train_gen=generate_batches(training_input,  data_type='train', batch_size=batch_size)

            for batch in train_gen:
                batch_x, batch_base_seq, batch_ref_seq, batch_y, genome =batch
                batch_x, batch_base_seq, batch_ref_seq, batch_y=batch_x.to(dev), batch_base_seq.to(dev), batch_ref_seq.to(dev), batch_y.to(dev)
                
                optimizer.zero_grad()
                score= net(batch_x, batch_base_seq, batch_ref_seq)
                loss =  criterion(score, batch_y)

                loss.backward()
                optimizer.step()

                len_train[genome]+=len(batch_y)
                acc_train[genome]+=sum(batch_y==(score>=0)).cpu()
                loss_train[genome]+=loss.item()*len(batch_y)
                
                if include_ref and train_w_wo_ref:
                    dummy_batch_ref_seq=dummy_ref_seq[:batch_ref_seq.size(0)]
                    optimizer.zero_grad()
                    score= net(batch_x, batch_base_seq, dummy_batch_ref_seq)
                    loss =  criterion(score, batch_y)

                    loss.backward()
                    optimizer.step()

                    len_train[genome+'_wo_ref']+=len(batch_y)
                    acc_train[genome+'_wo_ref']+=sum(batch_y==(score>=0)).cpu()
                    loss_train[genome+'_wo_ref']+=loss.item()*len(batch_y)
                    
            with torch.no_grad():
                net.eval()
                test_gen=generate_batches(testing_input,  data_type='test', batch_size=batch_size)
                
                for batch in test_gen:
                    batch_x, batch_base_seq, batch_ref_seq, batch_y, genome = batch
                    batch_x, batch_base_seq, batch_ref_seq, batch_y=batch_x.to(dev), batch_base_seq.to(dev), batch_ref_seq.to(dev), batch_y.to(dev)

                    score= net(batch_x, batch_base_seq, batch_ref_seq)
                    loss =  criterion(score, batch_y)

                    len_test[genome]+=len(batch_y)
                    acc_test[genome]+=sum(batch_y==(score>=0)).cpu()
                    loss_test[genome]+=loss.item()*len(batch_y)
                    
                    if include_ref and train_w_wo_ref:
                        dummy_batch_ref_seq=dummy_ref_seq[:batch_ref_seq.size(0)]
                        score= net(batch_x, batch_base_seq, dummy_batch_ref_seq)
                        loss =  criterion(score, batch_y)

                        len_test[genome+'_wo_ref']+=len(batch_y)
                        acc_test[genome+'_wo_ref']+=sum(batch_y==(score>=0)).cpu()
                        loss_test[genome+'_wo_ref']+=loss.item()*len(batch_y)                    
                    
            train_loss_str='Loss     Train:'
            for g in sorted(loss_train.keys()):
                train_loss_str+='  %s: %.4f' %(g, loss_train[g]/len_train[g])
            
            total_train_loss=sum(loss_train.values())/sum(len_train.values())
            train_loss_str+='  %s: %.4f' %('Total', total_train_loss)

            train_acc_str='Accuracy Train:'
            for g in sorted(acc_train.keys()):
                train_acc_str+='  %s: %.4f' %(g, acc_train[g]/len_train[g])

            total_train_acc=float(sum(acc_train.values())/sum(len_train.values()))
            train_acc_str+='  %s: %.4f' %('Total', total_train_acc)

            test_loss_str='Loss     Validation:'
            for g in sorted(loss_test.keys()):
                test_loss_str+='    %s: %.4f' %(g, loss_test[g]/len_test[g])
            
            total_test_loss=sum(loss_test.values())/sum(len_test.values())
            test_loss_str+='  %s: %.4f' %('Total', total_test_loss)

            test_acc_str='Accuracy Validation:'
            for g in sorted(acc_test.keys()):
                test_acc_str+='  %s: %.4f' %(g, acc_test[g]/len_test[g])
            
            total_test_acc=float(sum(acc_test.values())/sum(len_test.values()))
            test_acc_str+='  %s: %.4f' %('Total', total_test_acc)

            epoch_log='\nEpoch %d: #Train=%d  #Test=%d  Time=%.4f\n%s\n%s\n%s\n%s'\
                  %(j+1, sum(len_train.values()), sum(len_test.values()), time.time()-t, train_loss_str, train_acc_str, test_loss_str, test_acc_str)
            print(epoch_log, flush=True)
            log_file.write(epoch_log)
            
            model_path=os.path.join(model_save_path, 'model.epoch%d.%.4f' %(j+1, total_test_acc))
            torch.save({
            'epoch': j+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, model_path)
    
    return

if __name__=='__main__':
    start_time=time.time()
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--training_input", help='Training input. There are three ways to specify the input: 1) path to a folder containing .npz files in which case all npz files will be used for training, 2) path to a single .npz file, 3) path to a text file containing paths of .npz files to use for training.')
    parser.add_argument("--testing_input", help='Testing input. There are three ways to specify the input: 1) path to a folder containing .npz files in which case all npz files will be used for testing, 2) path to a single .npz file, 3) path to a text file containing paths of .npz files to use for testing.')
    parser.add_argument("--prefix", help='Prefix name for the model checkpoints', default='model')

    parser.add_argument("--model_save_path", help='Folder path for saving model checkpoints')
    parser.add_argument("--epochs", help='Number of total epochs', default=100, type=int)
    parser.add_argument("--batch_size", help='Batch Size', default=256, type=int)
    parser.add_argument("--retrain", help='Path to a model for retraining', default=None)

    parser.add_argument("--fc_type", help='Type of full connection to use in the classifier.', type=str, default='all', choices=['middle', 'all'])
    parser.add_argument("--model_type", help='Type of model to use', type=str, choices=['bilstm', 'transformer'])

    parser.add_argument("--num_layers", help='Number of transformer encoder or BiLSTM layers', type=int, default=3)
    parser.add_argument("--dim_feedforward", help='Dimension of feedforward layers in  transformer encoder or size of hidden units in BiLSTM layers', type=int, default=100)
    parser.add_argument("--num_fc", help='Size of fully connected layer between encoder/BiLSTM and classifier', type=int, default=16)
    parser.add_argument("--embedding_dim", help='Size of embedding dimension for read and reference bases', type=int, default=4)
    parser.add_argument("--embedding_type", help='Type of embedding for bases', type=str, choices=['learnable', 'one_hot'], default='one_hot')
    parser.add_argument("--pe_dim", help='Dimension for positional encoding/embedding', type=int, default=16)
    parser.add_argument("--pe_type", help='Type of positional encoding/embedding. fixed is sinusoid, embedding is is dictionary lookup, parameter weight matrix.', type=str, choices=['fixed', 'embedding', 'parameter'], default='fixed')
    parser.add_argument("--nhead", help='Number of self-attention heads in transformer encoder.', type=int, default=4)
    parser.add_argument("--include_ref", help='Whether to include reference positions as features', default=False, action='store_true')
    parser.add_argument("--train_w_wo_ref", help='Include both ref and without ref.', default=False, action='store_true')

    parser.add_argument("--lr", help='Learning rate', type=float, default=1e-4)
    parser.add_argument("--l2_coef", help='L2 regularization coefficient', type=float, default=1e-5)

    args = parser.parse_args()
              
    training_input=args.training_input
    testing_input=args.testing_input

    os.makedirs(args.model_save_path, exist_ok=True)

    if os.path.isdir(training_input):
        training_input=list(Path(training_input).rglob("*.npz"))
    elif training_input[-4:]=='.npz':
        training_input=[training_input]
    else:
        with open(training_input,'r') as training_input_file:
            training_input=training_input_file.read().splitlines()

    if os.path.isdir(testing_input):
        testing_input=list(Path(testing_input).rglob("*.npz"))
    elif testing_input[-4:]=='.npz':
        testing_input=[testing_input]
    else:
        with open(testing_input,'r') as testing_input_file:
            testing_input=testing_input_file.read().splitlines()

    model_config = dict(model_dims=(21,10), model_type=args.model_type,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
        num_fc=args.num_fc, embedding_dim=args.embedding_dim,
        embedding_type=args.embedding_type, include_ref=args.include_ref,
        pe_dim=args.pe_dim, nhead=args.nhead, pe_type=args.pe_type,
        l2_coef=args.l2_coef, lr=args.lr, model_save_path=args.model_save_path, fc_type=args.fc_type,
        train_w_wo_ref=args.train_w_wo_ref)

    args_dict=vars(args)
    args_str=''.join('%s: %s\n' %(k,str(v)) for k,v in args_dict.items())
    print(args_str, flush=True)

    res=train(training_input, testing_input, model_config, epochs=args.epochs,prefix=args.prefix, retrain=args.retrain, batch_size=args.batch_size, args_str=args_str)
    
    print('Time taken=%.4f' %(time.time()-start_time), flush=True)
