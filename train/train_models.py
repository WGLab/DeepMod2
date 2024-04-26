import math, time, argparse, re,  os, sys
import functools, itertools, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import optim, Tensor
import numpy as np
import multiprocessing as mp
import multiprocessing as mp
import queue
import pickle
from utils import *

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


    
def train(training_dataset, validation_dataset, validation_type, validation_fraction, model_config, epochs, prefix, retrain, batch_size, args_str, seed):
    print('Starting training.' , flush=True)
    torch.manual_seed(seed)
    model_type = model_config['model_type']
    model_save_path = model_config.pop('model_save_path')
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:
        dev = "cpu"
    
    weight_counts=np.array([np.sum(np.eye(2)[np.load(f)['label'].astype(int)],axis=0) for f in itertools.chain.from_iterable(training_dataset)])
    weight_counts=np.sum(weight_counts,axis=0)
    
    
    if model_config['weights']=='equal':
        pos_weight=torch.Tensor(np.array(1.0))
    
    elif model_config['weights']=='auto':
        pos_weight=torch.Tensor(np.array(weight_counts[0]/weight_counts[1]))
    
    else:
        pos_weight=torch.Tensor(np.array(float(model_config['weights'])))

    print('Number of Modified Instances={}\nNumber of Un-Modified Instances={}\nPositive Label Weight={}\n'.format(weight_counts[1],weight_counts[0],pos_weight), flush=True)

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
    
    optimizer = optim.Adam(net.parameters(), lr= model_config['lr'], weight_decay=model_config['l2_coef'])
    
    if retrain:
        checkpoint = torch.load(retrain)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    model_details=str(net.to(torch.device(dev)))
    print(model_details, flush=True)
    num_params=sum(p.numel() for p in net.parameters())
    print('# Parameters=', num_params, flush=True)
    
    config_path=os.path.join(model_save_path, '%s.cfg' %prefix)
    
    with open(config_path, 'wb') as handle:
        pickle.dump(model_config, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    log_file_path=os.path.join(model_save_path, '%s.log' %prefix)
    
    train_w_wo_ref=model_config['train_w_wo_ref']
    include_ref=model_config['include_ref']
    
    if include_ref and train_w_wo_ref:
        list_of_evals=['Normal', 'Without_Ref']
    else:
        list_of_evals=['Normal']
    
    dummy_ref_seq=(4+torch.zeros(batch_size, 21)).type(torch.LongTensor).to(dev)
    
    with open(log_file_path,'w') as log_file:
        log_file.write(args_str)
        log_file.write('\n# Parameters=%d\n' %num_params)
        log_file.write(model_details)
        
        for j in range(epochs):
            net.train()
            
            metrics_train={g:{'TP':0,'FP':0,'FN':0,'loss':0,'len':0,'true':0} for g in list_of_evals}
            metrics_test={g:{'TP':0,'FP':0,'FN':0,'loss':0,'len':0,'true':0} for g in list_of_evals}

            t=time.time()
            
            train_gen=generate_batches_mixed_can_mod(training_dataset, validation_type, validation_fraction,  data_type="train", batch_size=batch_size)
            
            for batch in train_gen:
                batch_x, batch_base_seq, batch_ref_seq, batch_y =batch
                batch_x, batch_base_seq, batch_ref_seq, batch_y=batch_x.to(dev), batch_base_seq.to(dev), batch_ref_seq.to(dev), batch_y.to(dev)
                
                optimizer.zero_grad()
                score= net(batch_x, batch_base_seq, batch_ref_seq)
                loss =  torch.nn.functional.binary_cross_entropy_with_logits(score, batch_y,pos_weight=pos_weight)

                loss.backward()
                optimizer.step()
                
                get_metrics(metrics_train,'Normal', batch_y, score, loss)
                
                if include_ref and train_w_wo_ref:
                    dummy_batch_ref_seq=dummy_ref_seq[:batch_ref_seq.size(0)]
                    optimizer.zero_grad()
                    score= net(batch_x, batch_base_seq, dummy_batch_ref_seq)
                    loss =  torch.nn.functional.binary_cross_entropy_with_logits(score, batch_y,pos_weight=pos_weight)

                    loss.backward()
                    optimizer.step()
                    get_metrics(metrics_train,'Without_Ref', batch_y, score, loss)
                    
            with torch.no_grad():
                net.eval()
                
                if validation_type=='split':
                    test_gen=generate_batches(list(itertools.chain.from_iterable(training_dataset)), validation_type, validation_fraction,  data_type="test", batch_size=batch_size)
                    
                else:
                    test_gen=generate_batches(validation_dataset, validation_type, validation_fraction,  data_type="test", batch_size=batch_size)
                
                for batch in test_gen:
                    batch_x, batch_base_seq, batch_ref_seq, batch_y = batch
                    batch_x, batch_base_seq, batch_ref_seq, batch_y=batch_x.to(dev), batch_base_seq.to(dev), batch_ref_seq.to(dev), batch_y.to(dev)

                    score= net(batch_x, batch_base_seq, batch_ref_seq)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(score, batch_y,pos_weight=pos_weight)

                    get_metrics(metrics_test,'Normal', batch_y, score, loss)
                    
                    if include_ref and train_w_wo_ref:
                        dummy_batch_ref_seq=dummy_ref_seq[:batch_ref_seq.size(0)]
                        score= net(batch_x, batch_base_seq, dummy_batch_ref_seq)
                        loss =  torch.nn.functional.binary_cross_entropy_with_logits(score, batch_y,pos_weight=pos_weight)
                        
                        get_metrics(metrics_test,'Without_Ref', batch_y, score, loss)
                                            
            train_str, _ = get_stats(metrics_train, 'Training')
            test_str, total_test_acc = get_stats(metrics_test, 'Testing')
                
            epoch_log='\n\nEpoch %d: #Train=%d  #Test=%d  Time=%.4f\n%s\n\n%s'\
                  %(j+1, sum(x['len'] for x in metrics_train.values()), sum(x['len'] for x in metrics_test.values()), time.time()-t, 
                    train_str, test_str)
            print(epoch_log, flush=True)
            log_file.write(epoch_log)
            
            model_path=os.path.join(model_save_path, 'model.epoch%d.%.4f' %(j+1, total_test_acc))
            torch.save({
            'epoch': j+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, model_path)
    
    return net

if __name__=='__main__':
    start_time=time.time()
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--mixed_training_dataset", nargs='*', help='Training dataset with mixed labels. A whitespace separated list of folders containing .npz files or paths to individual .npz files.')

    parser.add_argument("--can_training_dataset", nargs='*', help='Training dataset with unmodified or canonical base labels. A whitespace separated list of folders containing .npz files or paths to individual .npz files.')
    parser.add_argument("--mod_training_dataset", nargs='*', help='Training dataset with modified labels. A whitespace separated list of folders containing .npz files or paths to individual .npz files.')

    parser.add_argument("--validation_type",choices=['split','dataset'], help='How the validation is performed. "split" means that a fraction of training dataset, specified by --validation_fraction, will be used for validation. "dataset" means that additional validation dataset is provided via --validation_input parameter.',default="split")
    parser.add_argument("--validation_fraction", help='Fraction of training dataset to use for validation if --validation_type is set to "split", otherwise ignored.', type=float, default=0.2)
    parser.add_argument("--validation_dataset",nargs='*', help='Validation dataset if --validation_type is set to dataset. A whitespace separated list of folders containing .npz files or paths to individual .npz files.')

    parser.add_argument("--prefix", help='Prefix name for the model checkpoints', default='model')

    parser.add_argument("--weights", help='Weight of positive(modified) label used in binary cross entropy loss, the negative(unmodified) label will always have a fixed weight of 1. Higher weight for modified labels will favor recall and lower weight will favor precision. Choices are "equal", "auto" or your can specify the weight of positive(modified) labels. "equal" assigns a weight of 1 to modified labels, "auto" assigns a weight=num_negative_samples/num_positive_samples to modified labels.', default='equal')


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
    parser.add_argument("--pe_dim", help='Dimension for positional encoding/embedding for transformer model.', type=int, default=16)
    parser.add_argument("--pe_type", help='Type of positional encoding/embedding for transformer model. fixed is sinusoid, embedding is is dictionary lookup, parameter weight matrix.', type=str, choices=['fixed', 'embedding', 'parameter'], default='fixed')
    parser.add_argument("--nhead", help='Number of self-attention heads in transformer encoder  for transformer model.', type=int, default=4)
    parser.add_argument("--include_ref", help='Whether to include reference sequence as features. Recommended.', default=False, action='store_true')
    parser.add_argument("--train_w_wo_ref", help='Include examples with reference and without reference sequence. Recommended if you will be using referenve free modification detection.', default=False, action='store_true')

    parser.add_argument("--lr", help='Learning rate', type=float, default=1e-4)
    parser.add_argument("--l2_coef", help='L2 regularization coefficient', type=float, default=1e-5)
    parser.add_argument("--seed", help='Random seed to use in pytorch for reproducibility or reinitialization of weights', default=None)
    
    args = parser.parse_args()
    
    os.makedirs(args.model_save_path, exist_ok=True)

    mixed_training_dataset=get_files(args.mixed_training_dataset)
    can_training_dataset=get_files(args.can_training_dataset)
    mod_training_dataset=get_files(args.mod_training_dataset)

    validation_dataset=get_files(args.validation_dataset)
    validation_type=args.validation_type
    validation_fraction=args.validation_fraction

    valid_data, window, norm_type = check_training_files(mixed_training_dataset, can_training_dataset,\
                                           mod_training_dataset, validation_dataset)
    
    if not valid_data:
        sys.exit(3)

    model_config = dict(model_dims=(2*window+1,10),window=window, model_type=args.model_type,
    num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
    num_fc=args.num_fc, embedding_dim=args.embedding_dim,
    embedding_type=args.embedding_type, include_ref=args.include_ref,
    pe_dim=args.pe_dim, nhead=args.nhead, pe_type=args.pe_type,
    l2_coef=args.l2_coef, lr=args.lr, model_save_path=args.model_save_path, fc_type=args.fc_type,
    train_w_wo_ref=args.train_w_wo_ref, weights=args.weights, norm_type=norm_type)

    args_dict=vars(args)
    args_str=''.join('%s: %s\n' %(k,str(v)) for k,v in args_dict.items())
    print(args_str, flush=True)
    
    seed =random.randint(0, 0xffff_ffff_ffff_ffff) if args.seed is None else int(args.seed)
    
    training_dataset = [mixed_training_dataset, can_training_dataset, mod_training_dataset]
    
    with open(os.path.join(args.model_save_path,'args'),'w') as file:
        file.write('Command: python %s\n\n\n' %(' '.join(sys.argv)))
        file.write('------Parameters Used For Running DeepMod2------\n')
        for k in vars(args):
            file.write('{}: {}\n'.format(k,vars(args)[k]) )
            
    res=train(training_dataset, validation_dataset, validation_type, validation_fraction, model_config, epochs=args.epochs,prefix=args.prefix, retrain=args.retrain, batch_size=args.batch_size, args_str=args_str, seed=seed)
    
    print('Time taken=%.4f' %(time.time()-start_time), flush=True)
