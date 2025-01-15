import numpy as np
import torch, os, random
from pathlib import Path
from sklearn.model_selection import train_test_split
import sklearn
import itertools

def get_files(input_list):
    out_list=[]
    if type(input_list)==type(None):
        return out_list
    for item in input_list:
        
        if os.path.isdir(item):
            out_list.extend(list(Path(item).rglob("*.npz")))

        elif item[-4:]=='.npz':
            out_list.append(item)        
    
    random.seed(0)
    random.shuffle(out_list)
    return out_list

def read_from_file(file_name, validation_type, validation_fraction, data_type, get_pos=False):
    data=np.load(file_name)
    mat=data['mat']
    base_qual=data['base_qual']
    
    features=np.dstack((mat, base_qual[:,:,np.newaxis]))
    features=torch.Tensor(features)
    
    base_seq=torch.Tensor(data['base_seq']).type(torch.LongTensor)
    ref_seq=torch.Tensor(data['ref_seq']).type(torch.LongTensor)

    labels=torch.Tensor(data['label'][:,np.newaxis])
    
    if get_pos:
        pos=data['ref_coordinates']
        chrom=data['ref_name']
        pos_data=np.vstack([chrom, pos]).T
        
        return features, base_seq, ref_seq, labels, pos_data
        
    if validation_type=='split':
        features_train, features_test, base_seq_train, base_seq_test, \
        ref_seq_train, ref_seq_test, labels_train, labels_test=\
        train_test_split(features, base_seq, ref_seq, labels, test_size=validation_fraction, random_state=42)
        if data_type=='train':
            return features_train, base_seq_train, ref_seq_train, labels_train
        else:
            return features_test, base_seq_test, ref_seq_test, labels_test
        
    else:
        return features, base_seq, ref_seq, labels

def check_training_files(mixed_training_dataset, can_training_dataset,\
                                           mod_training_dataset, validation_dataset):
    norm_type=[str(np.load(file)['norm_type']) for file in itertools.chain.from_iterable([mixed_training_dataset, can_training_dataset,\
                                           mod_training_dataset, validation_dataset])]
    window=[int(np.load(file)['window']) for file in itertools.chain.from_iterable([mixed_training_dataset, can_training_dataset,\
                                               mod_training_dataset, validation_dataset])]
    strides_per_base=[int(np.load(file)['strides_per_base']) for file in itertools.chain.from_iterable([mixed_training_dataset, can_training_dataset,\
                                               mod_training_dataset, validation_dataset])]
    
    model_depth=[int(np.load(file)['model_depth']) for file in itertools.chain.from_iterable([mixed_training_dataset, can_training_dataset,\
                                               mod_training_dataset, validation_dataset])]
    if len(set(window))==1 and len(set(norm_type))==1 and len(set(strides_per_base))==1:
        return True, window[0], norm_type[0], strides_per_base[0], model_depth[0]
    
    elif len(set(window))>1:
        print('Inconsistent dataset with multiple window sizes')
        
    elif len(set(norm_type))>1:
        print('Inconsistent dataset with multiple normalization types')
        
    elif len(set(strides_per_base))>1:
        print('Inconsistent dataset with multiple strides_per_base')
        
    return False, window, norm_type, strides_per_base, model_depth

def generate_batches(files, validation_type, validation_fraction, data_type, batch_size=512):
    counter = 0
    
    print_freq=max(1, len(files)//10)
    
    while counter<len(files):
        file_name = files[counter]

        counter +=1
        
        data=read_from_file(file_name, validation_type, validation_fraction, data_type)

        features, base_seq, ref_seq, labels=data
        batch_size=max(batch_size,1)
        for local_index in range(0, labels.shape[0], batch_size):
            batch_x=features[local_index:(local_index + batch_size)]
            batch_base_seq=base_seq[local_index:(local_index + batch_size)]
            batch_ref_seq=ref_seq[local_index:(local_index + batch_size)]
            batch_y=labels[local_index:(local_index + batch_size)]          

            yield batch_x, batch_base_seq, batch_ref_seq, batch_y
        
        if counter%print_freq==0:
            print('.', end='',flush=True)

def generate_batches_mixed_can_mod(data_file_list, validation_type, validation_fraction, data_type, batch_size=128):
    
    data_file_list=[x for x in data_file_list if len(x)!=0]

    sizes=np.array([sum(len(np.load(f)['label']) for f in data) for data in data_file_list])
    batch_sizes=(batch_size*sizes/np.sum(sizes)).astype(int)
    batch_sizes[0]=batch_sizes[0]+batch_size-np.sum(batch_sizes)
    
    generators=[generate_batches(group_files, validation_type, validation_fraction, data_type, batch_size=group_batch_size) 
                for group_files, group_batch_size in zip(data_file_list, batch_sizes)]
    
    
    for multi_batch_data in zip(*generators):
        batch_x=torch.vstack([d[0] for d in multi_batch_data if len(d[0])>0])
        batch_base_seq=torch.vstack([d[1] for d in multi_batch_data if len(d[1])>0])
        batch_ref_seq=torch.vstack([d[2] for d in multi_batch_data if len(d[2])>0])
        batch_y=torch.vstack([d[3] for d in multi_batch_data if len(d[3])>0])
        
        yield batch_x, batch_base_seq, batch_ref_seq, batch_y

def get_stats(metrics_dict, dtype):
    
    loss_str='{}  Loss:'.format(dtype)
    acc_str='{} Accuracy:'.format(dtype)
    prec_str='{} Precision:'.format(dtype)
    rec_str='{} Recall:'.format(dtype)
    f1_str='{} F1:'.format(dtype)
    
    for g in sorted(metrics_dict.keys()):
        
        acc=metrics_dict[g]['true']/max(1,metrics_dict[g]['len'])
        loss=metrics_dict[g]['loss']/max(1,metrics_dict[g]['len'])
        precision=metrics_dict[g]['TP']/max(1,metrics_dict[g]['TP']+metrics_dict[g]['FP'])
        recall=metrics_dict[g]['TP']/max(1,metrics_dict[g]['TP']+metrics_dict[g]['FN'])
        f1=2*precision*recall/(precision+recall) if precision*recall!=0 else 0

        if len(metrics_dict.keys())==1:
            x='Total'
            total_acc=acc
        else:
            x=g
            
        loss_str+='  %s: %.4f' %(x, loss)
        acc_str+='  %s: %.4f' %(x, acc)
        prec_str+='  %s: %.4f' %(x, precision)
        rec_str+='  %s: %.4f' %(x, recall)
        f1_str+='  %s: %.4f' %(x, f1)
        
    if len(metrics_dict.keys())>1:
        x='Total'
        
        acc=sum(f['true'] for f in metrics_dict.values())/max(1,sum(f['len'] for f in metrics_dict.values()))
        loss=sum(f['loss'] for f in metrics_dict.values())/max(1,sum(f['len'] for f in metrics_dict.values()))
        precision=sum(f['TP'] for f in metrics_dict.values())/max(1,sum(f['TP'] for f in metrics_dict.values()) +sum(f['FP'] for f in metrics_dict.values()))
        recall=sum(f['TP'] for f in metrics_dict.values())/max(1,sum(f['TP'] for f in metrics_dict.values())+sum(f['FN'] for f in metrics_dict.values()))
        f1=2*precision*recall/(precision+recall) if precision*recall!=0 else 0
        
        total_acc=acc
        
        loss_str+='  %s: %.4f' %(x, loss)
        acc_str+='  %s: %.4f' %(x, acc)
        prec_str+='  %s: %.4f' %(x, precision)
        rec_str+='  %s: %.4f' %(x, recall)
        f1_str+='  %s: %.4f' %(x, f1)
    
    return '\n'.join([loss_str, acc_str,prec_str, rec_str, f1_str]), total_acc

def get_metrics(metrics_dict,name, batch_y, score, loss):
    eval_counts=sklearn.metrics.confusion_matrix(batch_y.cpu(),(score>0).cpu(),labels=[0,1])
    metrics_dict[name]['len']+=len(batch_y)
    metrics_dict[name]['TP']+=eval_counts[1,1]
    metrics_dict[name]['FP']+=eval_counts[0,1]
    metrics_dict[name]['FN']+=eval_counts[1,0]
    metrics_dict[name]['true']=metrics_dict[name]['true']+eval_counts[0,0]+eval_counts[1,1]
    metrics_dict[name]['loss']+=loss.item()*len(batch_y)
