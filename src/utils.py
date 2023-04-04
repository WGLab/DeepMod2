from subprocess import PIPE, Popen
import os, shutil, pysam
import numpy as np

model_dict={'guppy_hg1_R9.4':{'path':'models/guppy/guppy_r9.4/guppy_hg1_r9.4.h5', 
                             'help':'Model trained on chr1 of R9.4.1 NA12878 Guppy v5 basecalled FAST5 files from Nanopore WGS Consortium. Bisulphite methylation calls from two replicates (ENCFF279HCL, ENCFF835NTC) from ECNODE project were used as ground truth for training.'}, 
            'guppy_R9.4.1':{'path':'models/guppy/guppy_r9.4/guppy_hg2_r9.4.1.h5',
                                'help':'Model trained on chr1 of HG002 R9.4.1 Guppy v6 basecalled FAST5 files and bisulfite methylation calls from Oxford Nanoporetech release.'},
            'tombo_R9.4.1':{'path':'models/tombo/tombo_r9.4.1/model.30-0.9407.h5',
                             'help': 'Model trained on Tombo resquiggled R9.4.1 FAST5 files using positive and negative 5mC methylation E. coli and NA12878 control samples from Simpson (Nat Methods 2017), as well as Tombo resquiggled FAST5 files from chr1 of NA12878 from Nanopore WGS Consortium. Bisulphite methylation calls from two replicates (ENCFF279HCL, ENCFF835NTC) from ECNODE project were used as ground truth for training.'},
            'guppy_R10.4.1': {'path':'models/guppy/guppy_r10.4.1/guppy_hg2_r10.4.1.h5',
                              'help': 'Model trained on chr1 of HG002 R10.4.1 Guppy v6 basecalled FAST5 files and bisulfite methylation calls from Oxford Nanoporetech Q20+ data release.'}
           }

base_map={'A':'T','T':'A','C':'G','G':'C'}

def revcomp(s):
    return ''.join(base_map[x] for x in s[::-1])

def get_model_help():
    for n,model in enumerate(model_dict):
        print('-'*30)
        print('%d) Model Name: %s' %(n+1, model))
        print('Details: %s\n' %model_dict[model]['help'])
        
def get_model(model):
    if model in model_dict:
        dirname = os.path.dirname(__file__)
        return os.path.join(dirname, model_dict[model]['path'])
        
    elif os.path.exists(model):
        return model
     
    else:
        return None
    
    
def run_cmd(cmd, verbose=False, output=False,error=False):
    stream=Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = stream.communicate()
    
    stdout=stdout.decode('utf-8')
    stderr=stderr.decode('utf-8')
    
    if stderr:
        print(stderr, flush=True)
    
    if verbose:
        print(stdout, flush=True)
        
        
    if output:
        return stdout
    if error:
        return stderr
    
def split_list(l,n=1000):
    i=0    
    chunk = l[i*n:(i+1)*n]
    while chunk:
        yield chunk
        i+=1
        chunk = l[i*n:(i+1)*n]
        
def get_attr(f,suffix):
    keys = []
    f.visit(lambda key : keys.append(f[key].attrs[suffix]) if suffix in f[key].attrs else None)
    
    return keys[0]        

def get_per_read_stats(per_read):
    per_read_stats={}
    
    with open(per_read,'r') as file:
        file.readline()
        for line in file:
            line=line.rstrip('\n').split()
            rname, read_position, score= line[0], int(line[3]), float(line[5])
            if rname not in per_read_stats:
                per_read_stats[rname]=[[],[],[]]
            per_read_stats[rname][0].append(read_position-1)
            per_read_stats[rname][1].append(max(0,np.round(256*score-1).astype(int)))
            per_read_stats[rname][2].append(int(line[2]))
    return per_read_stats
