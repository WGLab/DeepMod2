from subprocess import PIPE, Popen
import os, shutil

model_dict={'guppy_na12878':{'path':'models/guppy/test_model', 
                             'help':'Model trained on NA12878 Guppy v5 basecalled FAST5 files using positive and negative 5mC methylation control samples from Simpson (Nat Methods 2017).'}, 
            'tombo_na12878':{'path':'models/tombo/test_model',
                             'help': 'Model trained on NA12878 Tombo resquiggled FAST5 files (after Guppy v5 basecalling) using positive and negative 5mC methylation control samples from Simpson (Nat Methods 2017).'},
            'guppy_na12878_native': {'path':'models/guppy/rel3_chr1/model.14-0.25.h5', 'help': 'Model trained on chr1 of NA12878 Guppy v5 basecalled FAST5 files from Nanopore WGS Consortium. Bisulphite methylation calls from two replicates (ENCFF279HCL, ENCFF835NTC) from ECNODE project were used as ground truth for training.'}
           }


def get_model_help():
    for model in model_dict:
        print('-'*30)
        print('Model Name: %s' %model)
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