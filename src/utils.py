from subprocess import PIPE, Popen
import os, shutil

model_dict={'guppy_R9.4.1':{'path':'models/guppy/guppy_r9.4.1/model.40-0.9370.h5', 
                             'help':'Model trained on chr1 of R9.4.1 NA12878 Guppy v5 basecalled FAST5 files from Nanopore WGS Consortium. Bisulphite methylation calls from two replicates (ENCFF279HCL, ENCFF835NTC) from ECNODE project were used as ground truth for training.'}, 
            'tombo_R9.4.1':{'path':'models/tombo/tombo_r9.4.1/model.30-0.9407.h5',
                             'help': 'Model trained on Tombo resquiggled R9.4.1 FAST5 files using positive and negative 5mC methylation E. coli and NA12878 control samples from Simpson (Nat Methods 2017), as well as Tombo resquiggled FAST5 files from chr1 of NA12878 from Nanopore WGS Consortium. Bisulphite methylation calls from two replicates (ENCFF279HCL, ENCFF835NTC) from ECNODE project were used as ground truth for training'},
            'guppy_R10.4': {'path':'models/guppy/guppy_r10.4/model.58-0.9800.h5', 'help': 'Model trained on chr1 of HG002 R10.4 Guppy v6 basecalled FAST5 files and bisulfite methylation calls from Oxford Nanoporetech Q20+ data release.'}
           }



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