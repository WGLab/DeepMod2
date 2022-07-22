from collections import defaultdict, ChainMap

import time, itertools
from tqdm import tqdm

import datetime, os, shutil, argparse, sys

import multiprocessing as mp
import numpy as np

from pathlib import Path

    
from .  import guppy
from .  import tombo

from .utils import *

def per_read_predict(params):
    
    
    temp_folder=os.path.join(params['output'],'intermediate_files')
    os.makedirs(temp_folder, exist_ok=True)
    
    
    f5files = list(Path(params['fast5']).rglob("*.fast5"))
    files_per_process=len(f5files)//params['threads'] + 1
    print('%s: Number of files: %d\n' %(str(datetime.datetime.now()), len(f5files)), flush=True)
    
    pool = mp.Pool(processes=params['threads'])
    
    read_info=None
    
    job_counter=itertools.count(start=1, step=1)
    
    
    
    if params['basecaller']=='guppy':
        print('%s: Processing BAM File.' %str(datetime.datetime.now()), flush=True)
        
        read_info=guppy.process_bam(params, pool)
        
        print('%s: Finished Processing BAM File.' %str(datetime.datetime.now()), flush=True)
        print('%s: Starting Per Read Methylation Detection.' %str(datetime.datetime.now()), flush=True)
        
        res=pool.imap_unordered(guppy.detect, zip(split_list(f5files, files_per_process), itertools.repeat(params), itertools.repeat(read_info), job_counter))
        
    else:
        print('%s: Starting Per Read Methylation Detection.' %str(datetime.datetime.now()), flush=True)
        res=pool.imap_unordered(tombo.detect, zip(split_list(f5files, files_per_process), itertools.repeat(params), itertools.repeat(read_info), job_counter))
    
    file_list=[file_name for file_name in res]
    
    output=os.path.join(params['output'], '%s.per_read' %params['file_name'])
    with open(output,'wb') as outfile:
        outfile.write(b'read_name\tchromosome\tposition\tread_position\tstrand\tmethylation_score\tmethylation_prediction\n')
        for f in file_list:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, outfile)
            #os.remove(f)
    
    #shutil.rmtree(temp_folder)
    
    pool.close()
    pool.join()
    
    print('%s: Finishing Per Read Methylation Detection.' %str(datetime.datetime.now()), flush=True)
    
    return output
    
def per_site_detect(read_pred_file_list, params):
    
    print('%s: Starting Per Site Methylation Detection.' %str(datetime.datetime.now()), flush=True)
    
    total_files=len(read_pred_file_list)
    print('%s: Reading %d files.' %(str(datetime.datetime.now()), total_files), flush=True)
    
    threshold=0.5
    
    output=os.path.join(params['output'], '%s.per_site' %params['file_name'])
    
    per_site_pred={}
    
    pbar = tqdm(total=total_files)
    
    
    for read_pred_file in read_pred_file_list:
        with open(read_pred_file,'r') as read_file:
            read_file.readline()
            for line in read_file:
                read, chrom, pos, read_pos, strand, score, meth=line.rstrip('\n').split('\t')

                if (chrom, pos, strand) not in per_site_pred:
                    per_site_pred[(chrom, pos, strand)]=[0,0,0]

                per_site_pred[(chrom, pos, strand)][int(meth)]+=1
                per_site_pred[(chrom, pos, strand)][2]+=float(score)
        
        pbar.update(1)
        
    pbar.close()
    
    print('%s: Writing Per Site Methylation Detection.' %str(datetime.datetime.now()), flush=True)
    
    with open(output,'w') as outfile:
        outfile.write('chromosome\tposition\tstrand\ttotal_coverage\tmethylation_coverage\tmethylation_percentage\tmean_methylation_probability\n')
        for x,y in per_site_pred.items():
            tot_cov=y[0]+y[1]
            p=y[2]/tot_cov
            outfile.write('%s\t%s\t%s\t%d\t%d\t%.4f\t%.4f\n' %(x[0], x[1], x[2], tot_cov, y[1], y[1]/tot_cov, p))
    
    print('%s: Finished Per Site Methylation Detection.' %str(datetime.datetime.now()), flush=True)
    
    return output