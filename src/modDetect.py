from collections import defaultdict, ChainMap

import time, itertools
from tqdm import tqdm

import datetime, os, shutil, argparse, sys, re, array

import multiprocessing as mp
import numpy as np

from pathlib import Path

    
from .  import guppy
from .  import tombo

from .utils import *

def per_read_predict(params):
    
    print('%s: Starting Per Read Methylation Detection.' %str(datetime.datetime.now()), flush=True)
 
    if params['basecaller']=='guppy':       
        guppy.call_manager(params)
        
    else:
        temp_folder=os.path.join(params['output'],'intermediate_files')
        os.makedirs(temp_folder, exist_ok=True)
            
        input_=params['input']
        f5files = input_ if os.path.isfile(input_) else list(Path(input_).rglob("*.%s" %params['file_type']))
        
        files_per_process=len(f5files)//params['threads'] + 1
        print('%s: Number of files: %d\n' %(str(datetime.datetime.now()), len(f5files)), flush=True)

        pool = mp.Pool(processes=params['threads'])


        job_counter=itertools.count(start=1, step=1)
        
        print('%s: Starting Per Read Methylation Detection.' %str(datetime.datetime.now()), flush=True)
        res=pool.imap_unordered(tombo.detect, zip(split_list(f5files, files_per_process), itertools.repeat(params), job_counter))
    
        file_list=[file_name for file_name in res]

        output=os.path.join(params['output'], '%s.per_read' %params['prefix'])
        with open(output,'wb') as outfile:
            outfile.write(b'read_name\tchromosome\tposition\tread_position\tstrand\tmethylation_score\tmean_read_qscore\tread_length\n')
            for f in file_list:
                with open(f,'rb') as fd:
                    shutil.copyfileobj(fd, outfile)
                os.remove(f)

        shutil.rmtree(temp_folder)

        pool.close()
        pool.join()
    
    print('%s: Finishing Per Read Methylation Detection.' %str(datetime.datetime.now()), flush=True)
    
    return 
    
def per_site_detect(read_pred_file_list, params):
    qscore_cutoff=params['qscore_cutoff']
    length_cutoff=params['length_cutoff']
    include_non_cpg_ref=params['include_non_cpg_ref']
    
    mod_t=params['mod_t']
    unmod_t=params['unmod_t']
    
    if params['ref']:
        ref_fasta=pysam.FastaFile(params['ref'])
        ref_pos_set_dict={rname:set(m.start(0) for m in re.finditer(r'CG', ref_fasta.fetch(rname).upper())) for rname in ref_fasta.references}    
    else:
        print('%s: No reference file provided, therefore per site output will include non CpG reference sites and assume --include_non_cpg_ref flag is set.'  %str(datetime.datetime.now()), flush=True)
        include_non_cpg_ref=True
        
    print('%s: Starting Per Site Methylation Detection.' %str(datetime.datetime.now()), flush=True)
    
    total_files=len(read_pred_file_list)
    print('%s: Reading %d files.' %(str(datetime.datetime.now()), total_files), flush=True)
    
    threshold=0.5
    
    output=os.path.join(params['output'], '%s.per_site' %params['prefix'])
    
    per_site_pred={}
    
    pbar = tqdm(total=total_files)
    
    
    for read_pred_file in read_pred_file_list:
        with open(read_pred_file,'r') as read_file:
            read_file.readline()
            for line in read_file:
                read, chrom, pos, read_pos, strand, score, mean_qscore, sequence_length = line.rstrip('\n').split('\t')
                score=float(score)

                if float(mean_qscore)<qscore_cutoff or int(sequence_length)<length_cutoff or pos=='NA':
                    continue
                
                #read, chrom, pos, read_pos, strand, score, meth = line.rstrip('\n').split('\t')
                
                if (chrom, pos, strand) not in per_site_pred:
                    # unmod, mod, score
                    per_site_pred[(chrom, pos, strand)]=[0,0,0]
                
                if score>=mod_t:
                    per_site_pred[(chrom, pos, strand)][1]+=1
                    per_site_pred[(chrom, pos, strand)][2]+=float(score)

                elif score<unmod_t:
                    per_site_pred[(chrom, pos, strand)][0]+=1
                    per_site_pred[(chrom, pos, strand)][2]+=score
        
        pbar.update(1)
        
    pbar.close()
    
    print('%s: Writing Per Site Methylation Detection.' %str(datetime.datetime.now()), flush=True)
    
    with open(output,'w') as outfile:
        outfile.write('#chromosome\tposition_before\tposition\tmethylation_percentage\tstrand\ttotal_coverage\tmethylation_percentage\tmean_methylation_probability\n')
        for x,y in per_site_pred.items():
            tot_cov=y[0]+y[1]
            chrom, pos, strand= x[0], int(x[1]), x[2]
            if tot_cov>0:
                if include_non_cpg_ref:
                    pass
                else:
                    if (strand=='+' and pos-1 not in ref_pos_set_dict[chrom]) or (strand=='-' and pos-2 not in ref_pos_set_dict[chrom]):
                        continue
                p=y[2]/tot_cov
                outfile.write('%s\t%d\t%d\t%.4f\t%s\t%d\t%d\t%.4f\n' %(chrom, pos-1, pos, y[1]/tot_cov, strand, tot_cov, y[1], p))
    
    print('%s: Finished Per Site Methylation Detection.' %str(datetime.datetime.now()), flush=True)
    
    return output
