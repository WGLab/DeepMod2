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

def get_per_site(params, data, data_from_text=True):
    qscore_cutoff=params['qscore_cutoff']
    length_cutoff=params['length_cutoff']
    
    mod_threshold=params['mod_t']
    unmod_threshold=params['unmod_t']
    
    cpg_ref_only=not params['include_non_cpg_ref']
    
    print('%s: Starting Per Site Methylation Detection.' %str(datetime.datetime.now()), flush=True)
    
    if data_from_text:    
        total_files=len(data)
        print('%s: Reading %d files.' %(str(datetime.datetime.now()), total_files), flush=True)
        pbar = tqdm(total=total_files)
        
        per_site_pred={}
        
        for read_pred_file in data:
            with open(read_pred_file,'r') as read_file:
                read_file.readline()
                for line in read_file:
                    read, chrom, pos, pos_after, read_pos, strand, score, mean_qscore, sequence_length, phase, is_ref_cpg = line.rstrip('\n').split('\t')

                    if pos=='NA' or float(mean_qscore)<qscore_cutoff or int(sequence_length)<length_cutoff:
                        continue

                    score=float(score)

                    if score<mod_threshold and score>unmod_threshold:
                        continue
                    else:
                        mod=score>=mod_threshold 

                    pos=int(pos)
                    phase=int(phase)
                    is_ref_cpg=True if is_ref_cpg =='True' else False
                    zero_based_fwd_pos=pos if strand=='+' else pos-1



                    if (chrom, zero_based_fwd_pos) not in per_site_pred:
                        per_site_pred[(chrom, zero_based_fwd_pos)]=CpG(chrom, zero_based_fwd_pos, is_ref_cpg)

                    per_site_pred[(chrom, zero_based_fwd_pos)].append((mod, strand, phase))

            pbar.update(1)
        pbar.close()
        
    else:
        per_site_pred=data
        
    print('%s: Writing Per Site Methylation Detection.' %str(datetime.datetime.now()), flush=True)    
    
    per_site_fields=['#chromosome', 'position_before', 'position','strand', 'ref_cpg',
                 'coverage','mod_coverage', 'unmod_coverage','mod_percentage',
                 'coverage_phase1','mod_coverage_phase1', 'unmod_coverage_phase1','mod_percentage_phase1',
                 'coverage_phase2','mod_coverage_phase2', 'unmod_coverage_phase2','mod_percentage_phase2']
    per_site_header='\t'.join(per_site_fields)
    per_site_fields.remove('strand')
    agg_per_site_header='\t'.join(per_site_fields)
    
    per_site_file_path=os.path.join(params['output'],'%s.per_site' %params['prefix'])
    agg_per_site_file_path=os.path.join(params['output'],'%s.per_site.aggregated' %params['prefix'])
        
    with open(per_site_file_path, 'w') as per_site_file, open(agg_per_site_file_path,'w') as agg_per_site_file:
        per_site_file.write(per_site_header)
        agg_per_site_file.write(agg_per_site_header)

        for x in sorted(per_site_pred.keys()):
            chrom, pos=x
            cpg=per_site_pred[x]
            
            if cpg_ref_only and cpg.is_ref_cpg==False:
                continue
            
            
            
            total_phases=cpg.get_all_phases()
            phase_1=cpg.phase_1
            phase_2=cpg.phase_2

            fwd_stats, rev_stats=cpg.get_stats_string(aggregate=False)
            agg_stats=cpg.get_stats_string()

            if agg_stats[0]>0:
                agg_per_site_file.write('\n'+agg_stats[1])

            if fwd_stats[0]>0:
                per_site_file.write('\n'+fwd_stats[1])

            if rev_stats[0]>0:
                per_site_file.write('\n'+rev_stats[1])
    
    print('%s: Finished Writing Per Site Methylation Output.' %str(datetime.datetime.now()), flush=True)
    print('%s: Per Site Prediction file: %s' %(str(datetime.datetime.now()), per_site_file_path), flush=True)
    print('%s: Aggregated Per Site Prediction file: %s' %(str(datetime.datetime.now()), agg_per_site_file_path), flush=True)