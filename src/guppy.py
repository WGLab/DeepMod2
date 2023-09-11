from collections import defaultdict, ChainMap

import time, itertools, h5py, pysam

import datetime, os, shutil, argparse, sys, re, array

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import multiprocessing as mp
import numpy as np

from pathlib import Path

from .utils import *
from ont_fast5_api.fast5_interface import get_fast5_file

from numba import jit

import queue
import pod5 as p5
        
@jit(nopython=True)
def get_events(signal, move):
    stride, start, move_table=move
    median=np.median(signal)
    mad=np.median(np.abs(signal-median))
    
    signal=(signal-median)/mad
    
    move_len=len(move_table)
    move_index=np.where(move_table)[0]
    rlen=len(move_index)
    
    data=np.zeros((rlen,9))
    
    for i in range(len(move_index)-1):
        prev=move_index[i]*stride+start
        sig_end=move_index[i+1]*stride+start
        
        sig_len=sig_end-prev
        data[i, 8]=sig_len
        data[i, 4]=np.median(signal[prev:sig_end])
        data[i, 5]=np.median(np.abs(signal[prev:sig_end]-data[i, 4]))
        data[i, 6]=np.mean(signal[prev:sig_end])
        data[i, 7]=np.std(signal[prev:sig_end])
        
        for j in range(4):
            tmp_cnt=0
            for t in range(j*sig_len//4,min(sig_len, (j+1)*sig_len//4)):
                data[i, j]+=signal[t+prev]
                tmp_cnt+=1
            data[i, j]=data[i, j]/tmp_cnt

    return data

def get_candidates(read_seq, align_data, ref_pos_dict):    
    if align_data[0]:
        is_mapped, is_forward, ref_name, reference_start, reference_end, read_length, aligned_pairs=align_data
        aligned_pairs=np.array(aligned_pairs)
        aligned_pairs[aligned_pairs==None]=-1
        aligned_pairs=aligned_pairs.astype(int)
        
        c_id={m.start(0):i for i,m in enumerate(re.finditer(r'C', read_seq))}
        cg_id=np.array([m.start(0) for m in re.finditer(r'CG', read_seq)])
        ref_motif_pos=ref_pos_dict[ref_name] if is_forward else ref_pos_dict[ref_name] +1

        common_pos=ref_motif_pos[(ref_motif_pos>=reference_start)&(ref_motif_pos<reference_end)]
        aligned_pairs_ref_wise=aligned_pairs[aligned_pairs[:,1]!=-1][common_pos-reference_start]

        aligned_pairs_ref_wise=aligned_pairs_ref_wise[aligned_pairs_ref_wise[:,0]!=-1]
        aligned_pairs_read_wise=aligned_pairs[aligned_pairs[:,0]!=-1]

        if not is_forward:
            aligned_pairs_ref_wise=aligned_pairs_ref_wise[::-1]
            aligned_pairs_ref_wise[:,0]=read_length-aligned_pairs_ref_wise[:,0]-1
            aligned_pairs_read_wise=aligned_pairs_read_wise[::-1]
            aligned_pairs_read_wise[:,0]=read_length-aligned_pairs_read_wise[:,0]-1
            
        if len(cg_id)>0:
            aligned_pairs_read_wise=aligned_pairs_read_wise[cg_id]
            
            #if need to disable clipped bases
            #aligned_pairs_read_wise=aligned_pairs_read_wise[(reference_start<=aligned_pairs_read_wise[:,1]) & (aligned_pairs_read_wise[:,1]<reference_end)]

            merged=np.vstack((aligned_pairs_ref_wise, aligned_pairs_read_wise))
            _,ind=np.unique(merged[:,0], return_index=True)
            merged=merged[ind]
            return c_id, merged
        
        else:
            return c_id, aligned_pairs_ref_wise
    
    else:
        c_id={m.start(0):i for i,m in enumerate(re.finditer(r'C', read_seq))}
        cg_id=np.array([[m.start(0),-1] for m in re.finditer(r'CG', read_seq)])
        return (c_id, cg_id)

def per_site_info(data):
    # unmod, mod, score
    try:
        cov=data[0]+data[1]
        mod_p=data[1]/cov
        mean_score=score/cov
        
        return cov, mod_p, mean_score
    
    except:
        return 0, 0, 0
    
def get_output(params, output_Q, methylation_event, header_dict, ref_pos_dict):
    header=pysam.AlignmentHeader.from_dict(header_dict)

    output=params['output']
    bam_output=os.path.join(output,'%s.bam' %params['prefix'])
    per_read_file_path=os.path.join(output,'%s.per_read' %params['prefix'])

    
    per_site_file_path=os.path.join(output,'%s.per_site' %params['prefix'])
    agg_per_site_file_path=os.path.join(output,'%s.per_site.aggregated' %params['prefix'])
    qscore_cutoff=params['qscore_cutoff']
    length_cutoff=params['length_cutoff']
    
    mod_threshold=params['mod_t']
    unmod_threshold=params['unmod_t']
    
    skip_per_site=params['skip_per_site']
    
    per_site_pred={}
    
    counter=0
    
    cpg_ref_only=not params['include_non_cpg_ref']
    
    ref_pos_set_dict={rname:set(motif_list) for rname, motif_list in ref_pos_dict.items()} if cpg_ref_only else None
       
        
    with open(per_read_file_path,'w') as per_read_file:
        per_read_file.write('read_name\tchromosome\tref_position_before\tref_position\tread_position\tstrand\tmethylation_score\tmean_read_qscore\tread_length\tread_phase\tref_cpg\n')
        
        
        with pysam.AlignmentFile(bam_output, "wb", header=header) as outf:
            while True:
                if methylation_event.is_set() and output_Q.empty():
                    break
                else:
                    try:
                        res = output_Q.get(block=False)
                        
                        if counter%10000==0:
                            print('%s: Number of reads processed: %d' %(str(datetime.datetime.now()), counter), flush=True)
                        if res[0]:
                            _, total_read_info, total_candidate_list, total_MM_list, read_qual_list, pred_list = res
                            for read_data, candidate_list, MM, ML, pred_list in zip(*res[1:]):
                                counter+=1
                                read_dict, read_info = read_data
                                read=pysam.AlignedSegment.from_dict(read_dict,header)
                                if MM:
                                    read.set_tag('MM',MM,value_type='Z')
                                    read.set_tag('ML',ML)
                                    
                                outf.write(read)
                                
                                read_name=read_dict['name']
                                is_forward, chrom, read_length, mean_qscore=read_info
                                chrom=chrom if chrom else 'NA'
                                strand='+' if is_forward else '-'
                                
                                phase=read.get_tag('HP') if read.has_tag('HP') else 0
                                
                                if float(mean_qscore)<qscore_cutoff or int(read_length)<length_cutoff:
                                    continue
                                    
                                for i in range(len(pred_list)):
                                    read_pos=candidate_list[i][0]+1
                                    ref_pos=candidate_list[i][1]
                                    score=pred_list[i]
                                    
                                    ref_pos_str_before=str(ref_pos) if ref_pos!=-1 else 'NA'
                                    ref_pos_str_after=str(ref_pos+1) if ref_pos!=-1 else 'NA'
                                    
                                    zero_based_fwd_pos=ref_pos if is_forward else ref_pos-1
                                    
                                    is_ref_cpg=False
                                    
                                    if ref_pos_str_before=='NA':    
                                        pass
                                    
                                    else:
                                        if cpg_ref_only:
                                            is_ref_cpg = (strand=='+' and zero_based_fwd_pos in ref_pos_set_dict[chrom]) or (strand=='-' and zero_based_fwd_pos in ref_pos_set_dict[chrom])
                                        if score<mod_threshold and score>unmod_threshold:
                                            pass
                                        elif not skip_per_site:
                                            mod=score>=mod_threshold
                                            
                                            if (chrom, zero_based_fwd_pos) not in per_site_pred:
                                                per_site_pred[(chrom, zero_based_fwd_pos)]=CpG(chrom, zero_based_fwd_pos, is_ref_cpg)

                                            per_site_pred[(chrom, zero_based_fwd_pos)].append((mod, strand, phase))

                                    per_read_file.write('%s\t%s\t%s\t%s\t%d\t%s\t%.4f\t%.2f\t%d\t%d\t%s\n' %(read_name, chrom, ref_pos_str_before, ref_pos_str_after, read_pos, strand, score, mean_qscore, read_length, phase, is_ref_cpg))
                        
                                
                        else:
                            _, total_read_info=res
                            for read_dict in total_read_info:
                                counter+=1
                                read=pysam.AlignedSegment.from_dict(read_dict,header)
                                outf.write(read)
                                
                    except queue.Empty:
                        pass    
    
    print('%s: Number of reads processed: %d' %(str(datetime.datetime.now()), counter), flush=True)
    print('%s: Finished Per-Read Methylation Output. Starting Per-Site output.' %str(datetime.datetime.now()), flush=True)        
    print('%s: Modification Tagged BAM file: %s' %(str(datetime.datetime.now()),bam_output), flush=True)
    print('%s: Per Read Prediction file: %s' %(str(datetime.datetime.now()), per_read_file_path), flush=True)
    print('%s: Writing Per Site Methylation Detection.' %str(datetime.datetime.now()), flush=True)    
    
    if skip_per_site:
        return 
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
    
    return

def process(params,ref_pos_dict, signal_Q, output_Q, input_event):
    from tensorflow import keras
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    base_map={'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
    window=10
    model=keras.models.load_model(get_model(params['model']))
    xla_fn = model if gpus else tf.function(model, jit_compile=True)
    chunk_size=2048 #params['chunk_size']
    
    total_candidate_list=[]
    total_feature_list=[]
    total_MM_list=[]
    total_read_info=[]
    total_c_idx=[]
    total_unprocessed_reads=[]
    r_count=0
    while True:
        if (signal_Q.empty() and input_event.is_set()):
            break
        
        try:
            data=signal_Q.get(block=False)
            
            signal, move, read_dict, align_data=data
            
            fq=read_dict['seq']
            qual=read_dict['qual']
            sequence_length=len(fq)
            reverse=16 & int(read_dict['flag'])
            fq=revcomp(fq) if reverse else fq
            qual=qual[::-1] if reverse else qual

            pos_list_c, pos_list_candidates=get_candidates(fq, align_data, ref_pos_dict)

            pos_list_candidates=pos_list_candidates[(pos_list_candidates[:,0]>window)\
                                                    &(pos_list_candidates[:,0]<sequence_length-window-1)] if len(pos_list_candidates)>0 else pos_list_candidates

            if len(pos_list_candidates)==0:
                total_unprocessed_reads.append(read_dict)
                continue
                
            if not move[0]:
                try:
                    tags={x.split(':')[0]:x for x in read_dict['tags']}
                    start=int(tags['ts'].split(':')[-1])
                    mv=tags['mv'].split(',')

                    stride=int(mv[1])
                    move_table=np.array([int(x) for x in mv[2:]])
                    move=(stride, start, move_table)
                except KeyError:
                    print('Read:%s no move table or stride or signal start found' %read_name)
                    total_unprocessed_reads.append(read_dict)
                    continue
                    
            base_seq=[base_map[x] for x in fq]
            base_seq=np.eye(4)[base_seq]
            base_qual=10.0**(-np.array([ord(q)-33 for q in qual])/10)[:,np.newaxis]
            mean_qscore=-10*np.log10(np.mean(base_qual))
            mat=get_events(signal, move)
            mat=np.array(mat)
            mat=np.hstack((mat, base_qual, base_seq))
            
            try:
                c_idx=[True if x in pos_list_c else False for x in pos_list_candidates[:,0]]
                c_idx_count=np.vectorize(pos_list_c.get)(pos_list_candidates[c_idx,0])
                c_idx_count[1:]=c_idx_count[1:]-c_idx_count[:-1]-1
                MM='C+m?,'+','.join(c_idx_count.astype(str))+';'
                total_c_idx.append(c_idx)
                total_MM_list.append(MM)
            
            except ValueError:
                total_c_idx.append([])
                total_MM_list.append(None)
            
            features=np.array([mat[candidate[0]-window: candidate[0]+window+1] for candidate in pos_list_candidates])

            total_candidate_list.append(pos_list_candidates)
            total_feature_list.append(features)
            
            total_read_info.append((read_dict, [align_data[1],align_data[2],align_data[5], mean_qscore]))
            
            
            if len(total_read_info)>100:
                read_counts=np.cumsum([len(x) for x in total_feature_list])[:-1]
                features_list=np.vstack(total_feature_list)
                features_list[:,:,8]=features_list[:,:,8]/np.sum(features_list[:,:,8],axis=1)[:, np.newaxis]

                pred_list=[xla_fn(chunk).numpy() for chunk in split_array(features_list,chunk_size)]
                if features_list.shape[0]%chunk_size:
                    pred_list.append(model(features_list[-1*(features_list.shape[0]%chunk_size):]).numpy())

                pred_list=np.vstack(pred_list)
                pred_list=np.split(pred_list.ravel(), read_counts)
                read_qual_list=[array.array('B',np.round(255*read_pred_list[c_idx]).astype(int)) for read_pred_list, c_idx in zip(pred_list, total_c_idx)]

                output_Q.put([True, total_read_info, total_candidate_list, total_MM_list, read_qual_list, pred_list])
                total_candidate_list, total_feature_list, total_MM_list, total_read_info, total_c_idx=[], [], [], [], []
                
            if len(total_unprocessed_reads)>100:
                output_Q.put([False, total_unprocessed_reads])
                total_unprocessed_reads=[]

        except queue.Empty:
            pass            

    if len(total_read_info)>0:
        read_counts=np.cumsum([len(x) for x in total_feature_list])[:-1]
        features_list=np.vstack(total_feature_list)
        features_list[:,:,8]=features_list[:,:,8]/np.sum(features_list[:,:,8],axis=1)[:, np.newaxis]

        pred_list=[xla_fn(chunk).numpy() for chunk in split_array(features_list,chunk_size)]
        if features_list.shape[0]%chunk_size:
            pred_list.append(model(features_list[-1*(features_list.shape[0]%chunk_size):]).numpy())

        pred_list=np.vstack(pred_list)

        pred_list=np.split(pred_list.ravel(), read_counts)

        read_qual_list=[array.array('B',np.round(255*read_pred_list[c_idx]).astype(int)) for read_pred_list, c_idx in zip(pred_list, total_c_idx)]

        output_Q.put([True, total_read_info, total_candidate_list, total_MM_list, read_qual_list, pred_list])

    if len(total_unprocessed_reads)>0:
        output_Q.put([False, total_unprocessed_reads])

    return

def get_input(params, signal_Q, output_Q, input_event):    
    bam=params['bam']
    bam_file=pysam.AlignmentFile(bam,'rb',check_sq=False)
    
    print('%s: Building BAM index.' %str(datetime.datetime.now()), flush=True)
    bam_index=pysam.IndexedReads(bam_file)
    bam_index.build()
    print('%s: Finished building BAM index.' %str(datetime.datetime.now()), flush=True)
    
    input_=params['input']
    signal_files= [input_] if os.path.isfile(input_) else Path(input_).rglob("*.%s" %params['file_type'])

    if params['file_type']=='fast5':
        guppy_group=params['guppy_group']
        for filename in signal_files:
            with get_fast5_file(filename, mode="r") as f5:
                for read in f5.get_reads():
                    if signal_Q.qsize()>10000:
                        time.sleep(10)
                    read_name=read.read_id
                    signal=read.get_raw_data()
                    
                    if params['fast5_move']:
                        segment=read.get_analysis_attributes(guppy_group)['segmentation']
                        start=read.get_analysis_attributes('%s/Summary/segmentation' %segment)['first_sample_template']
                        stride=read.get_summary_data(guppy_group)['basecall_1d_template']['block_stride']
                        move_table=read.get_analysis_dataset('%s/BaseCalled_template' %guppy_group, 'Move')
                        move=(stride, start, move_table)
                    else:
                        move=(None,None,None)
                        
                    try:
                        read_iter=bam_index.find(read_name)
                        non_primary_reads=[]
                        for read in read_iter:
                            read_dict=read.to_dict()
                            if not (read.is_supplementary or read.is_secondary):
                                read_dict=read.to_dict()
                                align_data=(read.is_mapped if params['ref'] else False, read.is_forward, read.reference_name, read.reference_start, read.reference_end, read.query_length, read.aligned_pairs)
                                data=(signal, move, read_dict, align_data)
                                signal_Q.put(data)
                            
                            else:
                                non_primary_reads.append(read_dict)
                        
                        if len(non_primary_reads)>0:
                            output_Q.put([False, non_primary_reads])
                        
                    except KeyError:
                        print('Read:%s not found in BAM file' %read_name)
                        continue
                    
    else:
        move=(None,None,None)
        for filename in signal_files:
            with p5.Reader(filename) as reader:
                for read in reader.reads():
                    if signal_Q.qsize()>10000:
                        time.sleep(10)
                        
                    read_name=str(read.read_id)
                    signal=read.signal
                    try:
                        read_iter=bam_index.find(read_name)
                        non_primary_reads=[]
                        for read in read_iter:
                            read_dict=read.to_dict()
                            if not (read.is_supplementary or read.is_secondary):
                                read_dict=read.to_dict()
                                align_data=(read.is_mapped if params['ref'] else False, read.is_forward, read.reference_name, read.reference_start, read.reference_end, read.query_length, read.aligned_pairs)
                                data=(signal, move, read_dict, align_data)
                                signal_Q.put(data)
                            
                            else:
                                non_primary_reads.append(read_dict)
                        
                        if len(non_primary_reads)>0:
                            output_Q.put([False, non_primary_reads])
                            
                    except KeyError:
                        print('Read:%s not found in BAM file' %read_name)
                        continue
    input_event.set()
    return
    
def call_manager(params):        
    bam=params['bam']
    bam_file=pysam.AlignmentFile(bam,'rb',check_sq=False)
    header_dict=bam_file.header.to_dict()
    
    print('%s: Getting motif positions from the reference.' %str(datetime.datetime.now()), flush=True)
    if params['ref']:
        ref_fasta=pysam.FastaFile(params['ref'])
        ref_pos_dict={rname:np.array([m.start(0) for m in re.finditer(r'CG', ref_fasta.fetch(rname).upper())]) for rname in ref_fasta.references}    
    else:
        ref_pos_dict={}
    print('%s: Finished getting motif positions from the reference.' %str(datetime.datetime.now()), flush=True)
    
    pmanager = mp.Manager()
    signal_Q = pmanager.Queue()
    output_Q = pmanager.Queue()
    methylation_event=pmanager.Event()
    input_event=pmanager.Event()
    
    handlers = []
    
    input_process = mp.Process(target=get_input, args=(params, signal_Q, output_Q, input_event))
    input_process.start()
    handlers.append(input_process)
    
    for hid in range(max(1,params['threads']-1)):
        p = mp.Process(target=process, args=(params, ref_pos_dict, signal_Q, output_Q, input_event));
        p.start();
        handlers.append(p);
    
    output_process=mp.Process(target=get_output, args=(params, output_Q, methylation_event, header_dict, ref_pos_dict));
    output_process.start();
    
    for job in handlers:
        job.join()
    
    methylation_event.set()
    output_process.join()
    
    return
