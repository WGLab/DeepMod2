from collections import defaultdict, ChainMap

import time, itertools, h5py, pysam

import datetime, os, shutil, argparse, sys, re, array

import os

import multiprocessing as mp
import numpy as np

from pathlib import Path

from .utils import *
from ont_fast5_api.fast5_interface import get_fast5_file

from numba import jit

import queue
import pod5 as p5

import torch

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
        data[i, 8]=np.log10(sig_len)
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

def get_candidates(read_seq, align_data, aligned_pairs, ref_pos_dict, exp_motif_seq, motif_base, motif_ind, position_based):    
    if align_data[0]:
        is_mapped, is_forward, ref_name, reference_start, reference_end, read_length=align_data
        
        base_id={m.start(0):i for i,m in enumerate(re.finditer(r'(?={})'.format(motif_base), read_seq))}
        
        motif_anchor=np.array([m.start(0) for m in re.finditer(r'(?={})'.format(exp_motif_seq), read_seq)])
        motif_id=np.array(sorted(list(set.union(*[set(motif_anchor+i) for i in motif_ind]))))
        
        ref_motif_pos=ref_pos_dict[ref_name][0] if is_forward else ref_pos_dict[ref_name][1]

        common_pos=ref_motif_pos[(ref_motif_pos>=reference_start)&(ref_motif_pos<reference_end)]
        aligned_pairs_ref_wise=aligned_pairs[aligned_pairs[:,1]!=-1][common_pos-reference_start]

        aligned_pairs_ref_wise=aligned_pairs_ref_wise[aligned_pairs_ref_wise[:,0]!=-1]
        aligned_pairs_read_wise_original=aligned_pairs[aligned_pairs[:,0]!=-1]
        aligned_pairs_read_wise=np.copy(aligned_pairs_read_wise_original)
        if not is_forward:
            aligned_pairs_ref_wise=aligned_pairs_ref_wise[::-1]
            aligned_pairs_ref_wise[:,0]=read_length-aligned_pairs_ref_wise[:,0]-1
            aligned_pairs_read_wise=aligned_pairs_read_wise[::-1]
            aligned_pairs_read_wise[:,0]=read_length-aligned_pairs_read_wise[:,0]-1
            
        if len(motif_id)>0 and not position_based:
            aligned_pairs_read_wise=aligned_pairs_read_wise[motif_id]
            
            #if need to disable clipped bases
            #aligned_pairs_read_wise=aligned_pairs_read_wise[(reference_start<=aligned_pairs_read_wise[:,1]) & (aligned_pairs_read_wise[:,1]<reference_end)]

            merged=np.vstack((aligned_pairs_ref_wise, aligned_pairs_read_wise))
            _,ind=np.unique(merged[:,0], return_index=True)
            merged=merged[ind]
            return base_id, merged, aligned_pairs_read_wise_original
        
        else:
            return base_id, aligned_pairs_ref_wise, aligned_pairs_read_wise_original
    
    else:
        base_id={m.start(0):i for i,m in enumerate(re.finditer(r'(?={})'.format(motif_base), read_seq))}
        
        motif_anchor=np.array([m.start(0) for m in re.finditer(r'(?={})'.format(exp_motif_seq), read_seq)])
        motif_id=np.array(sorted(list(set.union(*[set(motif_anchor+i) for i in motif_ind]))))
        motif_id=np.vstack([motif_id,-1*np.ones(len(motif_id))]).T.astype(int)
        
        return (base_id, motif_id, None)

def per_site_info(data):
    # unmod, mod, score
    try:
        cov=data[0]+data[1]
        mod_p=data[1]/cov
        mean_score=score/cov
        
        return cov, mod_p, mean_score
    
    except:
        return 0, 0, 0
    
def get_cpg_output(params, output_Q, methylation_event, header_dict, ref_pos_dict):
    header=pysam.AlignmentHeader.from_dict(header_dict)
    
    bam_threads=params['bam_threads']

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
    
    ref_pos_set_dict={rname:set(motif_list[0]) for rname, motif_list in ref_pos_dict.items() if rname!='lock' } if cpg_ref_only else None
       
    counter_check=0
    with open(per_read_file_path,'w') as per_read_file:
        per_read_file.write('read_name\tchromosome\tref_position_before\tref_position\tread_position\tstrand\tmethylation_score\tmean_read_qscore\tread_length\tread_phase\tref_motif\n')
        
        with pysam.AlignmentFile(bam_output, "wb", threads=bam_threads, header=header) as outf:
            while True:
                if methylation_event.is_set() and output_Q.empty():
                    break
                else:
                    try:
                        res = output_Q.get(block=False, timeout=10)
                        #continue
                        if counter//10000>counter_check:
                            counter_check=counter//10000
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
                                
                                phase=0
                                phase=read.get_tag('HP') if read.has_tag('HP') else 0
                                
                                idx=4*phase+2*is_forward
                                
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
                                                per_site_pred[(chrom, zero_based_fwd_pos)]=[0]*12+[is_ref_cpg]
                                            
                                            per_site_pred[(chrom, zero_based_fwd_pos)][idx+mod]+=1

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
                 'coverage','mod_coverage', 'unmod_coverage','mod_fraction',
                 'coverage_phase1','mod_coverage_phase1', 'unmod_coverage_phase1','mod_fraction_phase1',
                 'coverage_phase2','mod_coverage_phase2', 'unmod_coverage_phase2','mod_fraction_phase2']
    per_site_header='\t'.join(per_site_fields)+'\n'
    per_site_fields.remove('strand')
    agg_per_site_header='\t'.join(per_site_fields)+'\n'
    
    per_site_file_path=os.path.join(params['output'],'%s.per_site' %params['prefix'])
    agg_per_site_file_path=os.path.join(params['output'],'%s.per_site.aggregated' %params['prefix'])
        
    with open(per_site_file_path, 'w') as per_site_file, open(agg_per_site_file_path,'w') as agg_per_site_file:
        per_site_file.write(per_site_header)
        agg_per_site_file.write(agg_per_site_header)

        for x in sorted(per_site_pred.keys()):
            chrom, pos=x
            cpg=per_site_pred[x]
            is_ref_cpg=cpg[12]
            
            if cpg_ref_only and is_ref_cpg==False:
                continue
            #fwd_stats=[self.chrom, self.position, self.position+1, '+', self.is_ref_cpg]+self.get_all_phases().forward.stats() + self.phase_1.forward.stats() + self.phase_2.forward.stats()
            
            agg_stats, fwd_stats, rev_stats=get_stats_string_cpg(chrom, pos, is_ref_cpg, cpg)
            if agg_stats[0]>0:
                agg_per_site_file.write(agg_stats[1])

            if fwd_stats[0]>0:
                per_site_file.write(fwd_stats[1])

            if rev_stats[0]>0:
                per_site_file.write(rev_stats[1])
    
    print('%s: Finished Writing Per Site Methylation Output.' %str(datetime.datetime.now()), flush=True)
    print('%s: Per Site Prediction file: %s' %(str(datetime.datetime.now()), per_site_file_path), flush=True)
    print('%s: Aggregated Per Site Prediction file: %s' %(str(datetime.datetime.now()), agg_per_site_file_path), flush=True)
    
    return

def get_output(params, output_Q, methylation_event, header_dict, ref_pos_dict):
    header=pysam.AlignmentHeader.from_dict(header_dict)
    
    bam_threads=params['bam_threads']

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
       
    counter_check=0
    with open(per_read_file_path,'w') as per_read_file:
        per_read_file.write('read_name\tchromosome\tref_position_before\tref_position\tread_position\tstrand\tmethylation_score\tmean_read_qscore\tread_length\tread_phase\n')
        
        with pysam.AlignmentFile(bam_output, "wb", threads=bam_threads, header=header) as outf:
            while True:
                if methylation_event.is_set() and output_Q.empty():
                    break
                else:
                    try:
                        res = output_Q.get(block=False, timeout=10)
                        #continue
                        if counter//10000>counter_check:
                            counter_check=counter//10000
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
                                
                                phase=0
                                phase=read.get_tag('HP') if read.has_tag('HP') else 0
                                
                                
                                if float(mean_qscore)<qscore_cutoff or int(read_length)<length_cutoff:
                                    continue
                                    
                                for i in range(len(pred_list)):
                                    read_pos=candidate_list[i][0]+1
                                    ref_pos=candidate_list[i][1]
                                    score=pred_list[i]
                                    
                                    ref_pos_str_before=str(ref_pos) if ref_pos!=-1 else 'NA'
                                    ref_pos_str_after=str(ref_pos+1) if ref_pos!=-1 else 'NA'
                                    
                                    if ref_pos_str_before=='NA':    
                                        pass
                                    
                                    else:
                                        if score<mod_threshold and score>unmod_threshold:
                                                pass
                                        elif not skip_per_site:
                                            mod=score>=mod_threshold

                                            if (chrom, ref_pos,strand) not in per_site_pred:
                                                per_site_pred[(chrom, ref_pos,strand)]=[0]*6

                                            per_site_pred[(chrom, ref_pos,strand)][2*phase+mod]+=1

                                    per_read_file.write('%s\t%s\t%s\t%s\t%d\t%s\t%.4f\t%.2f\t%d\t%d\n' %(read_name, chrom, ref_pos_str_before, ref_pos_str_after, read_pos, strand, score, mean_qscore, read_length, phase))
                        
                                
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
    
    if skip_per_site:
        return 
    per_site_fields=['#chromosome', 'position_before', 'position','strand', 'ref_cpg',
                 'coverage','mod_coverage', 'unmod_coverage','mod_fraction',
                 'coverage_phase1','mod_coverage_phase1', 'unmod_coverage_phase1','mod_fraction_phase1',
                 'coverage_phase2','mod_coverage_phase2', 'unmod_coverage_phase2','mod_fraction_phase2']
    per_site_header='\t'.join(per_site_fields)+'\n'
    
    
    per_site_file_path=os.path.join(params['output'],'%s.per_site' %params['prefix'])
        
    with open(per_site_file_path, 'w') as per_site_file:
        per_site_file.write(per_site_header)

        for x in sorted(per_site_pred.keys()):
            chrom, pos, strand=x
            mod_call=per_site_pred[x]

            stats=get_stats_string(chrom, pos, strand, mod_call)
            if stats[0]>0:
                per_site_file.write(stats[1])
    
    print('%s: Finished Writing Per Site Methylation Output.' %str(datetime.datetime.now()), flush=True)
    print('%s: Per Site Prediction file: %s' %(str(datetime.datetime.now()), per_site_file_path), flush=True)
    return

def process(params,ref_pos_dict, signal_Q, output_Q, input_event, ref_seq_dict):
    torch.set_grad_enabled(False);
    
    dev=params['dev']
    motif_seq=params['motif_seq']
    exp_motif_seq=params['exp_motif_seq']
    motif_base=motif_seq[params['motif_ind'][0]]
    motif_ind=params['motif_ind']
    
    if params['mod_symbol']:
        mod_symbol=params['mod_symbol']
    elif motif_seq=='CG':
        mod_symbol='m'
    else:
        mod_symbol=motif_base
    
    seq_type=params['seq_type']
    
    position_based=params['position_based']
    
    base_map={'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
    
    cigar_map={'M':0, '=':0, 'X':0, 'D':1, 'I':2, 'S':2,'H':2, 'N':1, 'P':4, 'B':4}
    cigar_pattern = r'\d+[A-Za-z]'
    
    model, model_config=get_model(params)
    window=model_config['window']
    
    model.eval()
    model.to(dev);
    
    reads_per_round=100
    
    chunk_size=256 if dev=='cpu' else params['batch_size']
    
    total_candidate_list=[]
    total_feature_list=[]
    total_base_seq_list=[]
    total_MM_list=[]
    total_read_info=[]
    total_c_idx=[]
    total_unprocessed_reads=[]
    total_ref_seq_list=[]
    r_count=0
    
    dummy_ref_seq=4+np.zeros(2*window+1)
    
    ref_available=True if params['ref'] else False
    
    while True:
        if (signal_Q.empty() and input_event.is_set()):
            break
        
        try:
            
            chunk=signal_Q.get(block=False, timeout=10)
            #print('%s:  Output_qsize=%d   Signal_qsize=%d' %(str(datetime.datetime.now()), output_Q.qsize(), signal_Q.qsize()),flush=True)
            if output_Q.qsize()>200:
                time.sleep(30)
                if output_Q.qsize()>500:
                    time.sleep(60)
                print('Pausing output due to queue size limit. Output_qsize=%d   Signal_qsize=%d' %(output_Q.qsize(), signal_Q.qsize()), flush=True)
            for data in chunk:
                signal, move, read_dict, align_data=data

                is_mapped, is_forward, ref_name, reference_start, reference_end, read_length=align_data

                fq=read_dict['seq']
                qual=read_dict['qual']
                sequence_length=len(fq)
                reverse= not is_forward
                fq=revcomp(fq) if reverse else fq
                qual=qual[::-1] if reverse else qual

                if is_mapped and ref_available:
                    cigar_tuples = np.array([(int(x[:-1]), cigar_map[x[-1]]) for x in re.findall(cigar_pattern, read_dict['cigar'])])
                    ref_start=int(read_dict['ref_pos'])-1
                    aligned_pairs=get_aligned_pairs(cigar_tuples, ref_start)
                else:
                    aligned_pairs=None


                pos_list_c, pos_list_candidates, read_to_ref_pairs=get_candidates(fq, align_data, aligned_pairs, ref_pos_dict, exp_motif_seq, motif_base, motif_ind, position_based)

                pos_list_candidates=pos_list_candidates[(pos_list_candidates[:,0]>window)\
                                                        &(pos_list_candidates[:,0]<sequence_length-window-1)] if len(pos_list_candidates)>0 else pos_list_candidates

                if len(pos_list_candidates)==0:
                    total_unprocessed_reads.append(read_dict)
                    continue

                if not move[0]:
                    try:
                        tags={x.split(':')[0]:x for x in read_dict.pop('tags')}
                        start=int(tags['ts'].split(':')[-1])
                        mv=tags['mv'].split(',')

                        stride=int(mv[1])
                        move_table=np.fromiter(mv[2:], dtype=np.int8)
                        move=(stride, start, move_table)
                        read_dict['tags']=[x for x in tags.values() if x[:2] not in ['mv', 'ts', 'ML', 'MM']]
                    except KeyError:
                        print('Read:%s no move table or stride or signal start found' %read_dict['name'])
                        total_unprocessed_reads.append(read_dict)
                        continue

                base_seq=np.array([base_map[x] for x in fq])
                base_qual=10**((33-np.array([ord(x) for x in qual]))/10)
                mean_qscore=-10*np.log10(np.mean(base_qual))
                base_qual=(1-base_qual)[:,np.newaxis]

                if is_mapped and ref_available and not params['exclude_ref_features']:
                    ref_seq=ref_seq_dict[ref_name][:,1][read_to_ref_pairs[:, 1]][::-1] if reverse else ref_seq_dict[ref_name][:,0][read_to_ref_pairs[:, 1]]
                    per_site_ref_seq=np.array([ref_seq[candidate[0]-window: candidate[0]+window+1] for candidate in pos_list_candidates])
                else:
                    per_site_ref_seq=np.array([dummy_ref_seq for candidate in pos_list_candidates])

                mat=get_events(signal, move)
                if seq_type=='rna':
                    mat=np.flip(mat,axis=0)
                mat=np.hstack((mat, base_qual))

                try:
                    c_idx=[True if x in pos_list_c else False for x in pos_list_candidates[:,0]]
                    c_idx_count=np.vectorize(pos_list_c.get)(pos_list_candidates[c_idx,0])
                    c_idx_count[1:]=c_idx_count[1:]-c_idx_count[:-1]-1
                    MM='{}+{}?,'.format(motif_base,mod_symbol)+','.join(c_idx_count.astype(str))+';'
                    total_c_idx.append(c_idx)
                    total_MM_list.append(MM)

                except ValueError:
                    total_c_idx.append([])
                    total_MM_list.append(None)

                per_site_features=np.array([mat[candidate[0]-window: candidate[0]+window+1] for candidate in pos_list_candidates])
                per_site_base_seq=np.array([base_seq[candidate[0]-window: candidate[0]+window+1] for candidate in pos_list_candidates])

                total_candidate_list.append(pos_list_candidates)
                total_feature_list.append(per_site_features)
                total_base_seq_list.append(per_site_base_seq)
                total_ref_seq_list.append(per_site_ref_seq)

                total_read_info.append((read_dict, [align_data[1],align_data[2],align_data[5], mean_qscore]))
            
            if len(total_read_info)>=reads_per_round:
                read_counts=np.cumsum([len(x) for x in total_feature_list])[:-1]
                features_list=np.vstack(total_feature_list)
                base_seq_list=np.vstack(total_base_seq_list)
                ref_seq_list=np.vstack(total_ref_seq_list)
                
                pred_list=[model(batch_x.to(dev), batch_base_seq.to(dev), batch_ref_seq.to(dev)).cpu().numpy() for batch_x, batch_base_seq, batch_ref_seq in generate_batches(features_list, base_seq_list, window, ref_seq=ref_seq_list, batch_size = chunk_size)]

                                
                pred_list=np.vstack(pred_list)
                pred_list=np.split(pred_list.ravel(), read_counts)
                read_qual_list=[array.array('B',np.round(255*read_pred_list[c_idx]).astype(int)) for read_pred_list, c_idx in zip(pred_list, total_c_idx)]

                output_Q.put([True, total_read_info, total_candidate_list, total_MM_list, read_qual_list, pred_list])
                total_candidate_list, total_feature_list, total_base_seq_list, total_MM_list, total_read_info, total_c_idx=[], [], [], [], [], []
                total_ref_seq_list=[]
                
            if len(total_unprocessed_reads)>100:
                output_Q.put([False, total_unprocessed_reads])
                total_unprocessed_reads=[]

        except queue.Empty:
            pass            

    if len(total_read_info)>0:
        read_counts=np.cumsum([len(x) for x in total_feature_list])[:-1]
        features_list=np.vstack(total_feature_list)
        base_seq_list=np.vstack(total_base_seq_list)
        ref_seq_list=np.vstack(total_ref_seq_list)
        
        pred_list=[model(batch_x.to(dev), batch_base_seq.to(dev), batch_ref_seq.to(dev)).cpu().numpy() for batch_x, batch_base_seq, batch_ref_seq in generate_batches(features_list, base_seq_list, window, ref_seq=ref_seq_list, batch_size = chunk_size)]

        pred_list=np.vstack(pred_list)
        pred_list=np.split(pred_list.ravel(), read_counts)
        read_qual_list=[array.array('B',np.round(255*read_pred_list[c_idx]).astype(int)) for read_pred_list, c_idx in zip(pred_list, total_c_idx)]

        output_Q.put([True, total_read_info, total_candidate_list, total_MM_list, read_qual_list, pred_list])

    if len(total_unprocessed_reads)>0:
        output_Q.put([False, total_unprocessed_reads])

    return

def get_input(params, signal_Q, output_Q, input_event):   
    chrom_list=params['chrom_list']
    length_cutoff=params['length_cutoff']
    
    skip_unmapped=params['skip_unmapped']
    
    bam=params['bam']
    bam_file=pysam.AlignmentFile(bam,'rb',check_sq=False)
    
    print('%s: Building BAM index.' %str(datetime.datetime.now()), flush=True)
    bam_index=pysam.IndexedReads(bam_file)
    bam_index.build()
    print('%s: Finished building BAM index.' %str(datetime.datetime.now()), flush=True)
    
    input_=params['input']
    signal_files= [input_] if os.path.isfile(input_) else Path(input_).rglob("*.%s" %params['file_type'])

    chunk=[]
    non_primary_reads=[]
    reads_per_chunk=100
    
    if params['file_type']=='fast5':
        guppy_group=params['guppy_group']
        for filename in signal_files:
            with get_fast5_file(filename, mode="r") as f5:
                for read in f5.get_reads():
                    if signal_Q.qsize()>200:
                        time.sleep(20)
                        #print('Pausing input due to INPUT queue size limit. Signal_qsize=%d' %(signal_Q.qsize()), flush=True)
                    read_name=read.read_id
                    non_primary_reads=[]
                    try:
                        read_iter=bam_index.find(read_name)
                        for bam_read in read_iter:
                            if (params['ref'] and bam_read.is_mapped and bam_read.reference_name not in chrom_list)\
                            or bam_read.query_length < length_cutoff \
                            or (bam_read.is_mapped==False and skip_unmapped==True):
                                continue
                            
                            elif not (bam_read.is_supplementary or bam_read.is_secondary):
                                read_dict=bam_read.to_dict()
                                signal=read.get_raw_data()
                                
                                if params['fast5_move']:
                                    segment=read.get_analysis_attributes(guppy_group)['segmentation']
                                    start=read.get_analysis_attributes('%s/Summary/segmentation' %segment)['first_sample_template']
                                    stride=read.get_summary_data(guppy_group)['basecall_1d_template']['block_stride']
                                    move_table=read.get_analysis_dataset('%s/BaseCalled_template' %guppy_group, 'Move')
                                    move=(stride, start, move_table)
                                else:
                                    move=(None,None,None)
                                
                                align_data=(bam_read.is_mapped if params['ref'] else False, 
                                            bam_read.is_forward, bam_read.reference_name, bam_read.reference_start, bam_read.reference_end, bam_read.query_length)
                                data=(signal, move, read_dict, align_data)                                
                                chunk.append(data)
                                if len(chunk)>=reads_per_chunk:
                                    signal_Q.put(chunk)
                                    chunk=[]
                            
                            else:
                                pass
                                #non_primary_reads.append(read_dict)
                        
                        if len(non_primary_reads)>0:
                            output_Q.put([False, non_primary_reads])
                        
                    except KeyError:
                        continue
                    
    else:
        move=(None,None,None)
        for filename in signal_files:
            with p5.Reader(filename) as reader:
                for read in reader.reads():
                    if signal_Q.qsize()>200:
                        time.sleep(20)
                        print('Pausing input due to INPUT queue size limit. Signal_qsize=%d' %(signal_Q.qsize()), flush=True)
                        
                    read_name=str(read.read_id)
                    non_primary_reads=[]
                    try:
                        read_iter=bam_index.find(read_name)
                        for bam_read in read_iter:
                            
                            if (params['ref'] and bam_read.is_mapped and bam_read.reference_name not in chrom_list)\
                            or bam_read.query_length < length_cutoff \
                            or (bam_read.is_mapped==False and skip_unmapped==True):
                                continue
                            
                            elif not (bam_read.is_supplementary or bam_read.is_secondary):
                                read_dict=bam_read.to_dict()
                                signal=read.signal
                                align_data=(bam_read.is_mapped if params['ref'] else False, 
                                            bam_read.is_forward, bam_read.reference_name, bam_read.reference_start, bam_read.reference_end, bam_read.query_length)
                                data=(signal, move, read_dict, align_data)
                                chunk.append(data)
                                if len(chunk)>=reads_per_chunk:
                                    signal_Q.put(chunk)
                                    chunk=[]
                            
                            else:
                                pass
                                #non_primary_reads.append(read_dict)
                        
                        if len(non_primary_reads)>0:
                            output_Q.put([False, non_primary_reads])
                            
                    except KeyError:
                        #print('Read:%s not found in BAM file' %read_name, flush=True)
                        continue
    
    if len(chunk)>0:
        signal_Q.put(chunk)
        chunk=[]
        
    if len(non_primary_reads)>0:
        output_Q.put([False, non_primary_reads])
            
    input_event.set()
    return
    
def call_manager(params):
    print('%s: Starting Per Read Methylation Detection.' %str(datetime.datetime.now()), flush=True)
    if params['dev']!='cpu':
        torch.multiprocessing.set_start_method('spawn')
    
    torch.set_num_threads(1)
    
    pmanager = mp.Manager()
    
    bam=params['bam']
    bam_file=pysam.AlignmentFile(bam,'rb',check_sq=False)
    header_dict=bam_file.header.to_dict()
    
    print('%s: Getting motif positions from the reference.' %str(datetime.datetime.now()), flush=True)
    
    ref_seq_dict={}
    ref_pos_dict={}
    
    mod_positions_list=get_pos(params['mod_positions']) if params['mod_positions'] else None
    position_based=True if params['mod_positions'] else False
    
    if position_based:
        params['chrom_list']=[x for x in params['chrom_list'] if x in mod_positions_list.keys()]
    
    _=get_ref_to_num('ACGT')
    
    if params['ref'] and len(params['chrom_list'])>0:
        with mp.Pool(processes=params['threads']) as pool:
            res=pool.map(get_ref_info, zip(repeat(params), params['chrom_list']))
            for r in res:
                chrom, seq_array, fwd_pos_array, rev_pos_array=r
                ref_seq_dict[chrom]=seq_array
                
                if position_based:
                    ref_pos_dict[chrom]=(np.array(sorted(list(set(fwd_pos_array)&set(mod_positions_list[chrom][0])))),
                                         np.array(sorted(list(set(rev_pos_array)&set(mod_positions_list[chrom][1])))))
                   
                else:
                    ref_pos_dict[chrom]=(fwd_pos_array, rev_pos_array)
                
                
    params['position_based']=True if position_based or params['reference_motif_only'] else False
    
    res=None
    
    print('%s: Finished getting motif positions from the reference.' %str(datetime.datetime.now()), flush=True)
    
    signal_Q = pmanager.Queue()
    output_Q = pmanager.Queue()
    methylation_event=pmanager.Event()
    input_event=pmanager.Event()
    
    handlers = []
    
    input_process = mp.Process(target=get_input, args=(params, signal_Q, output_Q, input_event))
    input_process.start()
    
    if params['motif_seq']=='CG':
        output_process=mp.Process(target=get_cpg_output, args=(params, output_Q, methylation_event, header_dict, ref_pos_dict));
    else:
        output_process=mp.Process(target=get_output, args=(params, output_Q, methylation_event, header_dict, ref_pos_dict));
    output_process.start();
    
    for hid in range(max(1,params['threads']-1)):
        p = mp.Process(target=process, args=(params, ref_pos_dict, signal_Q, output_Q, input_event, ref_seq_dict));
        p.start();
        handlers.append(p);    
    
    input_process.join()
    print('%s:Reading inputs complete.' %str(datetime.datetime.now()), flush=True)   

    for job in handlers:
        job.join()
    
    methylation_event.set()
    
    print('%s: Model predictions complete. Wrapping up output.' %str(datetime.datetime.now()), flush=True)   
    
    output_process.join()
    
    return