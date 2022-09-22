from collections import defaultdict, ChainMap

import time, itertools, h5py, pysam

import datetime, os, shutil, argparse, sys

import multiprocessing as mp
import numpy as np

from pathlib import Path

from .utils import *
from ont_fast5_api.fast5_interface import get_fast5_file

from tensorflow import keras
import tensorflow as tf
from numba import jit

def get_bam_info(args):
    
    chrom, bam_path, fasta_path, supplementary=args
    bam=pysam.Samfile(bam_path,'rb')
    read_info={}
    
    fastafile=pysam.FastaFile(fasta_path)
    
    if chrom not in fastafile.references:
        return read_info
    
    ref_seq=fastafile.fetch(chrom)
    
    if supplementary:
        flag=0x4|0x100|0x200|0x400
    else:
        flag=0x4|0x100|0x200|0x400|0x800
        
    for pcol in bam.pileup(contig=chrom, flag_filter=flag, truncate=True, min_base_quality = 0):
        try:
            if ref_seq[pcol.pos].upper()=='C' and ref_seq[pcol.pos+1].upper()=='G':
                for read in pcol.pileups:
                    if read.alignment.is_reverse==False:
                        if not read.is_del:
                            if read.alignment.qname not in read_info:
                                read_info[read.alignment.qname]='+%s\t' %chrom
                            read_info[read.alignment.qname]+=',%d|%d' %(pcol.pos+1, read.query_position)
        
            elif ref_seq[pcol.pos].upper()=='G' and ref_seq[pcol.pos-1].upper()=='C':
                for read in pcol.pileups:
                    if read.alignment.is_reverse:
                        if not read.is_del:
                            if read.alignment.qname not in read_info:
                                read_info[read.alignment.qname]='-%s\t' %chrom
                            read_info[read.alignment.qname]+=',%d|%d' %(pcol.pos+1, read.alignment.query_length-read.query_position-1)
                            
        except IndexError:
            continue
    
    return read_info

def process_motifs(params, pool):
    fastafile=pysam.FastaFile(params['fasta_path'])
    
    motif_info=[x for x in pool.imap_unordered(get_motif_pos, zip(params['chrom_list'], itertools.repeat(params['bam_path']), itertools.repeat(params['fasta_path']), itertools.repeat(params['supplementary'])))]
    
    read_info=ChainMap(*bam_info)
        
    return read_info
    
def process_bam(params, pool):
    fastafile=pysam.FastaFile(params['fasta_path'])
    
    bam_info=[x for x in pool.imap_unordered(get_bam_info, zip(params['chrom_list'], itertools.repeat(params['bam_path']), itertools.repeat(params['fasta_path']), itertools.repeat(params['supplementary'])))]
    
    read_info=ChainMap(*bam_info)
        
    return read_info

@jit(nopython=True)
def get_events(signal, move, start, stride):
    rlen=np.sum(move)
    
    lengths=np.zeros(rlen)
    mean=np.zeros(rlen)
    std=np.zeros(rlen)
    data=np.zeros((rlen,9))
    cnt=0
    
    prev=start
    for i in range(move.shape[0]):
        if move[i]:
            sig_end=(i+1)*stride+start
            sig_len=sig_end-prev
            data[cnt, 8]=sig_len
            data[cnt, 4]=np.median(signal[prev:sig_end])
            data[cnt, 5]=np.median(np.abs(signal[prev:sig_end]-data[cnt, 4]))
            
            for y in range(prev, sig_end):
                data[cnt, 6]+=signal[y]
                
            data[cnt, 6]=data[cnt, 6]/data[cnt, 8]
        
            for y in range(prev, sig_end):
                data[cnt, 7]+=np.square(signal[y]-data[cnt, 6])

            data[cnt, 7]=np.sqrt(data[cnt, 7]/data[cnt, 8])
            
            
            for i in range(4):
                tmp_cnt=0
                for t in range(i*sig_len//4,min(sig_len, (i+1)*sig_len//4)):
                    data[cnt, i]+=signal[t+prev]
                    tmp_cnt+=1
                data[cnt, i]=data[cnt, i]/tmp_cnt
            
            prev=sig_end
            cnt+=1
            
    return data, rlen

def get_read_signal(read, guppy_group):
    segment=read.get_analysis_attributes(guppy_group)['segmentation']
    
    start=read.get_analysis_attributes('%s/Summary/segmentation' %segment)['first_sample_template']
    stride=read.get_summary_data(guppy_group)['basecall_1d_template']['block_stride']
    mean_qscore=read.get_summary_data(guppy_group)['basecall_1d_template']['mean_qscore']
    sequence_length=read.get_summary_data(guppy_group)['basecall_1d_template']['sequence_length']
    
    signal=read.get_raw_data()
    
    median=np.median(signal)
    mad=np.median(np.abs(signal-median))
    
    norm_signal=(signal-median)/mad
    move=read.get_analysis_dataset('%s/BaseCalled_template' %guppy_group, 'Move')
    
    base_level_data, rlen = get_events(norm_signal, move, start, stride)
    
    return base_level_data, rlen, mean_qscore, sequence_length





def detect(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    f5files, params, read_info, job_number = args
    
    output=os.path.join(params['output'],'intermediate_files', 'part_%d' %job_number)
    model=keras.models.load_model(get_model(params['model']))
    
    threshold=0.5
    strand_map={0:'+', 1:'-'}
    base_map={'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
    
    window=params['window']
    qscore_cutoff=params['qscore_cutoff']
    length_cutoff=params['length_cutoff']
    
    counter=0
    with open(output, 'w') as outfile:
        features_list, pos_list, pos_list_read, chr_list, strand_list, read_names_list, mean_qscore_list, sequence_length_list  = [], [], [], [], [], [], [], []
        
        for filename in f5files:
            with get_fast5_file(filename, mode="r") as f5:
                for read in f5.get_reads():
                    read_name=read.read_id
                    
                    try:
                        read_info_string=read_info[read_name].split()
                        mapped_strand, mapped_chrom=read_info_string[0][0], read_info_string[0][1:] 
                        read_pos_list=read_info_string[1][1:].split(',')
                    except KeyError:
                        continue

                    base_level_data, seq_len, mean_qscore, sequence_length = get_read_signal(read, params['guppy_group'])
                    
                    if mean_qscore<qscore_cutoff or sequence_length<length_cutoff:
                        continue
                
                    read_fastq_record=read.get_analysis_dataset('%s/BaseCalled_template' %params['guppy_group'], 'Fastq').split('\n')
                    fq=read_fastq_record[1]
                    qual=read_fastq_record[3]

                    for x in read_pos_list:

                        x=x.split('|')
                        pos, read_pos=int(x[0]), int(x[1])
                        if read_pos>window and read_pos<seq_len-window-1:
                            mat=base_level_data[read_pos-window: read_pos+window+1]
                            base_seq=[base_map[fq[x]] for x in range(read_pos-window, read_pos+window+1)]
                            base_qual=10.0**(-np.array([ord(q)-33 for q in qual[read_pos-window : read_pos+window+1]])/10)[:,np.newaxis]
                            base_seq=np.eye(4)[base_seq]
                            mat=np.hstack((np.array(mat), base_qual, base_seq))

                            if np.size(mat)!=21*14:  
                                continue

                            features_list.append(mat)
                            pos_list.append(pos)
                            pos_list_read.append(read_pos)
                            chr_list.append(mapped_chrom)
                            strand_list.append(mapped_strand)
                            read_names_list.append(read_name)     
                            mean_qscore_list.append(mean_qscore)
                            sequence_length_list.append(sequence_length)
                            counter+=1

                            if counter==1000:
                                counter=0
                                features_list=np.array(features_list)
                                features_list[:,:,8]=features_list[:,:,8]/np.sum(features_list[:,:,8],axis=1)[:, np.newaxis]
                                pred_list=model.predict(features_list)

                                for i in range(len(pos_list)):
                                    pos, read_pos, chrom, strand, read_name, mean_qscore, sequence_length = pos_list[i], pos_list_read[i], chr_list[i], strand_list[i], read_names_list[i], mean_qscore_list[i], sequence_length_list[i]
                                    outfile.write('%s\t%s\t%d\t%d\t%s\t%.4f\t%d\t%.4f\t%d\n' %(read_name, chrom, pos, read_pos+1, strand, pred_list[i], 1 if pred_list[i]>=threshold else 0, mean_qscore, sequence_length))

                                features_list, pos_list, pos_list_read, chr_list, strand_list, read_names_list, mean_qscore_list, sequence_length_list = [], [], [], [], [], [], [], []
                                outfile.flush()
                                os.fsync(outfile.fileno())

                                
        if counter>0:

            features_list=np.array(features_list)
            features_list[:,:,8]=features_list[:,:,8]/np.sum(features_list[:,:,8],axis=1)[:, np.newaxis]
            pred_list=model.predict(features_list)

            for i in range(len(pos_list)):
                pos, read_pos, chrom, strand, read_name, mean_qscore, sequence_length = pos_list[i], pos_list_read[i], chr_list[i], strand_list[i], read_names_list[i], mean_qscore_list[i], sequence_length_list[i]
                outfile.write('%s\t%s\t%d\t%d\t%s\t%.4f\t%d\t%.4f\t%d\n' %(read_name, chrom, pos, read_pos+1, strand, pred_list[i], 1 if pred_list[i]>=threshold else 0, mean_qscore, sequence_length))

            features_list, pos_list, pos_list_read, chr_list, strand_list, read_names_list, mean_qscore_list, sequence_length_list = [], [], [], [], [], [], [], []
            
            outfile.flush()
            os.fsync(outfile.fileno())       

        
    return output