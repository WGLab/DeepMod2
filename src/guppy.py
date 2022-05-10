from collections import defaultdict, ChainMap
import ont_fast5_api
import csv, time, itertools, copy, h5py, time, pysam

import datetime, os, shutil, argparse, sys, random

import multiprocessing as mp
import numpy as np

import numpy.lib.recfunctions as rf

from pathlib import Path

from .utils import *
from ont_fast5_api.fast5_interface import get_fast5_file

from tensorflow import keras
import tensorflow as tf


def get_bam_info(args):
    
    chrom, bam_path, fasta_path=args
    bam=pysam.Samfile(bam_path,'rb')
    read_info={}
    
    fastafile=pysam.FastaFile(fasta_path)
    
    if chrom not in fastafile.references:
        return read_info
    
    ref_seq=fastafile.fetch(chrom)

    for pcol in bam.pileup(contig=chrom, flag_filter=0x4|0x100|0x200|0x400|0x800, truncate=True):
        try:
            if ref_seq[pcol.pos].upper()=='C' and ref_seq[pcol.pos+1].upper()=='G':
                for read in pcol.pileups:
                    if read.alignment.is_reverse==False:
                        if not read.is_del:
                            if read.alignment.qname not in read_info:
                                read_info[read.alignment.qname]=[(0, chrom)]
                            read_info[read.alignment.qname].append((pcol.pos+1, read.query_position))
        
            elif ref_seq[pcol.pos].upper()=='G' and ref_seq[pcol.pos-1].upper()=='C':
                for read in pcol.pileups:
                    if read.alignment.is_reverse:
                        if not read.is_del:
                            if read.alignment.qname not in read_info:
                                read_info[read.alignment.qname]=[(1, chrom)]
                            read_info[read.alignment.qname].append((pcol.pos+1, read.alignment.query_length-read.query_position-1))
                            
        except IndexError:
            continue
    
    return read_info


def process_bam(params, pool):
    fastafile=pysam.FastaFile(params['fasta_path'])
    
    bam_info=[x for x in pool.imap_unordered(get_bam_info, zip(params['chrom_list'], itertools.repeat(params['bam_path']), itertools.repeat(params['fasta_path'])))]
    
    read_info=ChainMap(*bam_info)
        
    return read_info


def get_read_signal(read, guppy_group):
    segment=read.get_analysis_attributes(guppy_group)['segmentation']
    
    start=read.get_analysis_attributes('%s/Summary/segmentation' %segment)['first_sample_template']
    stride=read.get_summary_data(guppy_group)['basecall_1d_template']['block_stride']
    
    signal=read.get_raw_data()
    
    median=np.median(signal)
    mad=np.median(np.abs(signal-median))
    
    norm_signal=(signal-median)/mad
    
    base_level_data=[]
    base=0
    
    prev=start
    for i,x in enumerate(read.get_analysis_dataset('%s/BaseCalled_template' %guppy_group, 'Move')):
        if x:        
            base_level_data.append([prev,(i+1)*stride+start-prev])
            prev=(i+1)*stride+start
    
    return norm_signal, base_level_data





def detect(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    f5files, params, read_info, job_number = args
    
    output=os.path.join(params['output'],'intermediate_files', 'part_%d' %job_number)
    model=keras.models.load_model(get_model(params['model']))
    
    threshold=0.5
    strand_map={0:'+', 1:'-'}
    base_map={'A':0, 'C':1, 'G':2, 'T':3}
    
    window=params['window']

    
    counter=0
    with open(output, 'w') as outfile:
        features_list, pos_list, chr_list, strand_list, read_names_list = [], [], [], [], []
        
        for filename in f5files:
            with get_fast5_file(filename, mode="r") as f5:
                for read in f5.get_reads():
                    read_name=read.read_id
                    
                    try:
                        read_pos_list=read_info[read_name]
                    except KeyError:
                        continue

                    norm_signal, base_level_data = get_read_signal(read, params['guppy_group'])
                    
                    seq_len=len(base_level_data)

                    fq=read.get_analysis_dataset('Basecall_1D_000/BaseCalled_template', 'Fastq').split('\n')[1]
                    
                    mapped_strand, mapped_chrom=read_pos_list[0]

                    for x in read_pos_list[1:]:
                        pos, read_pos=x

                        if read_pos>window and read_pos<seq_len-window-1:
                            mat=[]
                            base_seq=[]

                            for x in range(read_pos-window, read_pos+window+1):
                                mat.append([np.mean(norm_signal[base_level_data[x][0]:base_level_data[x][0]+base_level_data[x][1]]), np.std(norm_signal[base_level_data[x][0]:base_level_data[x][0]+base_level_data[x][1]]), base_level_data[x][1]])
                                base_seq.append(base_map[fq[x]])

                            base_seq=np.eye(4)[base_seq]
                            mat=np.hstack((np.array(mat), base_seq))

                            if np.size(mat)!=21*7:  
                                continue

                            features_list.append(mat)
                            pos_list.append(pos)
                            chr_list.append(mapped_chrom)
                            strand_list.append(mapped_strand)
                            read_names_list.append(read_name)     

                            counter+=1

                            if counter==1000:
                                counter=0
                                pred_list=model.predict(np.array(features_list))

                                for i in range(len(pos_list)):
                                    pos, chrom, strand, read_name = pos_list[i], chr_list[i], strand_list[i], read_names_list[i]
                                    outfile.write('%s\t%s\t%d\t%s\t%.4f\t%d\n' %(read_name, chrom, pos, strand_map[strand], pred_list[i], 1 if pred_list[i]>=threshold else 0))

                                features_list, pos_list, chr_list, strand_list, read_names_list = [], [], [], [], []
                                outfile.flush()
                                os.fsync(outfile.fileno())

                                
        if counter>0:

            pred_list=model.predict(np.array(features_list))

            for i in range(len(pos_list)):
                pos, chrom, strand, read_name = pos_list[i], chr_list[i], strand_list[i], read_names_list[i]
                outfile.write('%s\t%s\t%d\t%s\t%.4f\t%d\n' %(read_name, chrom, pos, strand_map[strand], pred_list[i], 1 if pred_list[i]>=threshold else 0))

            features_list, pos_list, chr_list, strand_list, read_names_list = [], [], [], [], []
            outfile.flush()
            os.fsync(outfile.fileno())       

        
    return output
