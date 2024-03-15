from collections import defaultdict, ChainMap

import time, itertools, h5py, pysam

import datetime, os, shutil, argparse, sys, re, array

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import multiprocessing as mp
import numpy as np

from pathlib import Path

from ont_fast5_api.fast5_interface import get_fast5_file

from numba import jit

import queue, gzip
import pod5 as p5


base_to_num_map={'A':0, 'C':1, 'G':2, 'T':3, 'U':3,'N':4}

num_to_base_map={0:'A', 1:'C', 2:'G', 3:'T', 4:'N'}

comp_base_map={'A':'T','T':'A','C':'G','G':'C'}

def revcomp(s):
    return ''.join(comp_base_map[x] for x in s[::-1])

@jit(nopython=True)
def get_ref_to_num(x):
    b=np.full((len(x)+1,4),fill_value=0,dtype=np.int8)
    
    for i,l in enumerate(x):
        if l=='A':
            b[i,0]=0
            b[i,1]=3
            
        elif l=='T':
            b[i,0]=3
            b[i,1]=0
            
        elif l=='C':
            b[i,0]=1
            b[i,1]=2
            
        elif l=='G':
            b[i,0]=2
            b[i,1]=1
            
        else:
            b[i,0]=4
            b[i,1]=4
    
    b[-1,0]=4
    b[-1,1]=4
            
    return b

@jit(nopython=True)
def get_events(signal, move):
    stride, start, move_table=move
    median=np.median(signal)
    mad=np.median(np.abs(signal-median))
    
    signal=(signal-median)/mad
    
    signal[signal>5]=5
    signal[signal<-5]=-5
    
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

def get_pos(path):
    labelled_pos_list={}
    strand_map={'+':0, '-':1}
    
    with open(path) as file:
            for line in file:
                line=line.rstrip('\n').split('\t')
                if line[0] not in labelled_pos_list:
                    labelled_pos_list[line[0]]={0:{}, 1:{}}
                    
                labelled_pos_list[line[0]][strand_map[line[2]]][int(line[1])]=int(line[3])
    
    return labelled_pos_list

def write_to_npz(output_file_path, mat, base_qual, base_seq, ref_seq, label):
    mat=np.vstack(mat)
    base_qual=np.vstack(base_qual)
    base_seq=np.vstack(base_seq).astype(np.int8)
    ref_seq=np.vstack(ref_seq).astype(np.int8)
    label=np.hstack(label).astype(np.int8)
    np.savez(output_file_path, mat=mat, base_qual=base_qual, base_seq=base_seq, ref_seq=ref_seq, label=label)
                       
def get_output(params, output_Q, process_event):
    output=params['output']
    
    reads_per_chunk=params['reads_per_chunk']
    
    chunk=1
    read_count=0
    output_file_path=os.path.join(output,'%s.features.%d.npz' %(params['prefix'], chunk))
    mat, base_qual, base_seq, ref_seq, label=[], [], [], [], []
    
    while True:
            if process_event.is_set() and output_Q.empty():
                break
            else:
                try:
                    res = output_Q.get(block=False)
                    
                    mat.append(res[0])
                    base_qual.append(res[1])
                    base_seq.append(res[2])
                    ref_seq.append(res[3])
                    label.append(res[4])
                    
                    read_count+=1
                   
                    if read_count%reads_per_chunk==0:
                        print('%s: Number of reads processed = %d.' %(str(datetime.datetime.now()), read_count), flush=True)
                        
                        write_to_npz(output_file_path, mat, base_qual, base_seq, ref_seq, label)
                        
                        chunk+=1
                        output_file_path=os.path.join(output,'%s.features.%d.npz' %(params['prefix'], chunk))
                        mat, base_qual, base_seq, ref_seq, label=[], [], [], [], []
                        
                except queue.Empty:
                    pass
                    
    if read_count>0:
        write_to_npz(output_file_path, mat, base_qual, base_seq, ref_seq, label) 
    
    return

def process(params, ref_seq_dict, signal_Q, output_Q, input_event):
    base_map={'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
    
    window=10
    window_range=np.arange(-window,window+1)
    
    div_threshold=params['div_threshold']
    
    while True:
        if (signal_Q.empty() and input_event.is_set()):
            break
        
        try:
            data=signal_Q.get(block=False)
            signal, move, read_dict, align_data=data
            is_mapped, is_forward, ref_name, reference_start,reference_end, read_length,aligned_pairs_raw=align_data
            aligned_pairs=np.array(aligned_pairs_raw)
            aligned_pairs[aligned_pairs==None]=-1
            aligned_pairs=aligned_pairs[aligned_pairs[:,0]!=-1]
            aligned_pairs=aligned_pairs.astype(int)

            ref_data=ref_seq_dict[ref_name][aligned_pairs[:, 1]] if is_forward else ref_seq_dict[ref_name][aligned_pairs[:, 1]][::-1]
            label_pos=np.where(ref_data[:,3-is_forward]!=0)[0]
            label_pos=label_pos[(label_pos>window) & (label_pos<read_length-window-1)]            
            
            if len(label_pos)==0:
                continue
                
            init_label_range=(label_pos+(window_range[:,None])).transpose()
            
            ref_seq=ref_data[:,0] if is_forward else ref_data[:,1]
            
            fq=read_dict['seq'] if is_forward else revcomp(read_dict['seq'])
            base_seq=np.array([base_map[x] for x in fq])
            
            #filter segments with poor alignment
            ref_seq_filt=np.take(ref_seq, init_label_range, axis=0)
            base_seq_filt=np.take(base_seq, init_label_range, axis=0)
            segment_filter=np.mean(ref_seq_filt!=base_seq_filt,axis=1)<=div_threshold
            label_pos=label_pos[segment_filter]
            
            if len(label_pos)==0:
                continue
            
            label_range=(label_pos+(window_range[:,None])).transpose()
            
            base_qual=read_dict['qual']  if is_forward else read_dict['qual'][::-1]
            base_qual=1-10**((33-np.array([ord(x) for x in base_qual]))/10)
            
            if not move[0]:
                try:
                    tags={x.split(':')[0]:x for x in read_dict['tags']}
                    start=int(tags['ts'].split(':')[-1])
                    mv=tags['mv'].split(',')

                    stride=int(mv[1])
                    move_table=np.array([int(x) for x in mv[2:]])
                    move=(stride, start, move_table)
                except KeyError:
                    continue

            
            label=ref_data[label_pos,3-is_forward]-1
            mat=get_events(signal, move)
            
            
            read_chunks=[np.take(x, label_range, axis=0) for x in [mat, base_qual, base_seq, ref_seq]]
            read_chunks.append(label)
                        
            output_Q.put(read_chunks)


        except queue.Empty:
            pass            

    return

def get_input(params, signal_Q, output_Q, input_event):    
    chrom_list=params['chrom']
    
    length_cutoff=params['length_cutoff']
    
    bam=params['bam']
    bam_file=pysam.AlignmentFile(bam,'rb',check_sq=False)
    
    print('%s: Building BAM index.' %str(datetime.datetime.now()), flush=True)
    bam_index=pysam.IndexedReads(bam_file)
    bam_index.build()
    print('%s: Finished building BAM index.' %str(datetime.datetime.now()), flush=True)
    
    input_=params['input']
    signal_files= [input_] if os.path.isfile(input_) else Path(input_).rglob("*.%s" %params['file_type'])
    
    counter=0
    if params['file_type']=='fast5':
        guppy_group=params['guppy_group']
        for filename in signal_files:
            with get_fast5_file(filename, mode="r") as f5:
                for read in f5.get_reads():
                    counter+=1
                    if counter%10000==0:
                        print('%s: Number of reads read = %d.' %(str(datetime.datetime.now()), counter), flush=True)
                    if signal_Q.qsize()>10000:
                        time.sleep(10)
                    
                    read_name=read.read_id
                    try:
                        read_iter=bam_index.find(read_name)
                        for bam_read in read_iter:
                            
                            if bam_read.flag & 0x900==0 and bam_read.reference_name in chrom_list  and bam_read.query_length>=length_cutoff:
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
                                            bam_read.is_forward, bam_read.reference_name, bam_read.reference_start, bam_read.reference_end, bam_read.query_length, bam_read.aligned_pairs)
                                data=(signal, move, read_dict, align_data)
                                signal_Q.put(data)
                        
                    except KeyError as error:
                        continue
            
    else:
        move=(None,None,None)
        for filename in signal_files:
            with p5.Reader(filename) as reader:
                for read in reader.reads():
                    counter+=1
                    if counter%10000==0:
                        print('%s: Number of reads processed = %d.' %(str(datetime.datetime.now()), counter), flush=True)
                        
                    if signal_Q.qsize()>10000:
                        time.sleep(10)
                        
                    read_name=str(read.read_id)
                    try:
                        read_iter=bam_index.find(read_name)
                        for bam_read in read_iter:
                            read_dict=bam_read.to_dict()
                            if bam_read.flag & 0x900==0 and bam_read.reference_name in chrom_list and bam_read.query_length>=length_cutoff:
                                read_dict=bam_read.to_dict()
                                signal=read.signal
                                align_data=(bam_read.is_mapped if params['ref'] else False, 
                                            bam_read.is_forward, bam_read.reference_name, bam_read.reference_start, bam_read.reference_end, bam_read.query_length, bam_read.aligned_pairs)
                                data=(signal, move, read_dict, align_data)
                                signal_Q.put(data)
                            
                    except KeyError:
                        continue
    
    input_event.set()
    return

def call_manager(params):        
    bam=params['bam']
    bam_file=pysam.AlignmentFile(bam,'rb',check_sq=False)
    header_dict=bam_file.header.to_dict()
    
    print('%s: Getting motif positions from the reference.' %str(datetime.datetime.now()), flush=True)
        
    ref_fasta=pysam.FastaFile(params['ref'])
    ref_seq_dict={rname: get_ref_to_num(ref_fasta.fetch(rname)) for rname in params['chrom']}    
    
    labelled_pos_list=get_pos(params['pos_list'])
    
    for chrom in ref_seq_dict.keys():
        for strand in [0,1]:
            for pos in labelled_pos_list[chrom][strand]:
                ref_seq_dict[chrom][pos,strand+2]=labelled_pos_list[chrom][strand][pos]+1
                
    print('%s: Finished getting motif positions from the reference.' %str(datetime.datetime.now()), flush=True)
    
    pmanager = mp.Manager()
    signal_Q = pmanager.Queue()
    output_Q = pmanager.Queue()
    process_event=pmanager.Event()
    input_event=pmanager.Event()
    
    handlers = []
    
    input_process = mp.Process(target=get_input, args=(params, signal_Q, output_Q, input_event))
    input_process.start()
    handlers.append(input_process)
    
    for hid in range(max(1,params['threads']-1)):
        p = mp.Process(target=process, args=(params, ref_seq_dict, signal_Q, output_Q, input_event));
        p.start();
        handlers.append(p);
    
    output_process=mp.Process(target=get_output, args=(params, output_Q, process_event));
    output_process.start();
    
    for job in handlers:
        job.join()
    
    process_event.set()
    output_process.join()
    
    return 
    
if __name__ == '__main__':

    t=time.time()

    print('%s: Starting feature generation.' %str(datetime.datetime.now()))
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--bam", help='Path to bam file', type=str, required=True)
    parser.add_argument("--prefix", help='Prefix for the output files',type=str, default='output')
    parser.add_argument("--input", help='Path to folder containing POD5 or FAST5 files. Files will be recusrviely searched.', type=str, required=True)
    
    parser.add_argument("--output", help='Path to folder where features will be stored', type=str, required=True)
    
    parser.add_argument("--threads", help='Number of processors to use',type=int, default=1)
    
    parser.add_argument("--div_threshold", help='Divergence Threshold. 21bp windowsThere are three ways to specify the input: 1) path to a folder containing .npz files in which case all npz files will be used for training, 2) path to a single .npz file, 3) path to a text file containing paths of .npz files to use for training. ',type=float, default=0.25)
    
    parser.add_argument("--reads_per_chunk", help='reads_per_chunk',type=int, default=100000)
    
    parser.add_argument("--ref", help='Path to reference FASTA file to anchor methylation calls to reference loci. If no reference is provided, only the motif loci on reads will be used.', type=str)
    
    parser.add_argument("--pos_list", help='Tab separated chrom pos strand label. The position is 0-based reference coordinate, strand is + for forward and - for negative strand; label is 1 for mod, 0 for unmod).', type=str)
    parser.add_argument("--file_type", help='Specify whether the signal is in FAST5 or POD5 file format. If POD5 file is used, then move table must be in BAM file.',choices=['fast5','pod5'], type=str, default='fast5',required=True)
    
    parser.add_argument("--guppy_group", help='Name of the guppy basecall group',type=str, default='Basecall_1D_000')
    parser.add_argument("--chrom", nargs='*',  help='A space/whitespace separated list of contigs, e.g. chr3 chr6 chr22. If not list is provided then all chromosomes in the reference are used.')
    parser.add_argument("--length_cutoff", help='Minimum cutoff for read length',type=int, default=0)
    parser.add_argument("--fast5_move", help='Use move table from FAST5 file instead of BAM file. If this flag is set, specify a basecall group for FAST5 file using --guppy_group parameter and ensure that the FAST5 files contains move table.', default=False, action='store_true')
    
    args = parser.parse_args()
    
    if not args.output:
        args.output=os.getcwd()
    
    os.makedirs(args.output, exist_ok=True)
    
    
    if args.chrom:
        chrom_list=args.chrom
    else:
        chrom_list=pysam.Samfile(args.bam).references
        
        
    params=dict(bam=args.bam, 
            pos_list=args.pos_list, 
            ref=args.ref, 
            input=args.input,
            file_type=args.file_type,
            guppy_group=args.guppy_group,
            fast5_move=args.fast5_move,
            chrom=chrom_list, threads=args.threads,
            length_cutoff=args.length_cutoff, 
            output=args.output, prefix=args.prefix,
            div_threshold=args.div_threshold, reads_per_chunk=args.reads_per_chunk)
    
    print(args)
    call_manager(params)
    
    print('\n%s: Time elapsed=%.4fs' %(str(datetime.datetime.now()),time.time()-t))
