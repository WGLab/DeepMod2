from collections import defaultdict, ChainMap
import time, itertools, h5py, pysam
import datetime, os, shutil, argparse, sys, re, array
import os
from itertools import repeat
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

def get_candidates(read_seq, align_data, aligned_pairs, ref_pos_dict):    
    is_mapped, is_forward, ref_name, reference_start, reference_end, read_length=align_data

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

    return aligned_pairs_ref_wise, aligned_pairs_read_wise_original


@jit(nopython=True)
def get_aligned_pairs(cigar_tuples, ref_start):
    alen=np.sum(cigar_tuples[:,0])
    pairs=np.zeros((alen,2)).astype(np.int32)

    i=0
    ref_cord=ref_start-1
    read_cord=-1
    pair_cord=0
    for i in range(len(cigar_tuples)):
        len_op, op= cigar_tuples[i,0], cigar_tuples[i,1]
        if op==0:
            for k in range(len_op):            
                ref_cord+=1
                read_cord+=1

                pairs[pair_cord,0]=read_cord
                pairs[pair_cord,1]=ref_cord
                pair_cord+=1

        elif op==2:
            for k in range(len_op):            
                read_cord+=1            
                pairs[pair_cord,0]=read_cord
                pairs[pair_cord,1]=-1
                pair_cord+=1

        elif op==1:
            for k in range(len_op):            
                ref_cord+=1            
                pairs[pair_cord,0]=-1
                pairs[pair_cord,1]=ref_cord
                pair_cord+=1
    return pairs

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

def get_ref_info(args):
    params, chrom=args
    motif_seq, motif_ind=params['motif_seq'], params['motif_ind']
    ref_fasta=pysam.FastaFile(params['ref'])
    seq=ref_fasta.fetch(chrom).upper()
    seq_array=get_ref_to_num(seq)
    
    fwd_pos_array, rev_pos_array=None, None
    if motif_seq:
        fwd_motif_anchor=np.array([m.start(0) for m in re.finditer(r'{}'.format(motif_seq), seq)])
        rev_motif_anchor=np.array([m.start(0) for m in re.finditer(r'{}'.format(revcomp(motif_seq)), seq)])

        fwd_pos_array=np.array(sorted(list(set.union(*[set(fwd_motif_anchor+i) for i in motif_ind]))))
        rev_pos_array=np.array(sorted(list(set.union(*[set(rev_motif_anchor+len(motif_seq)-1-i) for i in motif_ind]))))
    
    return chrom, seq_array, fwd_pos_array, rev_pos_array

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
                    
                labelled_pos_list[line[0]][strand_map[line[2]]][int(line[1])]=float(line[3])
    
    return labelled_pos_list

def write_to_npz(output_file_path, mat, base_qual, base_seq, ref_seq, label, ref_coordinates, read_name, ref_name):
    np.savez(output_file_path, mat=mat, base_qual=base_qual, base_seq=base_seq, ref_seq=ref_seq, label=label, ref_coordinates=ref_coordinates, read_name=read_name, ref_name=ref_name)
                       
def get_output(params, output_Q, process_event):
    output=params['output']
    
    reads_per_chunk=params['reads_per_chunk']
    
    chunk=1
    read_count=0
    
    output_file_path=os.path.join(output,'%s.features.%d.npz' %(params['prefix'], chunk))
        
    mat, base_qual, base_seq, ref_seq, label=[], [], [], [], []
    ref_coordinates, read_name, ref_name= [], [], []
    
    while True:
            if process_event.is_set() and output_Q.empty():
                break
            else:
                try:
                    res = output_Q.get(block=False)
                    #per_site_features, per_site_base_qual, per_site_base_seq, per_site_ref_seq, per_site_ref_coordinates, per_site_label, read_name_array, ref_name_array
                    
                    mat.append(res[0])
                    base_qual.append(res[1])
                    base_seq.append(res[2])
                    ref_seq.append(res[3])
                    ref_coordinates.append(res[4])
                    label.append(res[5])
                    read_name.append(res[6])
                    ref_name.append(res[7])
                    
                    read_count+=1
                   
                    if read_count%reads_per_chunk==0 and len(mat)>0:
                        mat=np.vstack(mat)
                        base_qual=np.vstack(base_qual)
                        base_seq=np.vstack(base_seq).astype(np.int8)
                        ref_seq=np.vstack(ref_seq).astype(np.int8)
                        label=np.hstack(label).astype(np.float16)
                        ref_coordinates=np.hstack(ref_coordinates)
                        read_name=np.hstack(read_name)
                        ref_name=np.hstack(ref_name)
                        
                        idx=np.random.permutation(np.arange(len(label)))
                        mat=mat[idx]
                        base_qual=base_qual[idx]
                        base_seq=base_seq[idx]
                        ref_seq=ref_seq[idx]
                        label=label[idx]
                        ref_coordinates=ref_coordinates[idx]
                        read_name=read_name[idx]
                        ref_name=ref_name[idx]
                        
                        print('%s: Number of reads processed = %d.' %(str(datetime.datetime.now()), read_count), flush=True)
                        
                        
                        write_to_npz(output_file_path, mat, base_qual, base_seq, ref_seq, label, ref_coordinates, read_name, ref_name)

                        chunk+=1
                        output_file_path=os.path.join(output,'%s.features.%d.npz' %(params['prefix'], chunk))
                        mat, base_qual, base_seq, ref_seq, label=[], [], [], [], []
                        ref_coordinates, read_name, ref_name= [], [], []
                        
                except queue.Empty:
                    pass
                    
    if read_count>0 and len(mat)>0:
        mat=np.vstack(mat)
        base_qual=np.vstack(base_qual)
        base_seq=np.vstack(base_seq).astype(np.int8)
        ref_seq=np.vstack(ref_seq).astype(np.int8)
        label=np.hstack(label).astype(np.float16)
        ref_coordinates=np.hstack(ref_coordinates)
        read_name=np.hstack(read_name)
        ref_name=np.hstack(ref_name)

        idx=np.random.permutation(np.arange(len(label)))
        mat=mat[idx]
        base_qual=base_qual[idx]
        base_seq=base_seq[idx]
        ref_seq=ref_seq[idx]
        label=label[idx]
        ref_coordinates=ref_coordinates[idx]
        read_name=read_name[idx]
        ref_name=ref_name[idx]

        print('%s: Number of reads processed = %d.' %(str(datetime.datetime.now()), read_count), flush=True)


        write_to_npz(output_file_path, mat, base_qual, base_seq, ref_seq, label, ref_coordinates, read_name, ref_name)

    return

def process(params, ref_pos_dict, signal_Q, output_Q, input_event, ref_seq_dict, labelled_pos_list):
    base_map={'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
    
    window=params['window']
    window_range=np.arange(-window,window+1)
    
    div_threshold=params['div_threshold']
    cigar_map={'M':0, '=':0, 'X':0, 'D':1, 'I':2, 'S':2,'H':2, 'N':3, 'P':4, 'B':4}
    cigar_pattern = r'\d+[A-Za-z]'
    
    ref_available=True if params['ref'] else False
    
    while True:
        if (signal_Q.empty() and input_event.is_set()):
            break
        
        try:
            data=signal_Q.get(block=False)
            signal, move, read_dict, align_data=data

            is_mapped, is_forward, ref_name, reference_start,reference_end, read_length=align_data

            fq=read_dict['seq']
            qual=read_dict['qual']
            sequence_length=len(fq)
            reverse= not is_forward
            fq=revcomp(fq) if reverse else fq
            qual=qual[::-1] if reverse else qual

            if is_mapped and True:
                cigar_tuples = np.array([(int(x[:-1]), cigar_map[x[-1]]) for x in re.findall(cigar_pattern, read_dict['cigar'])])
                ref_start=int(read_dict['ref_pos'])-1
                aligned_pairs=get_aligned_pairs(cigar_tuples, ref_start)
            else:
                continue

            init_pos_list_candidates, read_to_ref_pairs=get_candidates(fq, align_data, aligned_pairs, ref_pos_dict)
            init_pos_list_candidates=init_pos_list_candidates[(init_pos_list_candidates[:,0]>window)\
                                                    &(init_pos_list_candidates[:,0]<sequence_length-window-1)] if len(init_pos_list_candidates)>0 else init_pos_list_candidates
            
            if len(init_pos_list_candidates)==0:
                continue
                
            base_seq=np.array([base_map[x] for x in fq])
            ref_seq=ref_seq_dict[ref_name][:,1][read_to_ref_pairs[:, 1]][::-1] if reverse else ref_seq_dict[ref_name][:,0][read_to_ref_pairs[:, 1]]

            
            label_filter_idx=np.array([np.mean(ref_seq[candidate[0]-window: candidate[0]+window+1]!=\
                                                base_seq[candidate[0]-window: candidate[0]+window+1])<=div_threshold \
                                                for candidate in init_pos_list_candidates])
            pos_list_candidates=init_pos_list_candidates[label_filter_idx]
            
            
            if len(pos_list_candidates)==0:
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
                
            base_qual=10**((33-np.array([ord(x) for x in qual]))/10)
            mean_qscore=-10*np.log10(np.mean(base_qual))
            base_qual=(1-base_qual)
            
            mat=get_events(signal, move)
            
            per_site_features=np.array([mat[candidate[0]-window: candidate[0]+window+1] for candidate in pos_list_candidates])
            per_site_base_qual=np.array([base_qual[candidate[0]-window: candidate[0]+window+1] for candidate in pos_list_candidates])
            per_site_base_seq=np.array([base_seq[candidate[0]-window: candidate[0]+window+1] for candidate in pos_list_candidates])
            per_site_ref_seq=np.array([ref_seq[candidate[0]-window: candidate[0]+window+1] for candidate in pos_list_candidates])
            per_site_ref_coordinates=pos_list_candidates[:,1]
            per_site_label=np.array([labelled_pos_list[ref_name][1-is_forward][coord] for coord in per_site_ref_coordinates])
            read_name_array=np.array([read_dict['name'] for candidate in pos_list_candidates])
            ref_name_array=np.array([ref_name for candidate in pos_list_candidates])
            
            read_chunks=[per_site_features, per_site_base_qual, per_site_base_seq, per_site_ref_seq, per_site_ref_coordinates, per_site_label, read_name_array, ref_name_array]
                        
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
                                            bam_read.is_forward, bam_read.reference_name, bam_read.reference_start, bam_read.reference_end, bam_read.query_length)
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
                                            bam_read.is_forward, bam_read.reference_name, bam_read.reference_start, bam_read.reference_end, bam_read.query_length)
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
    
    ref_seq_dict={}
    ref_pos_dict={}
    
    labelled_pos_list={}
    
    if params['pos_list']:
        labelled_pos_list=get_pos(params['pos_list'])
        params['chrom']=[x for x in params['chrom'] if x in labelled_pos_list.keys()]
    
    motif_label=params['motif_label']
    _=get_ref_to_num('ACGT')
    ref_seq_dict={}
    
    with mp.Pool(processes=params['threads']) as pool:
        res=pool.map(get_ref_info, zip(repeat(params), params['chrom']))
        for r in res:
            chrom, seq_array, fwd_pos_array, rev_pos_array=r
            ref_seq_dict[chrom]=seq_array
            
            if params['pos_list']:
                ref_pos_dict[chrom]=(np.array(sorted(list(labelled_pos_list[chrom][0].keys()))), np.array(sorted(list(labelled_pos_list[chrom][1].keys()))))

            else:
                ref_pos_dict[chrom]=(fwd_pos_array, rev_pos_array)
                labelled_pos_list[chrom]={0:{}, 1:{}}
                for strand in [0,1]:
                    for pos in ref_pos_dict[chrom][strand]:
                        labelled_pos_list[chrom][strand][pos]=float(motif_label)
                        
                
                
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
        p = mp.Process(target=process, args=(params, ref_pos_dict, signal_Q, output_Q, input_event, ref_seq_dict, labelled_pos_list));
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
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--bam", help='Path to bam file', type=str, required=True)
    parser.add_argument("--window", help='Number of bases before or after the base of interest to include in the model. Total number of bases included in teh model will be 2xwindow+1.', type=int, default=10)
    parser.add_argument("--prefix", help='Prefix for the output files',type=str, default='output')
    parser.add_argument("--input", help='Path to folder containing POD5 or FAST5 files. Files will be recusrviely searched.', type=str, required=True)
    
    parser.add_argument("--output", help='Path to folder where features will be stored', type=str, required=True)
    
    parser.add_argument("--threads", help='Number of processors to use',type=int, default=1)
    
    parser.add_argument("--div_threshold", help='Divergence Threshold.',type=float, default=0.25)
    
    parser.add_argument("--reads_per_chunk", help='reads_per_chunk',type=int, default=100000)
    
    parser.add_argument("--ref", help='Path to reference FASTA file to anchor methylation calls to reference loci. If no reference is provided, only the motif loci on reads will be used.', type=str)
    
    parser.add_argument("--pos_list", help='Text file containing a list of positions to generate features for. Use either --pos_list or --motif to specify how to choose loci for feature generation, but not both. The file should be whitespace separated with the following information on each line: chrom pos strand label. The position is 0-based reference coordinate, strand is + for forward and - for negative strand; label is 1 for mod, 0 for unmod).', type=str)
    
    parser.add_argument("--file_type", help='Specify whether the signal is in FAST5 or POD5 file format. If POD5 file is used, then move table must be in BAM file.',choices=['fast5','pod5'], type=str, default='fast5',required=True)
    
    parser.add_argument("--guppy_group", help='Name of the guppy basecall group',type=str, default='Basecall_1D_000')
    parser.add_argument("--chrom", nargs='*',  help='A space/whitespace separated list of contigs, e.g. chr3 chr6 chr22. If not list is provided then all chromosomes in the reference are used.')
    parser.add_argument("--length_cutoff", help='Minimum cutoff for read length',type=int, default=0)
    parser.add_argument("--fast5_move", help='Use move table from FAST5 file instead of BAM file. If this flag is set, specify a basecall group for FAST5 file using --guppy_group parameter and ensure that the FAST5 files contains move table.', default=False, action='store_true')
    
    parser.add_argument("--motif", help='Motif for generating features followed by zero-based indices of nucleotides within the motif to generate features for. Use either --pos_list or --motif to specify how to choose loci for feature generation, but not both. Features will be generated for all loci of the read that map to a reference sequence that matches the motif. Multiple indices can be specified but they should refer to the same nucleotide letter.  If you use --motif, it is assumed that all loci have the same modification label and you need to specify the label using --motif_label.', nargs='*')
    
    parser.add_argument("--motif_label", help='Modification label for the motif. 0 is for unmodified and 1 is for modified.',type=int, choices=[0,1])
        
    args = parser.parse_args()
    
    if not args.output:
        args.output=os.getcwd()
    
    os.makedirs(args.output, exist_ok=True)
    
    
    if args.chrom:
        chrom_list=args.chrom
    else:
        chrom_list=pysam.Samfile(args.bam).references

        
     
    if args.motif and len(args.motif)>0:
        if args.pos_list is not None:
            print('Use either --motif or --pos_list but not both', flush=True)
            sys.exit(3)
            
        if args.motif_label is None:
            print('--motif_label should be specified with --motif option', flush=True)
            sys.exit(3)
            
        if len(args.motif)<2 or len(set(args.motif[0])-set('ACGT'))>0 or  all([a.isnumeric() for a in args.motif[1:]])==False:
            print('--motif not specified correctly', len(args.motif)<2, len(set(args.motif[0])-set('ACGT'))==0, all([a.isnumeric() for a in args.motif[1:]])==False,flush=True)
            sys.exit(3)

        else:
            motif=args.motif[0]
            motif_ind=[int(x) for x in args.motif[1:]]
            if len(set(motif[x] for x in motif_ind))!=1:
                print('motif base should be same for all indices', flush=True)
                sys.exit(3)
    else:
        motif=None
        motif_ind=None
        
        if args.pos_list is None:
            print('Use either --motif or --pos_list', flush=True)
            sys.exit(3)
            
    params=dict(bam=args.bam, 
            window=args.window,
            pos_list=args.pos_list, 
            ref=args.ref, 
            input=args.input,
            motif_seq=motif,
            motif_ind=motif_ind,
            motif_label=args.motif_label,
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
