from subprocess import PIPE, Popen
import os, shutil, pysam, sys, datetime
import numpy as np
from numba import jit
import torch
from itertools import repeat
import pysam
from .models import *
import torch.nn.utils.prune as prune
from tqdm import tqdm

model_dict={
    
    'bilstm_r9.4.1' : {       'path' : 'models/bilstm/R9.4.1',
                               'help':'BiLSTM model trained on chr2-21 of HG002, HG003 and HG004 R9.4.1 flowcells.',
                               'model_type':'bilstm'},
    
    'bilstm_r10.4.1_4khz_v3.5': {  'path' : 'models/bilstm/R10.4.1_4kHz_v3.5',
                              'help': 'BiLSTM model trained on chr2-21 of HG002, HG003 and HG004 R10.4.1 flowcells with 4kHz sampling, with basecalling performed by v3.5 Guppy/Dorado basecaller model.',
                              'model_type':'bilstm'},
    
        'bilstm_r10.4.1_4khz_v4.1': {  'path' : 'models/bilstm/R10.4.1_4kHz_v4.1',
                              'help': 'BiLSTM model trained on chr2-21 of HG002, HG003 and HG004 R10.4.1 flowcells with 4kHz sampling, with basecalling performed by v4.1 Guppy/Dorado basecaller model.',
                              'model_type':'bilstm'},
    
    'bilstm_r10.4.1_5khz_v4.3': {  'path' : 'models/bilstm/R10.4.1_5kHz_v4.3',
                              'help': 'BiLSTM model trained on chr2-21 of HG002, HG003 and HG004 R10.4.1 flowcells with 5kHz sampling, with basecalling performed by v4.3 Guppy/Dorado basecaller model.',
                              'model_type':'bilstm'},
    
    
    'transformer_r9.4.1' : {  'path' : 'models/transformer/R9.4.1',
                              'help':'Transformer model trained on chr2-21 of HG002, HG003 and HG004 R9.4.1 flowcells.',
                              'model_type':'transformer'},
    
    'transformer_r10.4.1_4khz_v3.5': { 'path' : 'models/transformer/R10.4.1_4kHz_v3.5',
                                  'help': 'Transfromer model trained on chr2-21 of HG002, HG003 and HG004 R10.4.1 flowcells with 4kHz sampling, with basecalling performed by v3.5 Guppy/Dorado basecaller model.',
                                  'model_type':'transformer'},    
    
        'transformer_r10.4.1_4khz_v4.1': { 'path' : 'models/transformer/R10.4.1_4kHz_v4.1',
                                  'help': 'Transfromer model trained on chr2-21 of HG002, HG003 and HG004 R10.4.1 flowcells with 4kHz sampling, with basecalling performed by v4.1 Guppy/Dorado basecaller model.',
                                  'model_type':'transformer'},    
    
        'transformer_r10.4.1_5khz_v4.3': { 'path' : 'models/transformer/R10.4.1_5kHz_v4.3',
                                  'help': 'Transfromer model trained on chr2-21 of HG002, HG003 and HG004 R10.4.1 flowcells with 5kHz sampling, with basecalling performed by v4.3 Guppy/Dorado basecaller model.',
                                  'model_type':'transformer'}, 
}

comp_base_map={'A':'T','T':'A','C':'G','G':'C'}

def revcomp(s):
    return ''.join(comp_base_map[x] for x in s[::-1])

def get_model_help():
    for n,model in enumerate(model_dict):
        print('-'*30)
        print('%d) Model Name: %s' %(n+1, model))
        print('Details: %s\n' %model_dict[model]['help'])
        
def get_model(params):
    model_name=params['model']
    model_type=None
    model_path=None
    
    if model_name in model_dict:
        dirname = os.path.dirname(__file__)
        model_info=model_dict[model_name]
        model_type = model_info['model_type']
        model_path = os.path.join(dirname,model_info['path'])
    
    else:
        try:
            model_type_ = model_name.split(',')[0]
            model_path_ = model_name.split(',')[1]

            if os.path.isfile(model_path_) and model_type_ in ['bilstm','transformer']:
                model_type = model_type_
                model_path = model_path_
                
        except IndexError:
            print('Incorrect model specified')
            sys.exit(2)
            
    if model_type=='bilstm':
        model = BiLSTM(model_dims=(21,10), num_layers=2, \
                     dim_feedforward=128, num_fc=128);

        checkpoint = torch.load(model_path,  map_location ='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if not params['disable_pruning']:
            module=model.classifier.fc
            prune.l1_unstructured(module, name="weight", amount=0.95)
            prune.remove(module, 'weight')

        return model

    elif model_type=='transformer':
        model = TransformerModel(model_dims=(21,10), num_layers=4, nhead=8, \
                     dim_feedforward=256, pe_dim=64, num_fc=128);

        checkpoint = torch.load(model_path,  map_location ='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])

        if not params['disable_pruning']:
            module=model.classifier.fc
            prune.l1_unstructured(module, name="weight", amount=0.5)
            prune.remove(module, 'weight')
            for l in model.transformer_encoder.layers:
                module=l.linear1
                prune.l1_unstructured(module, name="weight", amount=0.25)
                prune.remove(module, 'weight')
                module=l.linear2
                prune.l1_unstructured(module, name="weight", amount=0.25)
                prune.remove(module, 'weight')
                module=l.self_attn.out_proj
                prune.l1_unstructured(module, name="weight", amount=0.25)
                prune.remove(module, 'weight')

        return model
       
    else:
        print('Model: %s not found.' %params['model'], flush=True)
        sys.exit(2)

def generate_batches(features, base_seq, ref_seq=None, batch_size=512):        
        if len(ref_seq)==0:
            ref_seq=(4+torch.zeros(features.shape[0], 21)).type(torch.LongTensor)       
        else:
            ref_seq=torch.Tensor(ref_seq).type(torch.LongTensor)
            
        features=torch.Tensor(features)
        base_seq=torch.Tensor(base_seq).type(torch.LongTensor)
        
        for local_index in range(0, features.shape[0], batch_size):
            batch_x=features[local_index:(local_index + batch_size)]
            batch_base_seq=base_seq[local_index:(local_index + batch_size)]
            batch_ref_seq=ref_seq[local_index:(local_index + batch_size)]
            
            yield batch_x, batch_base_seq, batch_ref_seq

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
    b=np.full((len(x)+1,2),fill_value=0,dtype=np.int8)
    
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
    
    cg=np.where((b[:-1,0]==1)&(b[1:,1]==1))[0]
    return b, cg

def get_ref_info(args):
    ref_path, chrom=args
    ref_fasta=pysam.FastaFile(ref_path)
    seq=ref_fasta.fetch(chrom).upper()
    seq_array, pos_array=get_ref_to_num(seq)
    return chrom, seq_array, pos_array

def get_stats_string(chrom, pos, is_ref_cpg, cpg):
    unphased_rev_unmod, unphased_rev_mod, unphased_fwd_unmod, unphased_fwd_mod=cpg[0:4]
    phase1_rev_unmod, phase1_rev_mod, phase1_fwd_unmod, phase1_fwd_mod=cpg[4:8]
    phase2_rev_unmod, phase2_rev_mod, phase2_fwd_unmod, phase2_fwd_mod=cpg[8:12]
    
    fwd_mod=unphased_fwd_mod+phase1_fwd_mod+phase2_fwd_mod
    fwd_unmod=unphased_fwd_unmod+phase1_fwd_unmod+phase2_fwd_unmod
    fwd_total_stats=[fwd_mod+fwd_unmod,fwd_mod,fwd_unmod,fwd_mod/(fwd_mod+fwd_unmod) if fwd_mod+fwd_unmod>0 else 0]
    fwd_phase1_stats=[phase1_fwd_mod+phase1_fwd_unmod, phase1_fwd_mod, phase1_fwd_unmod, phase1_fwd_mod/(phase1_fwd_mod+phase1_fwd_unmod) if phase1_fwd_mod+phase1_fwd_unmod>0 else 0]
    fwd_phase2_stats=[phase2_fwd_mod+phase2_fwd_unmod, phase2_fwd_mod, phase2_fwd_unmod, phase2_fwd_mod/(phase2_fwd_mod+phase2_fwd_unmod) if phase2_fwd_mod+phase2_fwd_unmod>0 else 0]
    
    fwd_str='{}\t{}\t{}\t+\t{}\t'.format(chrom, pos, pos+1, is_ref_cpg)+'{}\t{}\t{}\t{:.4f}\t'.format(*fwd_total_stats) + '{}\t{}\t{}\t{:.4f}\t'.format(*fwd_phase1_stats) + '{}\t{}\t{}\t{:.4f}\n'.format(*fwd_phase2_stats)
    
    
    rev_mod=unphased_rev_mod+phase1_rev_mod+phase2_rev_mod
    rev_unmod=unphased_rev_unmod+phase1_rev_unmod+phase2_rev_unmod
    rev_total_stats=[rev_mod+rev_unmod,rev_mod,rev_unmod,rev_mod/(rev_mod+rev_unmod) if rev_mod+rev_unmod>0 else 0]
    rev_phase1_stats=[phase1_rev_mod+phase1_rev_unmod, phase1_rev_mod, phase1_rev_unmod, phase1_rev_mod/(phase1_rev_mod+phase1_rev_unmod) if phase1_rev_mod+phase1_rev_unmod>0 else 0]
    rev_phase2_stats=[phase2_rev_mod+phase2_rev_unmod, phase2_rev_mod, phase2_rev_unmod, phase2_rev_mod/(phase2_rev_mod+phase2_rev_unmod) if phase2_rev_mod+phase2_rev_unmod>0 else 0]
    
    rev_str='{}\t{}\t{}\t-\t{}\t'.format(chrom, pos+1, pos+2, is_ref_cpg)+'{}\t{}\t{}\t{:.4f}\t'.format(*rev_total_stats) + '{}\t{}\t{}\t{:.4f}\t'.format(*rev_phase1_stats) + '{}\t{}\t{}\t{:.4f}\n'.format(*rev_phase2_stats)
    
    
    agg_total_stats=[fwd_total_stats[0]+rev_total_stats[0], fwd_total_stats[1]+rev_total_stats[1], fwd_total_stats[2]+rev_total_stats[2], (fwd_total_stats[1]+rev_total_stats[1])/(fwd_total_stats[0]+rev_total_stats[0]) if fwd_total_stats[0]+rev_total_stats[0]>0 else 0]
    
    agg_phase1_stats=[fwd_phase1_stats[0]+rev_phase1_stats[0], fwd_phase1_stats[1]+rev_phase1_stats[1], fwd_phase1_stats[2]+rev_phase1_stats[2], (fwd_phase1_stats[1]+rev_phase1_stats[1])/(fwd_phase1_stats[0]+rev_phase1_stats[0]) if fwd_phase1_stats[0]+rev_phase1_stats[0]>0 else 0]
    
    agg_phase2_stats=[fwd_phase2_stats[0]+rev_phase2_stats[0], fwd_phase2_stats[1]+rev_phase2_stats[1], fwd_phase2_stats[2]+rev_phase2_stats[2], (fwd_phase2_stats[1]+rev_phase2_stats[1])/(fwd_phase2_stats[0]+rev_phase2_stats[0]) if fwd_phase2_stats[0]+rev_phase2_stats[0]>0 else 0]
    
    agg_str='{}\t{}\t{}\t{}\t'.format(chrom, pos, pos+2, is_ref_cpg)+'{}\t{}\t{}\t{:.4f}\t'.format(*agg_total_stats) + '{}\t{}\t{}\t{:.4f}\t'.format(*agg_phase1_stats) + '{}\t{}\t{}\t{:.4f}\n'.format(*agg_phase2_stats)
    
    return [(agg_total_stats[0], agg_str),(fwd_total_stats[0], fwd_str),(rev_total_stats[0], rev_str)]

def get_per_site(params, input_list):
    qscore_cutoff=params['qscore_cutoff']
    length_cutoff=params['length_cutoff']
    
    mod_threshold=params['mod_t']
    unmod_threshold=params['unmod_t']
    
    cpg_ref_only=not params['include_non_cpg_ref']
        
    print('%s: Starting Per Site Methylation Detection.' %str(datetime.datetime.now()), flush=True)
    
    total_files=len(input_list)
    print('%s: Reading %d files.' %(str(datetime.datetime.now()), total_files), flush=True)
    pbar = tqdm(total=total_files)

    per_site_pred={}

    for read_pred_file in input_list:
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
                is_forward=1 if strand=='+' else 0
                
                idx=4*phase+2*is_forward
                
                is_ref_cpg=True if is_ref_cpg =='True' else False
                zero_based_fwd_pos=pos if strand=='+' else pos-1

                if (chrom, zero_based_fwd_pos) not in per_site_pred:
                    per_site_pred[(chrom, zero_based_fwd_pos)]=[0]*12+[is_ref_cpg]
                
                per_site_pred[(chrom, zero_based_fwd_pos)][idx+mod]+=1

        pbar.update(1)
    pbar.close()

    print('%s: Writing Per Site Methylation Detection.' %str(datetime.datetime.now()), flush=True)    
    
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
            
            agg_stats, fwd_stats, rev_stats=get_stats_string(chrom, pos, is_ref_cpg, cpg)
            if agg_stats[0]>0:
                agg_per_site_file.write(agg_stats[1])

            if fwd_stats[0]>0:
                per_site_file.write(fwd_stats[1])

            if rev_stats[0]>0:
                per_site_file.write(rev_stats[1])
    
    print('%s: Finished Writing Per Site Methylation Output.' %str(datetime.datetime.now()), flush=True)
    print('%s: Per Site Prediction file: %s' %(str(datetime.datetime.now()), per_site_file_path), flush=True)
    print('%s: Aggregated Per Site Prediction file: %s' %(str(datetime.datetime.now()), agg_per_site_file_path), flush=True)