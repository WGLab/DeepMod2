import time, itertools, h5py, re

import datetime, os, shutil, argparse, sys

import multiprocessing as mp
import numpy as np

import numpy.lib.recfunctions as rf

from pathlib import Path
from .utils import *

from tensorflow import keras
import tensorflow as tf

def get_tombo_events_summary(events):
    rf_events=rf.structured_to_unstructured(events['norm_mean','norm_stdev','length'])
    
    rf_signal_cordinates=rf.structured_to_unstructured(events['length', 'start'])
    
    seq=''.join([x.decode('utf-8') for x in events['base']])
    
    return seq, rf_events, rf_signal_cordinates

def get_tombo_alignment_info(alignment_attrs):
    mapped_chrom = alignment_attrs['mapped_chrom']
    mapped_strand = alignment_attrs['mapped_strand']
    mapped_start = int(alignment_attrs['mapped_start'])
    mapped_end = int(alignment_attrs['mapped_end'])
    
    return mapped_chrom, mapped_strand, mapped_start, mapped_end


def getFeatures(f5_list, params):
    base_map={'A':0, 'C':1, 'G':2, 'T':3}
    rev_strand_map={'+':0, '-':1}
    
    tombo_group=params['tombo_group']

    window=params['window']

    features_list=[]
    pos_list=[]
    chr_list=[]
    label_list=[]
    strand_list=[]
    read_names_list=[]
    
    matcher=re.compile('CG')


    for filename in f5_list:
        f5 = h5py.File(str(filename), 'r')
        
        #get tombo resquiggle data
        try:
            #get event information
            
            alignment_attrs = f5['Analyses/%s/BaseCalled_template/Alignment' %tombo_group].attrs
            mapped_chrom, mapped_strand, mapped_start, mapped_end = get_tombo_alignment_info(alignment_attrs)
                
            read_name=get_attr(f5['Raw/Reads'], 'read_id').decode('utf-8')
            
            events = f5['Analyses/%s/BaseCalled_template/Events' %tombo_group]
            seq, rf_events, rf_signal_cordinates=get_tombo_events_summary(events)
            
            #get alignment attributes
            
            
        except KeyError:
            continue            
            
        #for each CG found in sequence extract features
        for t in matcher.finditer(seq,window,len(events)-1-window):
            i=t.start(0)

            pos=i+mapped_start+1 if mapped_strand=='+' else len(events)-i+mapped_start

            m1=rf_events[i-window:i+window+1]
            m2=np.eye(4)[[base_map[x] for x in seq[i-window:i+window+1]]]
            mat=np.hstack([m1,m2])
            
            if np.size(mat)!=21*7:  
                    continue
                    
            #append new data for current position to the lsit of all features 
            features_list.append(mat)
            pos_list.append(pos)
            chr_list.append(mapped_chrom)
            strand_list.append(rev_strand_map[mapped_strand])
            read_names_list.append(read_name) 
    
    
    features_list=np.array(features_list)
    return pos_list, chr_list, strand_list, read_names_list, features_list


def detect(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        
    strand_map={0:'+', 1:'-'}
    
    f5files, params, read_info, job_number = args
    
    threshold=0.5
    
    output=os.path.join(params['output'],'intermediate_files', 'part_%d' %job_number)
    
    model=keras.models.load_model(get_model(params['model']))
    
    with open(output, 'w') as outfile: 
    
        for f5_chunk in split_list(f5files, 50):
            
            pos_list, chr_list, strand_list, read_names_list, features_list = getFeatures(f5_chunk, params)
                
            if len(features_list)==0:
                continue
            pred_list=model.predict(features_list)
            
            for i in range(len(pos_list)):
                pos, chrom, strand, read_name = pos_list[i], chr_list[i], strand_list[i], read_names_list[i]
                outfile.write('%s\t%s\t%d\tN/A\t%s\t%.4f\t%d\n' %(read_name, chrom, pos, strand_map[strand], pred_list[i], 1 if pred_list[i]>=threshold else 0))
            outfile.flush()
            os.fsync(outfile.fileno())
    
    return output