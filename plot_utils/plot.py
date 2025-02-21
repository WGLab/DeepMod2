import pod5 as p5
import numpy as np
from pathlib import Path
import os, pysam
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import plotly.graph_objects as go

comp_base_map={'A':'T','T':'A','C':'G','G':'C'}
base_map={'A':0, 'C':1, 'G':2, 'T':3, 'U':3}
rev_base_map={0:'A', 1:'C', 2:'G', 3:'T'}
strand_map={'+':0, '-':1}

def get_x_and_y_axes(split_signal):
    x_axes = []
    y_axes = []
    x_axis = np.array([])
    y_axis = []
    
    for i, base_signal in enumerate(split_signal):
        x_axis = np.hstack([x_axis, np.linspace(i+0.05, i+1-0.05, base_signal.shape[0])])
        y_axis.append(base_signal)

    return x_axis, np.hstack(y_axis)


def plot_read(read_data, marker_transparency=0.8, line_plot=True, marker_size=20, lim=0, display_average=True, save_path=None):
    split_signals, seq=read_data['signal'], read_data['seq']
    if lim==0:
        u_lim=np.max(np.hstack(split_signals))
        l_lim=np.min(np.hstack(split_signals))
    else:
        u_lim, l_lim=lim, -1*lim
        
    K=len(seq)

    p_x_axis, p_y_axis=get_x_and_y_axes(split_signals)

    plt.figure(figsize=(12, 3),dpi=100)

    if line_plot:
            plt.plot(p_x_axis, p_y_axis, linewidth=0.5, alpha=0.5, color='green')
    plt.scatter(p_x_axis, p_y_axis,alpha=marker_transparency, s=marker_size, color='green', edgecolor='black')
    
    plt.axhline(y=0, linestyle='dotted')
    for i in range(K):
        plt.axvline(x=i, linestyle='dotted', ymin=-4, ymax=4)
    
    handles=[]
    if display_average:
        for i in range(0,K):
            plt.hlines(y=np.mean(split_signals[i]),xmin=i,xmax=i+1,color='red',linestyle='-')
            plt.hlines(y=np.median(split_signals[i]),xmin=i,xmax=i+1,color='blue',linestyle='-')
        
        handles.append(mpatches.Patch(color='red', label='Mean Signal'))
        handles.append(mpatches.Patch(color='blue', label='Median Signal'))
        
    plt.axis([0, K, 1.1*l_lim, 1.1*u_lim])
    plt.xticks(np.arange(0, K) + 1/2, list(seq))
    
    handles.append(Line2D([0], [0], marker='o', color='white', label='Signal', alpha=0.5, markeredgecolor='black', markerfacecolor='green', markersize=5))
    leg=plt.legend(handles=handles,loc='upper right',fontsize=10)
    for patch in leg.get_patches():
        patch.set_height(5)
    
    plt.ylabel('Signal')
    plt.xlabel('Bases')

    if save_path!=None:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.show()
    
def plot_single_sample(data, line_plot=False, color='green',lim=0, marker_size=10, marker_transparency=0.2, display_average=True, save_path=None):
    plt.figure(figsize=(12, 3),dpi=100)
    
    if lim==0:
        u_lim=max(np.max(np.hstack(x['signal'])) for x in data.values())
        l_lim=min(np.min(np.hstack(x['signal'])) for x in data.values())
    else:
        u_lim, l_lim=lim, -1*lim
        
    
    cons_seq=get_consensus(data)
    K=len(cons_seq)
    
    bin_list=[[] for i in range(K)]
            
    for read_data in data.values():
        for i in range(K):
            for signal in read_data['signal'][i]:
                bin_list[i].append(signal)

        p_x_axis, p_y_axis=get_x_and_y_axes(read_data['signal'])

        
        if line_plot:
            plt.plot(p_x_axis, p_y_axis, linewidth=0.5, alpha=0.5, color=color)
        
        plt.scatter(p_x_axis, p_y_axis, alpha=marker_transparency, s=marker_size, color=color, edgecolor=None)
    
    plt.axhline(y=0, linestyle='dotted')
    for i in range(K):
        plt.axvline(x=i, linestyle='dotted', ymin=-4, ymax=4)
    
    handles=[]
    if display_average:
        for i in range(0,K):
            plt.hlines(y=np.mean(bin_list[i]),xmin=i,xmax=i+1,color='black',linestyle='-')
            plt.hlines(y=np.median(bin_list[i]),xmin=i,xmax=i+1,color='blue',linestyle='-')
        handles.append(mpatches.Patch(color='black', label='Mean Signal'))
        handles.append(mpatches.Patch(color='blue', label='Median Signal'))

    plt.axis([0, K, 1.1*l_lim, 1.1*u_lim])
    plt.xticks(np.arange(0, K) + 1/2, cons_seq)
    
    handles.append(Line2D([0], [0], marker='o', color='white', label='Signal', alpha=0.5, markeredgecolor='black', markerfacecolor=color, markersize=5))
    leg=plt.legend(handles=handles,loc='upper right',fontsize=10)
    for patch in leg.get_patches():
        patch.set_height(5)
    plt.ylabel('Signal')
    plt.xlabel('Bases')
    
    if save_path!=None:
        plt.savefig(save_path,bbox_inches='tight')
    
    plt.show()
    
def plot_two_samples(sample1_data, sample2_data, label1='Mod', label2='Unmod', line_plot=False, lim=0, marker_size=10, marker_transparency=0.2, display_average=True, average_type='median', save_path=None, color1='green', color2='red'):

    plt.figure(figsize=(12, 3),dpi=100)
    
    tmp_dict=dict(sample1_data)
    tmp_dict.update(sample2_data)
    cons_seq=get_consensus(tmp_dict)
    
    if lim==0:
        u_lim=max(np.max(np.hstack(x['signal'])) for x in tmp_dict.values())
        l_lim=min(np.min(np.hstack(x['signal'])) for x in tmp_dict.values())
    else:
        u_lim, l_lim=lim, -1*lim
        
    K=len(cons_seq)

    sample_1_bin_list=[[] for i in range(K)]
    sample_2_bin_list=[[] for i in range(K)]

    
    for read_data in sample1_data.values():
        for i in range(K):
            for signal in read_data['signal'][i]:
                sample_1_bin_list[i].append(signal)

        p_x_axis, p_y_axis=get_x_and_y_axes(read_data['signal'])

        
        if line_plot:
            plt.plot(p_x_axis, p_y_axis, linewidth=0.5, alpha=0.5, color=color1)

        plt.scatter(p_x_axis, p_y_axis, alpha=marker_transparency, s=marker_size, color=color1, edgecolor=None)
        
    for read_data in sample2_data.values():
        for i in range(K):
            for signal in read_data['signal'][i]:
                sample_2_bin_list[i].append(signal)

        p_x_axis, p_y_axis=get_x_and_y_axes(read_data['signal'])

        
        if line_plot:
            plt.plot(p_x_axis, p_y_axis, linewidth=0.5, alpha=0.5, color=color2)

        plt.scatter(p_x_axis, p_y_axis, alpha=marker_transparency, s=marker_size, color=color2, edgecolor=None)
    
    plt.axhline(y=0, linestyle='dotted')
    for i in range(K):
        plt.axvline(x=i, linestyle='dotted', ymin=-4, ymax=4)
    
    handles=[]
    if display_average:
        for i in range(0,K):
            if average_type=='mean':
                plt.hlines(y=np.mean(sample_1_bin_list[i]),xmin=i,xmax=i+1,color=color1,linestyle='-')
                plt.hlines(y=np.mean(sample_2_bin_list[i]),xmin=i,xmax=i+1,color=color2,linestyle='-')
            
            elif average_type=='median':
                plt.hlines(y=np.median(sample_1_bin_list[i]),xmin=i,xmax=i+1,color=color1,linestyle='-')
                plt.hlines(y=np.median(sample_2_bin_list[i]),xmin=i,xmax=i+1,color=color2,linestyle='-')
                
        handles.append(mpatches.Patch(color=color1, label='{} {} Signal'.format(label1, average_type.capitalize())))
        handles.append(mpatches.Patch(color=color2, label='{} {} Signal'.format(label2, average_type.capitalize())))

    plt.axis([0, K, 1.1*l_lim, 1.1*u_lim])
    plt.xticks(np.arange(0, K) + 1/2, cons_seq)
    
    handles.append(Line2D([0], [0], marker='o', color='white', label='{} Signal'.format(label1), alpha=0.5, markeredgecolor='black', markerfacecolor='green', markersize=5))
    handles.append(Line2D([0], [0], marker='o', color='white', label='{} Signal'.format(label2), alpha=0.5, markeredgecolor='black', markerfacecolor='red', markersize=5))
    leg=plt.legend(handles=handles,loc='upper right',fontsize=10)
    
    for patch in leg.get_patches():
        patch.set_height(5)
    plt.ylabel('Signal')
    plt.xlabel('Bases')
    
    if save_path!=None:
        plt.savefig(save_path,bbox_inches='tight')
        
    plt.show()
    
def violin_plot(sample_data, avg_type='median', static_display=False, meanline_visible=False, figure_width=1000, figure_height=500, save_path=None):

    cons_seq=get_consensus(sample_data)
    
    K=len(cons_seq)
    d=[]
    
   
    j=0
    for i in range(K):
        for read_data in sample_data.values():
            if avg_type=='mean':
                d.append([j,np.mean(read_data['signal'][i])])
            elif avg_type=='median':
                d.append([j,np.median(read_data['signal'][i])])
        j+=1
        
    df=pd.DataFrame(d)
    df.rename(columns={0:'Position',1:'Signal'}, inplace=True)
    
    fig = go.Figure()

    fig.add_trace(go.Violin(x=df['Position'],
                            y=df['Signal'],
                            points=False,
                            line=dict(color="blue", width=0.5),meanline=dict(color="blue", width=2),
                            meanline_visible=meanline_visible)
                 )
    
    tickvals=np.arange(0,len(cons_seq))
    ticktext=['%d<br>%s' %(a-len(cons_seq)//2,b) for a,b in zip(tickvals, cons_seq)]
    fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext)
    fig.update_layout(violingap=0, violinmode='overlay')
    

    
    if static_display:
        fig.update_layout(autosize=False, width=figure_width, height=figure_height)
        if save_path!=None:
            fig.write_image(save_path)
        fig.show(renderer="svg")
    else:
        if save_path!=None:
            fig.write_html(save_path)
        fig.show()
    
    return 
    
    
def compare_violin_plot(sample1_data, sample2_data, label1='Mod', label2='Unmod', avg_type='median', static_display=False, meanline_visible=False, figure_width=1000, figure_height=500, save_path=None, test_type='mw', test_method="auto", display_pval=True):
    tmp_dict=dict(sample1_data)
    tmp_dict.update(sample2_data)
    cons_seq=get_consensus(tmp_dict)
    
    K=len(cons_seq)
    d=[]
    
   
    j=0
    for i in range(K):
        for read_data in sample1_data.values():
            if avg_type=='mean':
                d.append([j,np.mean(read_data['signal'][i])])
            elif avg_type=='median':
                d.append([j,np.median(read_data['signal'][i])])
        j+=1
        
    pos_df=pd.DataFrame(d)
    pos_df.rename(columns={0:'Position',1:'Signal'}, inplace=True)
    pos_df['Sample']=label1

    d=[]
    
   
    j=0
    for i in range(K):
        for read_data in sample2_data.values():
            if avg_type=='mean':
                d.append([j,np.mean(read_data['signal'][i])])
            elif avg_type=='median':
                d.append([j,np.median(read_data['signal'][i])])
        j+=1
        
    neg_df=pd.DataFrame(d)
    neg_df.rename(columns={0:'Position',1:'Signal'}, inplace=True)
    neg_df['Sample']=label2

    df=pd.concat([pos_df, neg_df])
    fig = go.Figure()

    fig.add_trace(go.Violin(x=df['Position'][ df['Sample'] == label1],
                            y=df['Signal'][ df['Sample'] == label1 ],
                            legendgroup=label1, scalegroup=label1, name=label1, points=False,
                            side='negative',
                            line=dict(color="blue", width=0.5),meanline=dict(color="blue", width=2),
                            meanline_visible=meanline_visible)
                 )
    fig.add_trace(go.Violin(x=df['Position'][ df['Sample'] == label2],
                            y=df['Signal'][ df['Sample'] == label2 ],
                            legendgroup=label2, scalegroup=label2, name=label2, points=False,
                            side='positive',
                            line=dict(color="orange", width=0.5), meanline=dict(color="orange", width=2),
                            meanline_visible=meanline_visible)
                 )
    
    
    group_df=df.groupby(['Position', 'Sample'])['Signal'].apply(list)
    dist_stats={}
    for i in range(len(group_df)//2):
        if test_type=='ks':
            s=stats.ks_2samp(group_df.loc[i,:].loc[label1], group_df.loc[i,:].loc[label2], method=test_method)
        elif test_type=="mw":
            s=stats.mannwhitneyu(group_df.loc[i,:].loc[label1], group_df.loc[i,:].loc[label2], method=test_method)
        dist_stats[i]={'Position':i,'Base':cons_seq[i],'Statistic':s.statistic, 'Pvalue':s.pvalue}

    dist_stats_df=pd.DataFrame(dist_stats).T
    dist_stats_df=dist_stats_df.astype({'Position': 'int', 'Statistic': 'float32', 'Pvalue': 'float32',})

    tickvals=np.arange(0,len(cons_seq))
    if display_pval:
        ticktext=['{}<br>{}<br>{:0.1e}'.format(a-len(cons_seq)//2,b,c) for a,b,c in zip(tickvals, cons_seq, dist_stats_df.Pvalue)]
    else:
        ticktext=['%d<br>%s' %(a-len(cons_seq)//2,b) for a,b in zip(tickvals, cons_seq)]
    fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=ticktext,tickfont = dict(size = 8))
    fig.update_layout(violingap=0, violinmode='overlay')
    
    if static_display:
        fig.update_layout(autosize=False, width=figure_width, height=figure_height)
        if save_path!=None:
            fig.write_image(save_path)
        fig.show(renderer="svg")
    else:
        if save_path!=None:
            fig.write_html(save_path)
        fig.show()
    
    return dist_stats_df


def revcomp(s):
    return ''.join(comp_base_map[x] for x in s[::-1])

def get_consensus(data):
    K=len(next(iter(data.values()))['seq'])
    cons_seq_array=np.zeros((K,4))
    for x in data.values():
        for i in range(K):
            cons_seq_array[i][base_map[x['seq'][i]]]+=1

    cons_seq=''.join(rev_base_map[t] for t in np.argmax(cons_seq_array,axis=1))
    return cons_seq

def get_file_names(base_path):
    read_filename_dict={}
    
    if os.path.isdir(base_path):
        files=Path(base_path).rglob('*.pod5')
    else:
        files=[base_path]
    for read_path in files:
        read_path=str(read_path)
        with p5.Reader(read_path) as reader:
            for rname in reader.read_ids:
                read_filename_dict[rname]=read_path
                
    return read_filename_dict

def get_read_positions(bam_path, chrom, pos, strand, seq_type):
    flag=0x4|0x100|0x200|0x400|0x800

    read_info={}
    
    bam=pysam.Samfile(bam_path,'rb')

    for pcol in bam.pileup(contig=chrom, start=pos-1, end=pos, flag_filter=flag, truncate=True, min_base_quality = 0):
            if strand=='+':
                for read in pcol.pileups:
                    if read.alignment.is_reverse==False:
                        if read.is_del:
                            continue
                            print('DEL', read.alignment.qname)
                        else:
                            if seq_type=='dna':
                                read_info[read.alignment.qname]=(read.query_position, False, read.alignment.to_dict())
                            elif seq_type=='rna':
                                read_info[read.alignment.qname]=(read.alignment.query_length-read.query_position-1, False, read.alignment.to_dict())
            elif strand=='-':
                for read in pcol.pileups:
                    if read.alignment.is_reverse:
                        if read.is_del:
                            continue
                            print('DEL', read.alignment.qname)
                        else:
                            if seq_type=='dna':
                                read_info[read.alignment.qname]=(read.alignment.query_length-read.query_position-1, True, read.alignment.to_dict())
                            
                            elif seq_type=='rna':
                                read_info[read.alignment.qname]=(read.query_position, True, read.alignment.to_dict())
                            
    return read_info
    
def get_read_signal_raw(signal, move,norm_type):
    stride, start, move_table=move
    median=np.median(signal)
    mad=np.median(np.abs(signal-median))
    
    if norm_type=='MAD':
        norm_signal=(signal-median)/mad
    elif norm_type=='STD':
        norm_signal=(signal-np.mean(signal))/np.std(signal)
    
    move_len=len(move_table)
    move_index=np.where(move_table)[0]
    rlen=len(move_index)
    
    base_level_data=[]
    for i in range(len(move_index)-1):
        prev=move_index[i]*stride+start
        sig_end=move_index[i+1]*stride+start
        base_level_data.append([prev,sig_end-prev])

    return norm_signal, base_level_data

def get_signals(bam_path, chrom, pos, strand, read_filename_dict, base_path, seq_type, max_cov=1000, window_before=10, window_after=10, norm_type='STD'):
    read_info=get_read_positions(bam_path, chrom, pos, strand,seq_type)
    data={}
    
    if seq_type=='rna':
        window_before, window_after=window_after, window_before
        
    cov=0
    for read_name in read_info.keys():
        if cov > max_cov:
            break
        try:
            read_path=read_filename_dict[read_name]
        except KeyError:
            continue
            
        with p5.Reader(read_path) as reader:
                raw_read=next(reader.reads([read_name]))
                try:
                    read_pos, reverse, read_dict=read_info[read_name]

                except KeyError:
                    continue
                
                tags={x.split(':')[0]:x for x in read_dict['tags']}
                start=int(tags['ts'].split(':')[-1])
                mv=tags['mv'].split(',')

                stride=int(mv[1])
                move_table=np.fromiter(mv[2:], dtype=np.int8)
                move=(stride, start, move_table)
                
                signal=raw_read.signal
                
                fq=read_dict['seq']
                fq=revcomp(fq) if reverse else fq
                
                if seq_type=='rna':
                    fq=fq[::-1]
                    
                norm_signal, base_level_data = get_read_signal_raw(signal, move, norm_type)
                seq_len=len(fq)
                
                if read_pos>window_before+5 and read_pos<seq_len-window_after-1-5:
                    cov+=1
                    base_seq=fq[read_pos-window_before : read_pos+window_after+1]
                    
                    data[read_name]={}
                    
                    norm=[]
                    
                    if seq_type=='dna':
                        data[read_name]['seq']=base_seq
                        for x in range(read_pos-window_before, read_pos+window_after+1):
                            norm.append(np.array(norm_signal[base_level_data[x][0]:base_level_data[x][0]+base_level_data[x][1]]))
                        data[read_name]['signal']=norm
                        
                    elif seq_type=='rna':
                        data[read_name]['seq']=base_seq[::-1]
                        for x in range(read_pos-window_before, read_pos+window_after+1):
                            norm.append(np.array(norm_signal[base_level_data[x][0]:base_level_data[x][0]+base_level_data[x][1]][::-1]))
                        data[read_name]['signal']=norm[::-1]

    return data