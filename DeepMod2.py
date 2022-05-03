from collections import defaultdict
import csv, time, itertools, copy, h5py, time, re, random

import datetime, os, shutil, argparse, sys, pysam

import multiprocessing as mp
import numpy as np

import numpy.lib.recfunctions as rf

from pathlib import Path

def run(params):
    from src import modDetect
    read_pred_file=modDetect.per_read_predict(params)
    
    site_pred_file=modDetect.per_site_detect(read_pred_file, params)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    subparsers = parser.add_subparsers(title="basecaller", dest="basecaller")
    
    parent_parser = argparse.ArgumentParser(add_help=False)
    
    parent_parser.add_argument("--fast5", help='Path to folder containing tombo requiggle Fast5 files. Fast5 files will be recusrviely searched', type=str, required=True)
    parent_parser.add_argument("--output", help='Path to folder where features will be stored', type=str)
    parent_parser.add_argument("--threads", help='Number of processors to use',type=int, default=1)
    parent_parser.add_argument("--file_name", help='Name of the output file',type=str, default='output')
    parent_parser.add_argument("--chrom", nargs='*',  help='A space/whitespace separated list of contigs, e.g. chr3 chr6 chr22.')
    parent_parser.add_argument('--wgs_contigs_type', \
                        help="""Options are "with_chr", "without_chr" and "all",\
                        "with_chr" option will assume \
                        human genome and run DeepMod2 on chr1-22 X Y, "without_chr" will \
                        run on chromosomes 1-22 X Y if the BAM and reference genome files \
                        use chromosome names without "chr". "all" option will run \
                        DeepMod2 on each contig present in reference genome FASTA file.""", \
                        type=str, default='all') 

    
    guppy_parser = subparsers.add_parser("guppy", parents=[parent_parser],
                                      add_help=True,
                                      description="Guppy basecaller",
                                      help="Call methylation from Guppy FAST5 files")
    
    guppy_parser.add_argument("--model", help='Name of the model. Default model is "guppy_na12878" but you can provide a path to your own model.',type=str, default='guppy_na12878')
    
    guppy_parser.add_argument("--bam", help='Path to bam file if Guppy basecaller is user. BAM file is not needed with Tombo fast5 files.', type=str, required=True)
    guppy_parser.add_argument("--ref", help='Path to reference file', type=str, required=True)
    guppy_parser.add_argument("--guppy_group", help='Name of the guppy basecall group',type=str, default='Basecall_1D_000')
    

    
    tombo_parser = subparsers.add_parser("tombo", parents=[parent_parser],
                                      add_help=True,
                                      description="Tombo basecaller",
                                      help="Call methylation from Tombo FAST5 files")
    
    tombo_parser.add_argument("--tombo_group", help='Name of the tombo group',type=str, default='RawGenomeCorrected_000')
    tombo_parser.add_argument("--model", help='Name of the model. Default model is "tombo_na12878" but you can provide a path to your own model.',type=str, default='tombo_na12878')
    
    args = parser.parse_args()
    
    t=time.time()

    print('%s: Starting DeepMod2.' %str(datetime.datetime.now()), flush=True)
            
    if not args.output:
        args.output=os.getcwd()
        
    if args.chrom:
        chrom_list= args.chrom
        
    else:
        if args.wgs_contigs_type=='with_chr':
            chrom_list=['chr%d' %d for d in range(1,23)] + ['chrX','chrY']

        elif args.wgs_contigs_type == 'without_chr':
            chrom_list=['%d' %d for d in range(1,23)] + ['X', 'Y']

        elif args.wgs_contigs_type == 'all':
            fastafile=pysam.FastaFile(args.ref)
            chrom_list=fastafile.references
    
    params={'fast5':args.fast5, 'output':args.output, 'threads':args.threads, 'file_name':args.file_name, 'window':10, 'chrom_list':chrom_list, 'model':args.model, 'basecaller':args.basecaller}
    
    if args.basecaller == 'guppy':
        params.update({'bam_path':args.bam, 'fasta_path':args.ref , 'guppy_group':args.guppy_group})
        
    else:
        params.update({'tombo_group': args.tombo_group})
        
        
    os.makedirs(params['output'], exist_ok=True)
    
    print('\n%s: \nCommand: python %s\n' %(str(datetime.datetime.now()), ' '.join(sys.argv)), flush=True)
    
    with open(os.path.join(args.output,'args'),'w') as file:
        file.write('Command: python %s\n\n\n' %(' '.join(sys.argv)))
        file.write('------Parameters Used For Running DeepMod2------\n')
        for k in vars(args):
            file.write('{}: {}\n'.format(k,vars(args)[k]) )
                
    run(params)
    
    print('\n%s: Time elapsed=%.4fs' %(str(datetime.datetime.now()),time.time()-t), flush=True)
