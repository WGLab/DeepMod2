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
    
    if not params['skip_per_site']:
        site_pred_file=modDetect.per_site_detect([read_pred_file], params)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    main_subparsers = parser.add_subparsers(title="Options", dest="option")
    
    grand_parent_parser = argparse.ArgumentParser(add_help=False,)
    grand_parent_parser.add_argument("--file_name", help='Name of the output file',type=str, default='output')
    grand_parent_parser.add_argument("--output", help= 'Path to folder where intermediate and final files will be stored, default is current working directory', type=str)
    grand_parent_parser.add_argument("--threads", help='Number of processors to use',type=int, default=1)
    
    
    
    #detect_parser = main_subparsers.add_parser("detect", add_help=True, description="Detect 5mC methylation from Nanopore reads", help="Detect 5mC methylation from Nanopore reads", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    #detect_subparsers=detect_parser.add_subparsers(title="Basecaller", dest="basecaller")
    
    
    parent_parser = argparse.ArgumentParser(add_help=False, parents=[grand_parent_parser])
    
    parent_parser.add_argument("--fast5", help='Path to folder containing tombo requiggle Fast5 files. Fast5 files will be recusrviely searched', type=str, required=True)
    
    parent_parser.add_argument("--skip_per_site", help='Skip per site detection and stop after per-read methylation calling.' ,default=False, action='store_true')
    
    
    
    
    
    
    guppy_parser = main_subparsers.add_parser("detect-guppy", parents=[parent_parser],
                                      add_help=True,
                                      description="Guppy basecaller options",
                                      help="Call methylation from Guppy FAST5 files",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    
    guppy_parser.add_argument("--model", help='Name of the model. Default model is "guppy_na12878" but you can provide a path to your own model.',type=str, default='guppy_na12878')
    guppy_parser.add_argument("--chrom", nargs='*',  help='A space/whitespace separated list of contigs, e.g. chr3 chr6 chr22. If not list is provided then all chromosomes in the reference are used.')
    
    guppy_parser.add_argument("--bam", help='Path to bam file if Guppy basecaller is user. BAM file is not needed with Tombo fast5 files.', type=str, required=True)
    guppy_parser.add_argument("--ref", help='Path to reference file', type=str, required=True)
    guppy_parser.add_argument("--guppy_group", help='Name of the guppy basecall group',type=str, default='Basecall_1D_000')
    

    
    tombo_parser = main_subparsers.add_parser("detect-tombo", parents=[parent_parser],
                                      add_help=True,
                                      description="Tombo basecaller options",
                                      help="Call methylation from Tombo FAST5 files",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    tombo_parser.add_argument("--tombo_group", help='Name of the tombo group',type=str, default='RawGenomeCorrected_000')
    tombo_parser.add_argument("--model", help='Name of the model. Default model is "tombo_na12878" but you can provide a path to your own model.',type=str, default='tombo_na12878')
    
    
    merge_parser = main_subparsers.add_parser("merge", parents=[grand_parent_parser],
                                      add_help=True,
                                      description="Merge per-read calls into per-site calls",
                                      help="Merge per-read calls into per-site calls",  formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    merge_parser.add_argument("--inputs",   nargs='*', help= 'List of paths of per-read methylation calls to merge. File paths should be separated by space/whitespace. Use either --inputs or --list argument, but not both.')
    merge_parser.add_argument("--list",  help=  'A file containing paths to per-read methylation calls to merge (one per line). Use either --inputs or --list argument, but not both.', type=str)
    
    if len(sys.argv)==1:
        parser.print_help()
        parser.exit()
        
    
    elif len(sys.argv)==2:
        if sys.argv[1]=='merge':
            merge_parser.print_help()
            merge_parser.exit()
        
        elif sys.argv[1]=='detect-guppy':
            guppy_parser.print_help()
            guppy_parser.exit()
            
        elif sys.argv[1]=='detect-tombo':
            tombo_parser.print_help()
            tombo_parser.exit()
            
        else:
            print('invalid option')
            sys.exit()

    args = parser.parse_args()
    
    
    
    t=time.time()

    print('%s: Starting DeepMod2.' %str(datetime.datetime.now()), flush=True)
            
    if not args.output:
        args.output=os.getcwd()
    
    os.makedirs(args.output, exist_ok=True)

    if args.option=='merge':
        if args.inputs:
            read_pred_file_list= args.inputs 
        
        elif args.list:
            with open(args.list,'r') as file_list:
                read_pred_file_list=[x.rstrip('\n') for x in file_list.readlines()]
        
        from src import modDetect
        params={'output':args.output, 'file_name':args.file_name, 'threads':args.threads}
        site_pred_file=modDetect.per_site_detect(read_pred_file_list, params)
        
    else:
        
        basecaller='guppy' if args.option=='detect-guppy' else 'tombo'
        params={'fast5':args.fast5, 'output':args.output, 'threads':args.threads, 'file_name':args.file_name, 'window':10, 'model':args.model, 'basecaller':basecaller, 'skip_per_site':args.skip_per_site}
        
        if params['basecaller'] == 'guppy':
            if args.chrom:
                chrom_list= args.chrom

            else:
                bam_file=pysam.Samfile(args.bam,'rb')
                chrom_list=bam_file.references

            params.update({'bam_path':args.bam, 'fasta_path':args.ref , 'chrom_list':chrom_list, 'guppy_group':args.guppy_group})

        elif params['basecaller'] == 'tombo':
            params.update({'tombo_group': args.tombo_group})

        print('\n%s: \nCommand: python %s\n' %(str(datetime.datetime.now()), ' '.join(sys.argv)), flush=True)

        with open(os.path.join(args.output,'args'),'w') as file:
            file.write('Command: python %s\n\n\n' %(' '.join(sys.argv)))
            file.write('------Parameters Used For Running DeepMod2------\n')
            for k in vars(args):
                file.write('{}: {}\n'.format(k,vars(args)[k]) )

        run(params)
    
    print('\n%s: Time elapsed=%.4fs' %(str(datetime.datetime.now()),time.time()-t), flush=True)
