# DeepMod2
DeepMod2 is a computational tool for detecting DNA methylation and modifications from Oxford Nanopore reads. It uses a BiLSTM model to predict per-read and per-site 5mC methylations for CpG sites. DeepMod2 can call methylation from POD5 and FAST5 files basecalled with Guppy or Dorado, as well as FAST5 files resquiggled with Tombo. DeepMod2 
now supports 5mC methylation calling from R10.4.1 flowcells, and allows annotation of BAM files with methylation tags.

<p align="center"> <img src="https://github.com/WGLab/DeepMod2/assets/35819083/49081e1a-a6e9-4d1a-9249-26c7d1ec9f8c"  width="50%" > </p>

DeepMod2 is distributed under the [MIT License by Wang Genomics Lab](https://wglab.mit-license.org/).

## Installation
Please refer to [Installation](https://github.com/WGLab/DeepMod2/blob/main/docs/Install.md) for how to install DeepMod2.

## Usage
Quick usage guide:
1. Basecall your FAST5/POD5 files with Guppy or Dorado using `--bam_out --moves_out` parameters to get a BAM file with move tables, e.g.
   ```
   guppy_basecaller -i INPUT_DIR -s BASECALL_DIR --bam_out --moves_out -c model.cfg
   ```
   Make sure to use the appropriate Guppy/Dorado model for your sequencing kit, but do not select a model with modification calling. You can supply a reference genome to Guppy/Dorado to get aligned BAM files, or use minimap2 to align these BAM files later.
2. Merge BAM files using samtools:
   ```
   find BASECALL_DIR \( -path "*/pass/*" -o -path "*/fail/*" \) -type f -name "*.bam"|samtools cat -b - |samtools sort -O BAM -o BASECALL_DIR/merged.bam
   ```
3. Run DeepMod2 by providing the folder with FAST5 or POD5 signal files and BAM file as input. You can provide reference FASTA file if the BAM file is aligned to get reference anchored methylation calls. Use multile cores and/or tensorflow-gpu for speedup.
   ```
   python PATH_TO_DEEPMOD2_REPOSITORY/deepmod2 detect-guppy --bam BASECALL_DIR/merged.bam --input INPUT_DIR --threads NUM_THREADS --ref REF.fa --output MOD_CALLS
   ```
   This will give you a per-read prediction text file `MOD_CALLS/output.per_read` and a methylation annotated BAM file `MOD_CALLS/output.bam`.
4. Convert per-read predictions into per-site predictions:
   ```
   python PATH_TO_DEEPMOD2_REPOSITORY/deepmod2 merge --input MOD_CALLS/output.per_read --output MOD_CALLS
   ```
   This will give you a per-site methylation prediction file MOD_CALLS/output.per_site with methylated vs unmethylated counts for each stranded CpG locus in reference.
5. Visualize the annotated BAM in IGV file after sorting and indexing the BAM file produced by DeepMod2. In IGV, select 'Color alignments by' and 'base modifications (5mC)'.
   ```
   samtools sort MOD_CALLS/output.bam -o MOD_CALLS/output.sorted.bam --write-index
   ```
   
<p align="center"> <img src="https://github.com/WGLab/DeepMod2/assets/35819083/c693ab27-f218-4478-9780-c027f740999d"  width="75%" > </p>


Please refer to [Usage](https://github.com/WGLab/DeepMod2/blob/main/docs/Usage.md) for details on how to use DeepMod2.

## Examples

Please refer to [Example](https://github.com/WGLab/DeepMod2/blob/main/docs/Example.md) for a complete tutorial on how to run DeepMod2 on both Guppy and Tombo data.
