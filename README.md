# DeepMod2
DeepMod2 is a computational tool for detecting DNA 5mC methylation from Oxford Nanopore reads. It uses a BiLSTM model to predict per-read and per-site 5mC methylations for CpG sites and produces a methylation tagged BAM file. DeepMod2 can call methylation from POD5 and FAST5 files basecalled with Guppy or Dorado and provides models for R10.4.1 and R9.4.1 flowcells.

<p align="center"> <img src="https://github.com/WGLab/DeepMod2/assets/35819083/e0ef0b41-a469-427d-abaa-af2ba6292809"  width="50%" > </p>

DeepMod2 is distributed under the [MIT License by Wang Genomics Lab](https://wglab.mit-license.org/).

### Citing DeepMod2
Ahsan, M.U., Gouru, A., Chan, J. et al. A signal processing and deep learning framework for methylation detection using Oxford Nanopore sequencing. Nat Commun 15, 1448 (2024). https://doi.org/10.1038/s41467-024-45778-y
## Installation
Please refer to [Installation](https://github.com/WGLab/DeepMod2/blob/main/docs/Install.md) for how to install DeepMod2.

## Inference
Quick usage guide for model inference:
1. Basecall your FAST5/POD5 files with Dorado (using `--emit-moves`) or Guppy (using `--bam_out --moves_out`) parameters to get a BAM file with move tables:
   ```
   dorado basecaller MODEL INPUT_DIR --emit-moves > basecall.bam
   ```
   Make sure to use the appropriate Guppy/Dorado model for your sequencing kit. You can supply a reference genome to Guppy/Dorado to get aligned BAM files, or use minimap2 to align these BAM files later.
2. (Optional but recommended) Align basecalled reads to a reference genome while retaining the move tables:
   ```
   samtools fastq basecall.bam -T mv,ts | minimap2 -ax map-ont ref.fa - -y -t NUM_THREADS |samtools view -o aligned.bam
   ```
3. Run DeepMod2 by providing BAM file and the folder containing FAST5 or POD5 signal files as inputs. You can provide reference FASTA file to get reference anchored methylation calls and per-site frequencies if the BAM file is aligned. Specify the model you want to use and the file type of raw signal files. Use multiple cores and/or GPUs for speedup.
   
   a) If using an aligned BAM file input:
   ```
   python PATH_TO_DEEPMOD2_REPOSITORY/deepmod2 detect --bam reads.bam --input INPUT_DIR --model MODEL --file_type FILE_TYPE --threads NUM_THREADS --ref ref.fa --output MOD_CALLS
   ```
   
   b) If using an unaligned BAM file input:
   ```
   python PATH_TO_DEEPMOD2_REPOSITORY/deepmod2 detect --bam reads.bam --input INPUT_DIR --model MODEL --file_type FILE_TYPE --threads NUM_THREADS --output MOD_CALLS
   ```
   This will give you a per-read prediction text file `MOD_CALLS/output.per_read`, a per-site prediction file `MOD_CALLS/output.per_site`, a per-site prediction file with both strands aggregated `MOD_CALLS/output.per_site.aggregated`, and a methylation annotated BAM file `MOD_CALLS/output.bam`.
5. Visualize the annotated BAM file produced by DeepMod2 in IGV file. In IGV, select 'Color alignments by' and 'base modifications (5mC)'. The following steps will allow you to open the tagged BAM file in IGV:

   a) If an aligned BAM is given to DeepMod2, you only need to sort and index the DeepMod2 methylation tagged BAM file:
   ```
   samtools sort MOD_CALLS/output.bam -o MOD_CALLS/final.bam --write-index
   ```

   b) If an unaligned BAM is given to DeepMod2, first align the DeepMod2 methylation tagged BAM file (while preserving methylation tags MM and ML), then sort and index it:
   ```
   samtools fastq MOD_CALLS/output.bam -T MM,ML,mv,ts| minimap2 -ax map-ont ref.fa - -y -t NUM_THREADS |samtools sort -o MOD_CALLS/final.bam --write-index
   ```
   
<p align="center"> <img src="https://github.com/WGLab/DeepMod2/assets/35819083/c693ab27-f218-4478-9780-c027f740999d"  width="75%" > </p>


Please refer to [Usage.md](docs/Usage.md) for details on how to use DeepMod2.

## Training
For a detailed usage guide for model training, refer to [Training.md](docs/Training.md) and the code under [train](train/).

## Signal Plotting
For a detailed usage guide for signal plotting and comparison, refer to the Jupyter notebook [Signal_Plot_Examples.ipynb](plot_utils/Signal_Plot_Examples.ipynb) and the code under [plot_utils](plot_utils/). The code works with signal files in POD5 format and aligned BAM files with move tables. The Jupyter notebook uses the following data: [plot_files.tar.gz](https://github.com/WGLab/DeepMod2/files/14985308/plot_files.tar.gz).

## Models
The following models for 5mC detection in CpG motif are provided in the repository. Use `--model MODEL_NAME` to specify a model to use. You only need to provide the name of the model, not the path to it. Each model is compatible with a different Dorado or Guppy basecalling model version.
|Model Architecture|DeepMod2 Model Name|Flowcell<BR>(Sampling Rate)| Compatible Dorado/Guppy Basecalling Model|
|-|-|-|-|
|BiLSTM|**bilstm_r10.4.1_5khz_v4.3**|R10.4.1 (5kHz)|**dna_r10.4.1_e8.2_400bps_(fast\|hac\|sup)@v4.3.0**
|Transformer|**transformer_r10.4.1_5khz_v4.3**|R10.4.1 (5kHz)|**dna_r10.4.1_e8.2_400bps_(fast\|hac\|sup)@v4.3.0**
|BiLSTM|**bilstm_r10.4.1_4khz_v4.1**|R10.4.1 (4kHz)|**dna_r10.4.1_e8.2_400bps_(fast\|hac\|sup)@v4.1.0** in Dorado<BR>**dna_r10.4.1_e8.2_400bps_(fast\|hac\|sup).cfg** in Guppy 6.5.7|
|Transformer|**transformer_r10.4.1_4khz_v4.1**|R10.4.1 (4kHz)|**dna_r10.4.1_e8.2_400bps_(fast\|hac\|sup)@v4.1.0** in Dorado<BR>**dna_r10.4.1_e8.2_400bps_(fast\|hac\|sup).cfg** in Guppy 6.5.7|
|BiLSTM|**bilstm_r10.4.1_4khz_v3.5**<BR>(Published in DeepMod2 paper)|R10.4.1 (4kHz)|**dna_r10.4.1_e8.2_400bps_(fast\|hac\|sup)@v3.5.2** in Dorado<BR>**dna_r10.4.1_e8.2_400bps_fast\|hac\|sup).cfg** in Guppy 6.3.8|
|Transformer|**transformer_r10.4.1_4khz_v3.5**<BR>(Published in DeepMod2 paper)|R10.4.1 (4kHz)|**dna_r10.4.1_e8.2_400bps_(fast\|hac\|sup)@v3.5.2** in Dorado<BR>**dna_r10.4.1_e8.2_400bps_fast\|hac\|sup).cfg** in Guppy 6.3.8|
|BiLSTM|**bilstm_r9.4.1**<BR>(Published in DeepMod2 paper)|R9.4.1 (4kHz)|**dna_r9.4.1_e8_(hac\|sup)@v3.3** in Dorado<BR>**dna_r9.4.1_450bps_(hac\|sup).cfg** in Guppy 6.3.8 and 6.5.7|
|Transformer|**transformer_r9.4.1**<BR>(Published in DeepMod2 paper)|R9.4.1 (4kHz)|**dna_r9.4.1_e8_(hac\|sup)@v3.3** in Dorado<BR>**dna_r9.4.1_450bps_(hac\|sup).cfg** in Guppy 6.3.8 and 6.5.7|

## Examples

Please refer to [Example](https://github.com/WGLab/DeepMod2/blob/main/docs/Example.md) for a complete tutorial on how to run DeepMod2 under various scenarios. A test dataset consisting of a small example of ~60 reads in both POD5 and FAST5 format as well as the expected results in this release:https://github.com/WGLab/DeepMod2/releases/tag/v0.3.0.
