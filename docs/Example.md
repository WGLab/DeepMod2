# DeepMod2 Run Example

This example shows how to use DeepMod2 to prediction 5mC methylation from FAST5 files.


# Prepare directories
```
INPUT_DIR=data
BASECALL_DIR=basecall
PATH_TO_GUPPY=guppy
OUTPUT_DIR=mod

mkdir -p ${INPUT_DIR}/raw_fast5
mkdir -p ${BASECALL_DIR}
mkdir -p ${PATH_TO_GUPPY}
```
# Download Software Packges
```
git clone https://github.com/WGLab/DeepMod2.git ${INPUT_DIR}/DeepMod2
conda env create -f ${INPUT_DIR}/DeepMod2/environment.yml
conda activate deepmod2
conda install samtools -y
pip install awscli
wget -qO- https://cdn.oxfordnanoportal.com/software/analysis/ont-guppy_6.1.2_linux64.tar.gz| tar xzf - -C ${PATH_TO_GUPPY}
```

# Download FAST5 data and reference genome
```
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz -O -| gunzip -c > ${INPUT_DIR}/GRCh38.fa
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.fai -O ${INPUT_DIR}/GRCh38.fa.fai

aws s3 --no-sign-request cp  s3://nanopore-human-wgs/rel6_Fast5/UCSC/FAB43577-3574887596_Multi/DEAMERNANOPORE_20161117_FNFAB43577_MN16450_sequencing_run_MA_821_R9_4_NA12878_11_17_16_0.fast5 ./${INPUT_DIR}/raw_fast5
```


# Basecall FAST5 files and extract BAM file
```
${PATH_TO_GUPPY}/ont-guppy/bin/guppy_basecaller  -i ${INPUT_DIR}/raw_fast5 -s ${BASECALL_DIR} -c dna_r9.4.1_450bps_sup.cfg \
 --device auto --fast5_out --bam_out  --align_type full \
 --align_ref ${INPUT_DIR}/GRCh38.fa

ls -1 ${BASECALL_DIR}/pass/*bam|samtools merge ${BASECALL_DIR}/sample.bam -b - --write-index
```

# Methylation Calling from Guppy basecalled FAST5 files

## Run DeepMod2 on Guppy FAST5 files
Use `deepmod2 detect-guppy` to predict methylation calls from Guppy basecalled FAST5 files. We also need to provide a BAM file containing alignments for reads in the FAST5 files as well as the reference genome used for alignment.
```
python ${INPUT_DIR}/DeepMod2/deepmod2 detect-guppy --bam ${BASECALL_DIR}/sample.bam \
 --fast5 ${BASECALL_DIR}/workspace/ --ref ${INPUT_DIR}/GRCh38.fa --output ${OUTPUT_DIR}
```

## Results

DeepMod2 creates two output files for both per-read and per-site prediction.

### Per-read prediction
Per-read prediction file predicts 5mC methylation for each reference CpG site that a read overlaps.
 
```head -5 ${OUTPUT_DIR}/output.per_read```

|read_name|chromosome|position|read_position|strand|methylation_score|methylation_prediction|
|---------|----------|--------|-------------|------|-----------------|----------------------|
|004ad57f-67a8-4b91-81eb-0adae44ffb3d|chr10|80894048|5066|-|0.6070|1|
|004ad57f-67a8-4b91-81eb-0adae44ffb3d|chr10|80894104|5010|-|0.5816|1|
|004ad57f-67a8-4b91-81eb-0adae44ffb3d|chr10|80894196|4928|-|0.5416|1|
|004ad57f-67a8-4b91-81eb-0adae44ffb3d|chr10|80894591|4532|-|0.3594|0|

Per-read output has 7 columns:
 - `read_name` Read name
 - `chromosome` Reference chromosome
 - `position` Position on the reference chromosome for the CpG site (1-based)
 - `read_position` Position on the read for the CpG site (1-based)
 - `strand` Reference strand the read is mapped to at the CpG site
 - `methylation_score` Probability that the CpG site is methylated, calculated by deep learning model
 - `methylation_prediction` 0 or 1 prediction labels for whether the CpG site is methylated (1) or unmethylated (0), using 0.5 probability score threshold
 
 
### Per-site prediction
Per-site prediction file predicts 5mC methylation for each reference CpG site by collecting methylation status for all the overlapping reads.

```head -5 ${OUTPUT_DIR}/output.per_site```

|chromosome|position|strand|total_coverage|methylation_coverage|methylation_percentage|methylation_prediction|
|----------|--------|------|--------------|--------------------|----------------------|----------------------|
|chr1|3054056|+|1|1|100|1|
|chr1|3054082|+|1|0|0|0|
|chr1|3054134|+|1|0|0|0|
|chr1|3054169|+|1|1|100|1|

Per-site output has 7 columns:
 - `chromosome` Reference chromosome
 - `position` Reference position for CpG site
 - `strand` Reference strand the read is mapped to
 - `total_coverage` Total number of reads aligned to the CpG site
 - `methylation_coverage` Total number of reads supporting methylation at the CpG site
 - `methylation_percentage` Percentage of reads supporting methylation at the CpG site (100\*methylation_coverage/total_coverage)
 - `methylation_prediction` 0 or 1 prediction labels for whether the CpG site is methylated (1) or unmethylated (0), using 0.5 methylation percentage threshold
 
 
# Methylation Calling from Tombo resquiggled FAST5 files

## Tombo Resquiggle

First split multi-fast5 to single-fast5 format.

```
TOMBO_DIR=tombo_fast5
TOMBO_OUTPUT_DIR=tombo_mod
mkdir -p ${TOMBO_DIR}

conda install -c bioconda ont-tombo

multi_to_single_fast5 --input_path ${BASECALL_DIR}/workspace/ --save_path ${TOMBO_DIR}
```

Run tombo resquiggle algorithm with `--include-event-stdev` option enabled to include standard deviation for each event.

```
tombo resquiggle ${TOMBO_DIR} ${INPUT_DIR}/GRCh38.fa --dna --overwrite --include-event-stdev  --ignore-read-locks
```

## Run DeepMod2 on Tombo resquiggled FAST5 files
Use `deepmod2 detect-tombo` to predict methylation calls from Tombo resquiggled FAST5 files.

```
python ${INPUT_DIR}/DeepMod2/deepmod2 detect-tombo --fast5 ${TOMBO_DIR} --output ${TOMBO_OUTPUT_DIR}
```

## Results
DeepMod2 creates two output files for both per-read and per-site prediction, similar to methylation calling from Guppy FAST5 files. The results have same format as shown above.

### Per-read prediction
Per-read prediction file predicts 5mC methylation for each reference CpG site that a read overlaps. For `detect-tombo` we do not produce read positions for CpG site since Tombo aligns signals instead of nucleotide sequences.

```
head -5 ${TOMBO_OUTPUT_DIR}/output.per_read
```
|read_name|chromosome|position|read_position|strand|methylation_score|methylation_prediction|
|---------|----------|--------|-------------|------|-----------------|----------------------|
|fe768e41-7160-4d81-9ab9-7f6b2a625c33|chr4|120038475|N/A|-|0.9061|1|
|c0ef93b0-9e00-4cbd-b318-76cf03d5b0c1|chr13|101911288|N/A|-|0.1603|0|
|c0ef93b0-9e00-4cbd-b318-76cf03d5b0c1|chr13|101911276|N/A|-|0.1011|0|
|c0ef93b0-9e00-4cbd-b318-76cf03d5b0c1|chr13|101911125|N/A|-|0.0111|0|


### Per-site prediction
Per-site prediction file predicts 5mC methylation for each reference CpG site by collecting methylation status for all the overlapping reads.

```head -5 ${TOMBO_OUTPUT_DIR}/output.per_site```

|chromosome|position|strand|total_coverage|methylation_coverage|methylation_percentage|methylation_prediction|
|----------|--------|------|--------------|--------------------|----------------------|----------------------|
|chr1|3054038|+|1|1|100|1|
|chr1|3054056|+|1|0|0|0|
|chr1|3054082|+|1|1|100|1|
|chr1|3054134|+|1|0|0|0|
