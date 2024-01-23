# DeepMod2 Run Example

This example shows how to use DeepMod2 to prediction 5mC methylation from FAST5 files.


# 1. Prepare directories
```
INPUT_DIR=data
BASECALL_DIR=basecall
PATH_TO_GUPPY=guppy
OUTPUT_DIR=mod

mkdir -p ${INPUT_DIR}/raw_fast5
mkdir -p ${INPUT_DIR}/raw_pod5
mkdir -p ${BASECALL_DIR}
mkdir -p ${PATH_TO_GUPPY}
```
# 2. Download Software Packges
```
git clone https://github.com/WGLab/DeepMod2.git ${INPUT_DIR}/DeepMod2
conda env create -f ${INPUT_DIR}/DeepMod2/environment.yml
conda activate deepmod2
conda install samtools minimap2 -y
pip install awscli
wget -qO- https://cdn.oxfordnanoportal.com/software/analysis/ont-guppy_6.4.6_linux64.tar.gz| tar xzf - -C ${PATH_TO_GUPPY}
```

# 3. Download Nanopore data and reference genome
```
# Download reference genome
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz -O -| gunzip -c > ${INPUT_DIR}/GRCh38.fa
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.fai -O ${INPUT_DIR}/GRCh38.fa.fai

# Download POD5 files
aws s3 cp --no-sign-request s3://ont-open-data/giab_2023.05/flowcells/hg002/20230424_1302_3H_PAO89685_2264ba8c/pod5_pass/PAO89685_pass__2264ba8c_afee3a87_1243.pod5 ./${INPUT_DIR}/raw_pod5
```

# 4. DeepMod2 Methylation Calling from Guppy basecalled FAST5

Download Kit 14 (R10.4.1 flowcell) FAST5 files:
```
aws s3 cp --no-sign-request s3://ont-open-data/giab_lsk114_2022.12/flowcells/hg002/20221109_1654_5A_PAG65784_f306681d/fast5_pass/PAG65784_pass_f306681d_16a70748_982.fast5 ./${INPUT_DIR}/raw_fast5
```

Now, we will use Guppy to basecall our FAST5 files, and save the basecalled sequences and move tables in BAM format. We have two options here, we can either provide a reference genome to Guppy to give an aligned BAM file, or get unaligned BAM files from Guppy.

## 4.1 Use Guppy basecalling without alignment to reference genome
We will run Guppy with `--bam_out --moves_out` parameters to get move tables in unaligned BAM file. The BAM files will be under two folders: ${BASECALL_DIR}/fast5_data/pass and ${BASECALL_DIR}/fast5_data/fail. We will use all the reads but you can choose to use just the passed ones.

```
# Run Guppy
${PATH_TO_GUPPY}/ont-guppy/bin/guppy_basecaller  -i ${INPUT_DIR}/raw_fast5 -s ${BASECALL_DIR}/fast5_data -c dna_r10.4.1_e8.2_400bps_hac_prom.cfg \
 --device auto --bam_out --moves_out 
```

We now have two options for how to run DeepMod2: a) run DeepMod2 on unaligned BAM file and get methylation predictions for CpG motifs in the reads. b) Align the reads to reference genome, and run DeepMod2 on aligned reads to get methylation predictions for CpG motifs in both the reference genome and the reads.

### 4.1.1 DeepMod2 Methylation calling from aligned reads
First we will carry out alignment of unaligned BAM files using samtools fastq module and minimap2. You can alternatively use [dorardo aligner wrapper](https://github.com/nanoporetech/dorado#alignment) of minimap2 to align reads from unaligned BAM files.
```
find ${BASECALL_DIR}/fast5_data/ \( -path "*/pass/*" -o -path "*/fail/*" \) -type f -name "*.bam"|samtools cat -b -|samtools fastq - -T mv,ts | minimap2 -ax map-ont ${INPUT_DIR}/GRCh38.fa - -y|samtools view -Shu |samtools sort -O BAM  -o ${BASECALL_DIR}/fast5_data/merged.aligned.bam --write-index
```

We will provide the original FAST5 file to DeepMod2 as well as the aligned BAM file as inputs, and run `detect-guppy` module of DeepMod2 by specifying that the signal input file is in FAST5 format using `--file_type fast5` parameter. Running this command should take 1-2mins.
```
python ${INPUT_DIR}/DeepMod2/deepmod2 detect-guppy --file_type fast5 --bam  ${BASECALL_DIR}/fast5_data/merged.aligned.bam --input ${INPUT_DIR}/raw_fast5 --output ${OUTPUT_DIR}/mod_from_fast5_aligned_BAM  --ref ${INPUT_DIR}/GRCh38.fa
```

Afterwards, the folder ` ${OUTPUT_DIR}/mod_from_fast5_aligned_BAM` will have three files:
- `args`: shows the parameters and arguments used for DeepMod2 run.
- `output.bam`: BAM file with methylation tags MM and ML added.
- `output.per_read`: Per-read prediction file, a tab-separated text file with methylation calls for each CpG motids per read.

Per-read prediction file will look like this:

|read_name|chromosome|position|read_position|strand|methylation_score|mean_read_qscore|read_length|
|---------|----------|--------|-------------|------|-----------------|----------------|-----------|
|0008efa2-3cfe-4ae1-900f-044dd08a656c|chr5|175935915|95|+|0.3828|12.55|287|
|0009e497-f97a-45ed-9b5f-8b4deae5249c|chr1|41164199|66|-|0.3958|12.37|6029|
|0009e497-f97a-45ed-9b5f-8b4deae5249c|chr1|41164159|104|-|0.2506|12.37|6029|
|0009e497-f97a-45ed-9b5f-8b4deae5249c|chr1|41163812|452|-|0.9888|12.37|6029|

In order to visually inspect methylation tagged BAM file in IGV, we will sort and index it first:
```
samtools sort ${OUTPUT_DIR}/mod_from_fast5_aligned_BAM/output.bam -o ${OUTPUT_DIR}/mod_from_fast5_aligned_BAM/output.sorted.bam --write-index
```

Then, we will open the file ` ${OUTPUT_DIR}/mod_from_fast5_aligned_BAM/output.sorted.bam ` in IGV, select "Color alignments by" and then "base modification (5mC)", and navigate to chr1:817,517-817,556:

![image](https://github.com/WGLab/DeepMod2/assets/35819083/640ed0df-76a8-4b31-be34-f66696c71f24)

We can merge per-read prediction from multiple reads into per-site predictions for each stranded CpG site in reference genome using DeepMod2's `merge` module:
```
python ${INPUT_DIR}/DeepMod2/deepmod2  --input ${OUTPUT_DIR}/mod_from_fast5_aligned_BAM/output.per_read --output ${OUTPUT_DIR}/mod_from_fast5_aligned_BAM/
```
This will create a file: ${OUTPUT_DIR}/mod_from_fast5_aligned_BAM/output.per_site that will look like this:

|chromosome|position|strand|total_coverage|methylation_coverage|methylation_percentage|mean_methylation_probability|
|----------|--------|------|--------------|--------------------|----------------------|----------------------|
|chr1|41164199|-|1|0|0.0000|0.3958|
|chr1|41164159|-|1|0|0.0000|0.2506|
|chr1|41163812|-|1|1|1.0000|0.9888|
|chr1|41163555|-|1|0|0.0000|0.3825|

### 4.1.2 DeepMod2 Methylation calling from unaligned reads
We will provide the original FAST5 file to DeepMod2 as well as the unaligned BAM file as inputs. First we will combine the BAM files into a single BAM file:
```
# Merge unaligned BAM files into one BAM file
find ${BASECALL_DIR}/fast5_data/ \( -path "*/pass/*" -o -path "*/fail/*" \) -type f -name "*.bam"|samtools cat -b - -o ${BASECALL_DIR}/fast5_data/merged.unaligned.bam
```

Then we will run `detect-guppy` module of DeepMod2 by specifying that the signal input file is in FAST5 format using `--file_type fast5` parameter. Running this command which should take 1-2mins.

```
python ${INPUT_DIR}/DeepMod2/deepmod2 detect-guppy --file_type fast5 --bam  ${BASECALL_DIR}/fast5_data/merged.unaligned.bam --input ${INPUT_DIR}/raw_fast5 --output ${OUTPUT_DIR}/mod_from_fast5_unaligned_BAM 
```

Afterwards, the folder ` ${OUTPUT_DIR}/mod_from_fast5_unaligned_BAM` will have three files:
- `args`: shows the parameters and arguments used for DeepMod2 run.
- `output.bam`: BAM file with methylation tags MM and ML added.
- `output.per_read`: Per-read prediction file, a tab-separated text file with methylation calls for each CpG motids per read.

Per-read prediction file will look like this:

|read_name|chromosome|position|read_position|strand|methylation_score|mean_read_qscore|read_length|
|---------|----------|--------|-------------|------|-----------------|----------------|-----------|
|0008efa2-3cfe-4ae1-900f-044dd08a656c|NA|NA|95|+|0.3828|12.55|287|
|0009e497-f97a-45ed-9b5f-8b4deae5249c|NA|NA|66|+|0.3958|12.37|6029|
|0009e497-f97a-45ed-9b5f-8b4deae5249c|NA|NA|452|+|0.9888|12.37|6029|
|0009e497-f97a-45ed-9b5f-8b4deae5249c|NA|NA|720|+|0.3825|12.37|6029|
