# DeepMod2 Run Example

This example shows how to use DeepMod2 to prediction 5mC methylation from FAST5 files.

- [Methylation Calling from POD5 files with Dorado basecalling](Example.md#methylation-calling-from-pod5-files-with-dorado-basecalling)
   - [Reference Anchored Methylation Calling](Example.md#reference-anchored-methylation-calling)
     - [Dorado Basecalling and Read Alignment to Reference Genome](Example.md#dorado-basecalling-and-read-alignment-to-reference-genome)
     - [(Optional) Read Phasing for diploid genomes](Example.md#optional-read-phasing-for-diploid-genomes)
     - [Methylation Calling with DeepMod2](Example.md#methylation-calling-with-deepmod2)
     - [Visualizing DeepMod2 Methylation in IGV](Example.md#visualizing-deepmod2-methylation-in-igv)
   - Reference free methylation calling
     - Dorado Basecalling
     - Methylation Calling with DeepMod2
     - Optional Read Alignment to Reference Genome and Per-site frequency calculation with modkit


# Methylation Calling from POD5 files with Dorado basecalling
**Prepare Directories** 
```
INPUT_DIR=data
OUTPUT_DIR=mod

mkdir -p ${INPUT_DIR}
mkdir -p ${OUTPUT_DIR}
```
**Download Software Packges**
```
# Install DeepMod2
git clone https://github.com/WGLab/DeepMod2.git ${INPUT_DIR}/DeepMod2
conda env create -f ${INPUT_DIR}/DeepMod2/environment.yml
conda activate deepmod2
conda install samtools minimap2 bedtools -y

# Download Dorado Basecaller and model
wget -qO- https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.5.3-linux-x64.tar.gz | tar xzf - -C ${INPUT_DIR}
${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado download --model  dna_r10.4.1_e8.2_400bps_hac@v4.3.0 --directory ${INPUT_DIR}/dorado-0.5.3-linux-x64/models/
```

**Download Nanopore data and reference genome**
```
# Download reference genome
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz -O -| gunzip -c > ${INPUT_DIR}/GRCh38.fa
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.fai -O ${INPUT_DIR}/GRCh38.fa.fai

# Download POD5 files
mkdir -p ${INPUT_DIR}/nanopore_raw_data

wget -qO- URL| tar xzf - -C ${INPUT_DIR}/nanopore_raw_data
```

## Reference Anchored Methylation Calling
In order to perform reference anchored methylation calling, we will provide an aligned BAM to DeepMod2. In this case, DeepMod2 will detect 5mC in all CpG motifs found on the read, as wel as any bases of the read that map to a reference CpG site. Finally, it will combine per-read predictions for a given reference CpG site into a per-site methylation frequency. Any unaligned reads or unaligned segments of the reads will also be analyzed for 5mC and reported in per-read output and BAM file, but would not be used in per-site frequency calculation. 

### Dorado Basecalling and Read Alignment to Reference Genome
First we will perform basecalling of our nanopore signal file using Dorado basecaller. It is possible to align the reads during basecalling or align the reads after basecalling. Both options are shown below. Since we need move table for our basecalled DNA sequences, we will use `--emit-moves` while running Dorado, which will produce an aligned (Option A) or unaligned BAM file (Option B).

**Option A: Perform Read Alignment during Bascalling with Dorado**

Dorado has the option to perform read alignment using minimap2 during basecalling if a reference FASTA file is provided as `--reference` option. This can be be helpful in reducing the number of steps needed to run.

```
${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado basecaller --emit-moves --recursive --reference ${INPUT_DIR}/GRCh38.fa ${INPUT_DIR}/dorado-0.5.3-linux-x64/models/dna_r10.4.1_e8.2_400bps_hac@v4.3.0  ${INPUT_DIR}/nanopore_raw_data > ${OUTPUT_DIR}/aligned.bam
```

This will produce an aligned BAM file named `aligned.bam` under the `$OUTPUT_DIR` folder.

**Option B: Perform Read Alignment after Bascalling with Dorado**

It is possible to run Dorado basecaller without performing alignment. This can be helpful in speeding up basecalling process that requires the use of a GPU instance which can be expensive. It also allows you more flexibility in terms of how you want to perform alignment, with specific minimap2 parameters.

#### Basecalling with Dorado

```
# Perform basecalling
${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado basecaller --emit-moves --recursive ${INPUT_DIR}/dorado-0.5.3-linux-x64/models/dna_r10.4.1_e8.2_400bps_hac@v4.3.0  ${INPUT_DIR}/nanopore_raw_data > ${OUTPUT_DIR}/basecalled.bam
```

This will produce an unaligned BAM file named `basecalled.bam` under the `$OUTPUT_DIR` folder. 

#### Alignment with minimap2
We will convert this BAM file into FASTQ format while keeping all the tags and pipe into minimap2 for alignment.

```
# Align using minimap2 while copying move table information
samtools fastq  ${OUTPUT_DIR}/basecalled.bam -T "*"|minimap2 -ax map-ont ${INPUT_DIR}/GRCh38.fa - -y|samtools view -o ${OUTPUT_DIR}/aligned.bam
```

This will produce an aligned BAM file named `aligned.bam` under the `$OUTPUT_DIR` folder.

### (Optional) Read Phasing for diploid genomes
You can optionally use SNP calling and haplotyping tool such as NanoCaller or Clair3 to phase the BAM file into parental haplotypes. The phased BAM file can be provided as input to DeepMod2 instead of `${OUTPUT_DIR}/aligned.bam` to get haplotype specific methylation calls.

```
#install NanoCaller
conda install -c bioconda NanoCaller

#sort and index the BAM file
samtools sort ${OUTPUT_DIR}/aligned.bam -o ${OUTPUT_DIR}/aligned.sorted.bam
samtools index ${OUTPUT_DIR}/aligned.sorted.bam

#Run NanoCaller to phase the reads
NanoCaller --bam ${OUTPUT_DIR}/aligned.sorted.bam --ref ${INPUT_DIR}/GRCh38.fa --mode snps --phase --output ${OUTPUT_DIR}/nanocaller --wgs_contigs chr1-22XY --cpu 8 

 # Merge phased reads into a single BAM file
find ${OUTPUT_DIR}/nanocaller/intermediate_phase_files -type f -name '*bam'|samtools cat -b - -o ${OUTPUT_DIR}/phased.bam

```

### Methylation Calling with DeepMod2 
Now we will run DeepMod2's `detect` module using `bilstm_r10.4.1_5khz_v4.3` model and use the aligned BAM file and Nanopore signal files as input. Since we want to perform reference anchored methylation calling, we will provide the reference genome FASTA file as input as well. We will use the phased BAM file from the previous step, but you can also use `${OUTPUT_DIR}/aligned.bam` BAM file if you do not want to get haplotype specific methylation calls.

```
# Run DeepMod2
BAM_INPUT=${OUTPUT_DIR}/phased.bam # Use ${OUTPUT_DIR}/aligned.bam if you did not use NanoCaller to phase the reads
python ${INPUT_DIR}/DeepMod2/deepmod2 detect --model bilstm_r10.4.1_5khz_v4.3 --file_type pod5 --bam  ${OUTPUT_DIR}/phased.bam --input  ${INPUT_DIR}/nanopore_raw_data --output ${OUTPUT_DIR}/deepmod2/ --ref ${INPUT_DIR}/GRCh38.fa --threads 8
```
The output folder of DeepMod2 `${OUTPUT_DIR}/deepmod2/` will contain the following files:
```
args -> Shows the arguments and command use to run DeepMod2
output.bam -> Unsorted methylation tagged BAM file
output.per_read -> Per-read methylation calls in sorted BED file
output.per_site -> Per-site methylation calls for +- strands separately in sorted BED file.
output.per_site.aggregated -> Per-site methylation calls for with counts for +- strands combined. 
```

**Per-read Output**
We will inspect contents of the per-read output file using `head ${OUTPUT_DIR}/deepmod2/output.per_read`:


|read_name|chromosome|ref_position_before|ref_position|read_position|strand|methylation_score|mean_read_qscore|read_length|read_phase|ref_cpg|
|-|-|-|-|-|-|-|-|-|-|-|
|160f871b-f4c3-40de-a160-383fcd5033e7|chr11|2733569|2733570|35|-|0.0036|18.05|48187|1|TRUE|
|160f871b-f4c3-40de-a160-383fcd5033e7|chr11|2733457|2733458|146|-|0.9846|18.05|48187|1|TRUE|
|160f871b-f4c3-40de-a160-383fcd5033e7|chr11|2733439|2733440|164|-|0.0048|18.05|48187|1|TRUE|
|160f871b-f4c3-40de-a160-383fcd5033e7|chr11|2733362|2733363|242|-|0.2352|18.05|48187|1|TRUE|
|160f871b-f4c3-40de-a160-383fcd5033e7|chr11|2733356|2733357|248|-|0.841|18.05|48187|1|TRUE|
|160f871b-f4c3-40de-a160-383fcd5033e7|chr11|2733351|2733352|253|-|0.9893|18.05|48187|1|FALSE|
|160f871b-f4c3-40de-a160-383fcd5033e7|chr11|2733341|2733342|263|-|0.012|18.05|48187|1|TRUE|
|160f871b-f4c3-40de-a160-383fcd5033e7|chr11|2733155|2733156|449|-|0.9858|18.05|48187|1|TRUE|
|160f871b-f4c3-40de-a160-383fcd5033e7|chr11|2733143|2733144|461|-|0.966|18.05|48187|1|TRUE|


**Per-Site Output**

We will use `bedtools intersect` to inspect per-site methylation frequencies in chr11:2699000-2702000 imprinting control region from the per-site output file with stranded CpG counts:

 ```
 printf 'chr11\t2699000\t2702000'|bedtools intersect -header -a ${OUTPUT_DIR}/deepmod2/output.per_site -b -|head
 ```

|#chromosome|position_before|position|strand|ref_cpg|coverage|mod_coverage|unmod_coverage|mod_percentage|coverage_phase1|mod_coverage_phase1|unmod_coverage_phase1|mod_percentage_phase1|coverage_phase2|mod_coverage_phase2|unmod_coverage_phase2|mod_percentage_phase2|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|chr11|2699031|2699032|+|TRUE|16|8|8|0.5|8|0|8|0|8|8|0|1|
|chr11|2699032|2699033|-|TRUE|11|7|4|0.6364|4|0|4|0|7|7|0|1|
|chr11|2699037|2699038|+|TRUE|16|8|8|0.5|8|0|8|0|8|8|0|1|
|chr11|2699038|2699039|-|TRUE|11|7|4|0.6364|4|0|4|0|7|7|0|1|
|chr11|2699048|2699049|+|TRUE|16|6|10|0.375|8|0|8|0|8|6|2|0.75|
|chr11|2699049|2699050|-|TRUE|11|5|6|0.4545|4|0|4|0|7|5|2|0.7143|
|chr11|2699099|2699100|+|TRUE|15|7|8|0.4667|8|0|8|0|7|7|0|1|
|chr11|2699100|2699101|-|TRUE|11|7|4|0.6364|4|0|4|0|7|7|0|1|
|chr11|2699101|2699102|+|TRUE|16|7|9|0.4375|8|0|8|0|8|7|1|0.875|


**Aggregated Per-Site Output**

We will use `bedtools intersect` to inspect per-site methylation frequencies in chr11:2699000-2702000 imprinting control region from the per-site output file with aggregated CpG counts over +- strands:

 ```
 printf 'chr11\t2699000\t2702000'|bedtools intersect -header -a ${OUTPUT_DIR}/deepmod2/output.per_site.aggregated -b -|head
 ```

|#chromosome|position_before|position|ref_cpg|coverage|mod_coverage|unmod_coverage|mod_percentage|coverage_phase1|mod_coverage_phase1|unmod_coverage_phase1|mod_percentage_phase1|coverage_phase2|mod_coverage_phase2|unmod_coverage_phase2|mod_percentage_phase2|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|chr11|2699031|2699033|TRUE|27|15|12|0.5556|12|0|12|0|15|15|0|1|
|chr11|2699037|2699039|TRUE|27|15|12|0.5556|12|0|12|0|15|15|0|1|
|chr11|2699048|2699050|TRUE|27|11|16|0.4074|12|0|12|0|15|11|4|0.7333|
|chr11|2699099|2699101|TRUE|26|14|12|0.5385|12|0|12|0|14|14|0|1|
|chr11|2699101|2699103|TRUE|27|14|13|0.5185|12|0|12|0|15|14|1|0.9333|
|chr11|2699115|2699117|TRUE|27|14|13|0.5185|12|0|12|0|15|14|1|0.9333|
|chr11|2699120|2699122|TRUE|27|15|12|0.5556|12|0|12|0|15|15|0|1|
|chr11|2699180|2699182|TRUE|27|15|12|0.5556|12|0|12|0|15|15|0|1|
|chr11|2699187|2699189|TRUE|27|15|12|0.5556|12|0|12|0|15|15|0|1|

These results show that phase 1 is completely unmodified (column mod_percentage_phase1) whereas phase 2 is nearly completely modified (mod_percentage_phase2), which is what we expect for this imprinted region.


### Visualizing DeepMod2 Methylation in IGV
Since the methylation tagged BAM file produced by DeepMod2 is not sorted, we will first sort and index it:

```
samtools sort  ${OUTPUT_DIR}/deepmod2/output.bam -o ${OUTPUT_DIR}/deepmod2/output.sorted.bam --write-index
```

Open the BAM file `${OUTPUT_DIR}/deepmod2/output.sorted.bam` in IGV, select `Color alignments by base modificaition (5mC)`. If you used phased BAM file for methylation, you can select `Group alignments  by phase` to separate reads by haplotype. Go to the region `chr11:2699000-2702000` to see the following methylation tags:

![image](https://github.com/WGLab/DeepMod2/assets/35819083/b7e87a6c-9dda-4b31-be0e-93c13ecec1fb)
