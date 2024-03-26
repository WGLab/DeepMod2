
# Model Training with DeepMod2

1. [Train model from a sample with only modified bases and a sample with only unmodified (canonical) bases](Training.md#1-train-model-from-a-sample-with-only-modified-bases-and-a-sample-with-only-unmodified-canonical-bases)

2. [Train model from a single sample with mixed labels of modified and unmodified (canonical) bases](Training.md#2-train-model-from-a-single-sample-with-mixed-labels-of-modified-and-unmodified-canonical-bases)

3. [Train model from samples with mixed labels of modified and unmodified (canonical) bases, as well as samples with only modified or unmodified (canonical) bases](Training.md#3-train-model-from-samples-with-mixed-labels-of-modified-and-unmodified-canonical-bases-as-well-as-samples-with-only-modified-or-unmodified-canonical-bases)

# 1. Train model from a sample with only modified bases and a sample with only unmodified (canonical) bases
In this section, we will train a 5mC methylation model from POD5 files of two human datasets, one containing modified bases and another containing unmodified (canocial) bases. This situation is described in the figure below:

![image](https://github.com/WGLab/DeepMod2/assets/35819083/d9d4d7c1-9696-43b8-b0ff-96b43806b02c)

Modified data is contained in `mod.pod5` which has 5mC in all the reads at the CpG positions specified by a file `mod_list`. Unmodified data is contained in `unmod.pod5` which has 5mC in all the reads at the CpG positions specified by a file `unmod_list`. There are two ways to specify which genomic positions to use to generate features: 1) using a list of positions with a modification label (`--pos_list`), 2) using a motif to target all motif matches assuming they are all modifiedon unmodified (`--motif --motif_label`). In this demo we will use the first option. In our example, the coordinate files `mod_list` and `unmod_list` contain the same positions, but this does not have to be the case depending upon your sample. `mod_list` and `unmod_list` contain human reference genome coordinates that tell us which positions to use for creating features for model training, and also tell whether the position is on forward or reverse strand. These file have one position per line, containing four pieces of information separated by whitespace in the following format: 

```contig position strand label```

contig is the reference contig, position is 0-based (assuming the first base of a contig has index 0 instead of 1), strand is + or -, and label is 0 for unmodified and 1 for modified. For example, the first few lines of `mod_list` look like:

```
chr11   2698550 +       1
chr11   2698551 -       1
chr11   2698596 +       1
chr11   2698597 -       1
```

whereas the first few lines of `unmod_list` look like:

```
chr11   2698550 +       0
chr11   2698551 -       0
chr11   2698596 +       0
chr11   2698597 -       0
```

**Important Note:** Specifying positions of interest via a motif can be problematic if not all motifs occurrences are modified or unmodified in a sample, or if the modification label is unknown for certain occurrences of the motif. Using a list of coordinates can circumvent this problem by letting you specify only the most confident labels.

## 1.1 Data Preparation
First we will download the necessary datasets and software, and create directories to do the analysis.

**Prepare Directories**
```
#create directories for downloading and processing data
INPUT_DIR=raw_data # where we will download and process data for feature generation and evaluation
OUTPUT_DIR=training_data # where we will process and store generated features and models
PREDICTION_DIR=deepmod2_output # where we will store DeepMod2 evaluation results
DeepMod2_DIR=${INPUT_DIR}/DeepMod2 # where we will download DeepMod2

mkdir -p ${INPUT_DIR}
mkdir -p ${INPUT_DIR}/bam_files
mkdir -p ${OUTPUT_DIR}
mkdir -p ${PREDICTION_DIR}
```

**Download and install DeepMod2**

If you do not have conda installed, please follow the directions here to install conda first for your system: https://docs.anaconda.com/free/miniconda/#quick-command-line-install. Then, we will install DeepMod2 as follows:

```
#clone github repository of DeepMod2 and install conda environment
git clone https://github.com/WGLab/DeepMod2.git ${DeepMod2_DIR}
conda env create -f ${DeepMod2_DIR}/environment.yml
conda activate deepmod2
conda install samtools -y
```

The above commands also install a generic version of PyTorch library for deep learning. If you want to use GPU for model training, make sure to install CUDA enabled version of Pytorch that is compatible with your GPU driver: https://pytorch.org/get-started/locally/.


**Download sample training datasets and reference genome**

We will download sample training dataset from https://github.com/WGLab/DeepMod2/releases/tag/v0.3.1, as well as GRCh38 human reference genome.
```
#Download sample training datasets
wget -qO- https://github.com/WGLab/DeepMod2/files/14673227/data_download.tar.gz| tar xzf - -C ${INPUT_DIR}

#Download reference genome
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz -O -| gunzip -c > ${INPUT_DIR}/GRCh38.fa
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.fai -O ${INPUT_DIR}/GRCh38.fa.fai
```

**Download Dorado basecaller**

We will download [Dorado](https://github.com/nanoporetech/dorado) basecaller (v0.5.3) and `dna_r10.4.1_e8.2_400bps_hac@v4.3.0` basecalling model. Make sure to select the appropriate basecalling model for your sample depending upon flowcell chemistry. Use the following method to install Dorado if you are using a Linux OS:

```
#Download Dorado (Linux version) if you are using Linux
wget -qO- https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.5.3-linux-x64.tar.gz | tar xzf - -C ${INPUT_DIR}
${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado download --model  dna_r10.4.1_e8.2_400bps_hac@v4.3.0 --directory ${INPUT_DIR}/dorado-0.5.3-linux-x64/models/

DORADO_PATH=${INPUT_DIR}/dorado-0.5.3-linux-x64
```

Otherwise, if you are using macOS, the use following method to install Dorado:
```
#Download Dorado (macOS version) if you are using macOS
 wget https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.5.3-osx-arm64.zip
 unzip dorado-0.5.3-osx-arm64.zip -d ${INPUT_DIR}
 ${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado download --model  dna_r10.4.1_e8.2_400bps_hac@v4.3.0 --directory ${INPUT_DIR}/dorado-0.5.3-osx-arm64/models/

DORADO_PATH=${INPUT_DIR}/dorado-0.5.3-osx-arm64
```

`DORADO_PATH` variable stores the path to Dorado installation.

## 1.2 Dorado basecalling with move tables
In order to generate features for our model, we will first basecall our samples with Dorado using `--emit-moves` to create move tables in addition to basecalls. We will also provide reference genome to Dorado produced aligned BAM file, but you can also skip providing reference genome to Dorado and align yourself later using minimap2.

```
#basecall modified sample
${DORADO_PATH}/bin/dorado basecaller --emit-moves --reference ${INPUT_DIR}/GRCh38.fa ${DORADO_PATH}/models/dna_r10.4.1_e8.2_400bps_hac@v4.3.0  ${INPUT_DIR}/data_download/signal_files/mod.pod5 > ${INPUT_DIR}/bam_files/mod.bam 

#basecall unmodified sample
${DORADO_PATH}/bin/dorado basecaller --emit-moves --reference ${INPUT_DIR}/GRCh38.fa ${DORADO_PATH}/models/dna_r10.4.1_e8.2_400bps_hac@v4.3.0  ${INPUT_DIR}/data_download/signal_files/unmod.pod5 > ${INPUT_DIR}/bam_files/unmod.bam 

#basecall testing sample (which is actually just modified plus unmodified sample)
${DORADO_PATH}/bin/dorado basecaller --emit-moves --recursive --reference ${INPUT_DIR}/GRCh38.fa ${DORADO_PATH}/models/dna_r10.4.1_e8.2_400bps_hac@v4.3.0  ${INPUT_DIR}/data_download/signal_files/  > ${INPUT_DIR}/bam_files/all_reads.bam
```

The above commands will create
-  `${INPUT_DIR}/bam_files/mod.bam` BAM file from basecalling of modified sample `${INPUT_DIR}/data_download/signal_files/mod.pod5`
- `${INPUT_DIR}/bam_files/unmod.bam` BAM file from basecalling of unmodified (canonical) sample `${INPUT_DIR}/data_download/signal_files/unmod.pod5`
- `${INPUT_DIR}/bam_files/all_reads.bam` test BAM file from basecalling of sample `${INPUT_DIR}/data_download/signal_files/` which is just the modified and unmodified sample POD5 files for the purpose of demonstration. We will use this testing sample to run DeepMod2 on the trained model, not for model validation during training.

## 1.3 Generate Training Features
We will use `generate_features.py` module under `train` folder of DeepMod2 repository to generate features for model training. You can find a list of options using `python ${DeepMod2_DIR}/train/generate_features.py --help` command. We will generate features separately for modified and unmodified samples by providing signal POD5 file as `--input`, BAM file as `--bam` and a list of positions with modified/unmodified labels as `--pos_list`. In this case, we will use a window size of 10, which means how many bases before and after each base position of interest (from pos_list) to include in feature generation. At this stage, you do not need to specify a motif to `generate_features.py`, as you are providing a list of coordinates directly. You can use `--chrom  CHROM` to generate features for just a single chromosome and use `--threads NUM_THREADS` to speed up feature generation.

```
#generate features for modified sample
python ${DeepMod2_DIR}/train/generate_features.py --bam ${INPUT_DIR}/bam_files/mod.bam --input  ${INPUT_DIR}/data_download/signal_files/mod.pod5 --ref ${INPUT_DIR}/GRCh38.fa --file_type pod5 --threads 4 --output ${OUTPUT_DIR}/features/mod/ --pos_list  ${INPUT_DIR}/data_download/label_files/mod_list  --window 10

#generate features for unmodified sample
python ${DeepMod2_DIR}/train/generate_features.py --bam ${INPUT_DIR}/bam_files/unmod.bam --input  ${INPUT_DIR}/data_download/signal_files/unmod.pod5 --ref ${INPUT_DIR}/GRCh38.fa --file_type pod5 --threads 4 --output ${OUTPUT_DIR}/features/unmod/ --pos_list  ${INPUT_DIR}/data_download/label_files/unmod_list  --window 10
```

This will allow us to create modified sample features under `${OUTPUT_DIR}/features/mod/` directory, and unmodified sample features under `${OUTPUT_DIR}/features/unmod/`. We will supply these two folders to model training module.

## 1.4 Train DeepMod2 Model
We will train a deep learning model using `train_models.py` module under `train` folder of DeepMod2 repository. You can find a full list of options using `python ${DeepMod2_DIR}/train/train_models.py --help` command. In this demo, we will train a small model:

- using BiLSTM architecture (`--model_type bilstm`)
- model will have 2 recurrent layers (`--num_layers 2`) 
- model will have 32 hidden nodes in LSTM layers (`--dim_feedforward 32`)
- model will have 32 hidden nodes in fully connected classifier layer (`--num_fc 32`)
- trained for 10 epochs (`--epochs 10`)
- trained using a learning rate of 0.01 (`--lr 0.01`)
- trained using L2 regularization of 0.01 (`--l2_coef 0.01`)
- the model will include reference sequence as a features (`--include_ref`)
- the model will be validated using a random 50% validation split of training datasets (`--validation_type split --validation_fraction 0.5`)
- the model training will be reproducible using the same fixed seed (`--seed 0`)

We will provide the modified base features using `--mod_training_dataset` parameter and unmodified (canonical) base features using `--can_training_dataset`. The training module will mix examples of modified and unmodified instances in each batch as opposed to sequentially processing modified and unmodified instances which tends to make the model forget previous training examples.

```
#train model using modified and canonical base samples
python ${DeepMod2_DIR}/train/train_models.py --can_training_dataset ${OUTPUT_DIR}/features/unmod/ --mod_training_dataset ${OUTPUT_DIR}/features/mod/ --validation_type split --validation_fraction 0.5 --model_save_path ${OUTPUT_DIR}/can_mod_bilstm/ --epochs 10 --batch_size 128 --model_type bilstm --num_layers 2 --num_fc 32 --dim_feedforward 32 --lr 0.01 --window 10 --include_ref --l2_coef 0.01 --seed 0
```

This example uses a split of training dataset for validation, but you can generate features for a different genome or sample or chromosome using `generate_features.py`, and provide those features for validation by specifying validation type and a path to the validation dataset as follows `--validation_type dataset --validation_dataset path_to_validation_features`.

Since we fixed a random seed, we should be able to reproduce the following training and validation log:

<details>
  <summary>
    Training Log
  </summary>

```
mixed_training_dataset: None
can_training_dataset: ['training_data/features/unmod/']
mod_training_dataset: ['training_data/features/mod/']
validation_type: split
validation_fraction: 0.5
validation_dataset: None
prefix: model
weights: equal
model_save_path: training_data/can_mod_bilstm/
epochs: 10
batch_size: 128
retrain: None
fc_type: all
model_type: bilstm
window: 10
num_layers: 2
dim_feedforward: 32
num_fc: 32
embedding_dim: 4
embedding_type: one_hot
pe_dim: 16
pe_type: fixed
nhead: 4
include_ref: True
train_w_wo_ref: False
lr: 0.01
l2_coef: 0.01
seed: 0

Starting training.
Number of Modified Instances=3013.0
Number of Un-Modified Instances=2267.0
Positive Label Weight=1.0

BiLSTM(
  (emb): SeqEmbed(
    (seq_emb): RefandReadEmbed(
      (read_emb): OneHotEncode()
      (ref_emb): OneHotEncode()
    )
  )
  (bilstm): LSTM(19, 32, num_layers=2, batch_first=True, bidirectional=True)
  (classifier): ClassifierAll(
    (fc): Linear(in_features=1344, out_features=32, bias=True)
    (out): Linear(in_features=32, out_features=1, bias=True)
  )
)
# Parameters= 81729
....

Epoch 1: #Train=2639  #Test=2641  Time=0.8999
Training  Loss:  Total: 0.6306
Training Accuracy:  Total: 0.6491
Training Precision:  Total: 0.6539
Training Recall:  Total: 0.8181
Training F1:  Total: 0.7268

Testing  Loss:  Total: 0.5711
Testing Accuracy:  Total: 0.7088
Testing Precision:  Total: 0.7281
Testing Recall:  Total: 0.7817
Testing F1:  Total: 0.7539
....

Epoch 2: #Train=2639  #Test=2641  Time=0.1141
Training  Loss:  Total: 0.5449
Training Accuracy:  Total: 0.7325
Training Precision:  Total: 0.7509
Training Recall:  Total: 0.7948
Training F1:  Total: 0.7723

Testing  Loss:  Total: 0.5710
Testing Accuracy:  Total: 0.7308
Testing Precision:  Total: 0.7961
Testing Recall:  Total: 0.7100
Testing F1:  Total: 0.7506
....

Epoch 3: #Train=2639  #Test=2641  Time=0.1146
Training  Loss:  Total: 0.5098
Training Accuracy:  Total: 0.7526
Training Precision:  Total: 0.7830
Training Recall:  Total: 0.7835
Training F1:  Total: 0.7833

Testing  Loss:  Total: 0.5173
Testing Accuracy:  Total: 0.7550
Testing Precision:  Total: 0.8359
Testing Recall:  Total: 0.7100
Testing F1:  Total: 0.7679
....

Epoch 4: #Train=2639  #Test=2641  Time=0.1144
Training  Loss:  Total: 0.4751
Training Accuracy:  Total: 0.7753
Training Precision:  Total: 0.8133
Training Recall:  Total: 0.7869
Training F1:  Total: 0.7999

Testing  Loss:  Total: 0.5071
Testing Accuracy:  Total: 0.7652
Testing Precision:  Total: 0.8015
Testing Recall:  Total: 0.7823
Testing F1:  Total: 0.7918
....

Epoch 5: #Train=2639  #Test=2641  Time=0.1141
Training  Loss:  Total: 0.4646
Training Accuracy:  Total: 0.7783
Training Precision:  Total: 0.8093
Training Recall:  Total: 0.8001
Training F1:  Total: 0.8047

Testing  Loss:  Total: 0.4744
Testing Accuracy:  Total: 0.7758
Testing Precision:  Total: 0.7858
Testing Recall:  Total: 0.8348
Testing F1:  Total: 0.8095
....

Epoch 6: #Train=2639  #Test=2641  Time=0.1153
Training  Loss:  Total: 0.4383
Training Accuracy:  Total: 0.7939
Training Precision:  Total: 0.8254
Training Recall:  Total: 0.8101
Training F1:  Total: 0.8177

Testing  Loss:  Total: 0.4668
Testing Accuracy:  Total: 0.7834
Testing Precision:  Total: 0.7873
Testing Recall:  Total: 0.8500
Testing F1:  Total: 0.8175
....

Epoch 7: #Train=2639  #Test=2641  Time=0.1146
Training  Loss:  Total: 0.4247
Training Accuracy:  Total: 0.7992
Training Precision:  Total: 0.8297
Training Recall:  Total: 0.8154
Training F1:  Total: 0.8225

Testing  Loss:  Total: 0.4533
Testing Accuracy:  Total: 0.7868
Testing Precision:  Total: 0.8109
Testing Recall:  Total: 0.8169
Testing F1:  Total: 0.8139
....

Epoch 8: #Train=2639  #Test=2641  Time=0.1139
Training  Loss:  Total: 0.4237
Training Accuracy:  Total: 0.7995
Training Precision:  Total: 0.8259
Training Recall:  Total: 0.8220
Training F1:  Total: 0.8240

Testing  Loss:  Total: 0.4642
Testing Accuracy:  Total: 0.7834
Testing Precision:  Total: 0.7835
Testing Recall:  Total: 0.8573
Testing F1:  Total: 0.8188
....

Epoch 9: #Train=2639  #Test=2641  Time=0.1143
Training  Loss:  Total: 0.4099
Training Accuracy:  Total: 0.8124
Training Precision:  Total: 0.8386
Training Recall:  Total: 0.8313
Training F1:  Total: 0.8349

Testing  Loss:  Total: 0.4412
Testing Accuracy:  Total: 0.7986
Testing Precision:  Total: 0.8239
Testing Recall:  Total: 0.8228
Testing F1:  Total: 0.8234
....

Epoch 10: #Train=2639  #Test=2641  Time=0.1154
Training  Loss:  Total: 0.4090
Training Accuracy:  Total: 0.8121
Training Precision:  Total: 0.8344
Training Recall:  Total: 0.8367
Training F1:  Total: 0.8355

Testing  Loss:  Total: 0.4728
Testing Accuracy:  Total: 0.7846
Testing Precision:  Total: 0.7849
Testing Recall:  Total: 0.8573
Testing F1:  Total: 0.8195
Time taken=7.4799
```
</details>

At the end of the 10th epoch, we get 78.46% model validation accuracy. The model is saved after each epoch under `${OUTPUT_DIR}/can_mod_bilstm/` directory specified by `--model_save_path`, with each saved model checkpoint having the following naming format: "prefix.epochN.validation_accuracy", which can be see by listing the contents of model_save_path folder using the following command:

```
 ls -1trh ${OUTPUT_DIR}/can_mod_bilstm/
```

<details>
  <summary>
Folder contents:
  </summary>

```
model.cfg
model.epoch1.0.7088
model.epoch2.0.7308
model.epoch3.0.7550
model.epoch4.0.7652
model.epoch5.0.7758
model.epoch6.0.7834
model.epoch7.0.7868
model.epoch8.0.7834
model.epoch9.0.7986
model.epoch10.0.7846
model.log
```
</details>

The folder contains a long file `model.log`, 10 saved model checkpoints `model.epochY.0.XXXX`, and model configuration file `model.cfg`. When we want to use this model, we have to provide a saved checkpoint and the model configuration file to DeepMod2. For now, we will use the last checkpoint file for model inference:

```
model_path=`ls -1tr ${OUTPUT_DIR}/can_mod_bilstm/model.epoch*|tail -1`
```

## 1.5 Test DeepMod2 Model
We will now use the model `model.epoch10.0.7846` and `model.cfg` on the test dataset using DeepMod2's `detect` module. We will provide the model as `--model PATH_TO_MODEL_CONFIGURATION_FILE,PATH_TO_MODEL_CHECKPOINT` where we provide the paths to model configuration file and model checkpoint separated by a comma to `--model` parameter. 

For inference, we need to provide a motif using `--motif` parameter as `--motif CG 0` by specifying the motif and 0-based indices of the bases of interest within the motif separated by whitespace. This parameter tells which base in the motif to call modification on. The default behaviour is to call modifications on all loci of a read that either match the motif, or map to a reference position that matches the motif. You can choose to call modifications only on those read loci that map to a reference position with motif match by using `--reference_motif_only` parameter. You can further narrow down the loci where the modification is called if you are only interested in a select few reference motif positions by specifying a file with a list of coordinates of interest using `--mod_positions` parameter. This file is whitespace separated and has the following format: "contig position strand" on each line, where position should be zero-based and strand should be "+" or "-".

For now, we will just use `--motif CG 0` to call modifications, and provide BAM file, signal files and the reference genome to DeepMod2.

```
# Call modifications on the test dataset
python ${DeepMod2_DIR}/deepmod2 detect --model ${OUTPUT_DIR}/can_mod_bilstm/model.cfg,${model_path} --file_type pod5 --bam ${INPUT_DIR}/bam_files/all_reads.bam --input ${INPUT_DIR}/data_download/signal_files/ --output ${PREDICTION_DIR} --ref ${INPUT_DIR}/GRCh38.fa --threads 4 --motif CG 0
```

The modification calling results will be under `${PREDICTION_DIR}` folder specified by `--output` parameter. For a detailed description of the output files, see [Example.md](https://github.com/WGLab/DeepMod2/blob/main/docs/Example.md#113-methylation-calling-with-deepmod2). The output folder will have the following files:

```
args -> Shows the arguments and command use to run DeepMod2
output.bam -> Unsorted modification tagged BAM file
output.per_read -> Per-read modification calls in sorted BED file
output.per_site -> Per-site modification calls for +- strands separately in sorted BED file.
output.per_site.aggregated -> Per-site modification calls for with counts for +- strands combined. Produces only for CG motif. 
```

We can sort the modification tagged BAM file using samtools to view modification in IGV:

```
samtools sort ${PREDICTION_DIR}/output.bam -o ${PREDICTION_DIR}/sorted.bam --write-index
```

Open the BAM file ${PREDICTION_DIR}/sorted.bam in IGV, select Color alignments by base modification. Go to the region chr11:2699000-2702000 to see the following methylation tags:

<PNG>


-----------------------------------

# 2. Train model from a single sample with mixed labels of modified and unmodified (canonical) bases
In this section, we will train a 5mC methylation model from POD5 files of a single dataset that contains different loci that have modified or unmodified (canonical) bases. Here it is assumed that all reads at a modified locus contain only modified bases, whereas all reads at an unmodified locus contain only unmodified bases. Moreover, it is assumed that each locus appears only once with either modified (1) or unmodified (0) label. This situation is described in the figure below:

![image](https://github.com/WGLab/DeepMod2/assets/35819083/da111807-e660-4c7b-a899-4f3d7c6f09a0)


These genomic loci are contained in `mixed_list` file, with human reference genome coordinates that tell us which positions to use for creating features for model training, and also tell whether the position is on the forward or reverse strand, as well as the modification labels. This file has one position per line, containing four pieces of information separated by whitespace in the following format: 

```contig position strand label```

contig is the reference contig, position is 0-based (assuming the first base of a contig has index 0 instead of 1), strand is + or -, and label is 0 for unmodified and 1 for modified. For example, the first few lines of `mixed_list` look like:

```
chr11   2689255 +       0
chr11   2689799 +       0
chr11   2689800 -       0
chr11   2689940 +       0
chr11   2689941 -       0
chr11   2692201 +       1
chr11   2692210 +       1
```

## 2.1 Data Preparation
First we will download the necessary datasets and software, and create directories to do the analysis.

**Prepare Directories**
```
#create directories for downloading and processing data
INPUT_DIR=raw_data # where we will download and process data for feature generation and evaluation
OUTPUT_DIR=training_data # where we will process and store generated features and models
PREDICTION_DIR=deepmod2_output # where we will store DeepMod2 evaluation results
DeepMod2_DIR=${INPUT_DIR}/DeepMod2 # where we will download DeepMod2

mkdir -p ${INPUT_DIR}
mkdir -p ${INPUT_DIR}/bam_files
mkdir -p ${OUTPUT_DIR}
mkdir -p ${PREDICTION_DIR}
```

**Download and install DeepMod2**

If you do not have conda installed, please follow the directions here to install conda first for your system: https://docs.anaconda.com/free/miniconda/#quick-command-line-install. Then, we will install DeepMod2 as follows:

```
#clone github repository of DeepMod2 and install conda environment
git clone https://github.com/WGLab/DeepMod2.git ${DeepMod2_DIR}
conda env create -f ${DeepMod2_DIR}/environment.yml
conda activate deepmod2
conda install samtools -y

#switch to "general_motif" branch that currently has this experimental functionality
git -C ${DeepMod2_DIR} checkout general_motif
```

The above commands also install a generic version of PyTorch library for deep learning. If you want to use GPU for model training, make sure to install CUDA enabled version of Pytorch that is compatible with your GPU driver: https://pytorch.org/get-started/locally/.


**Download sample training datasets and reference genome**

We will download sample training dataset from https://github.com/WGLab/DeepMod2/releases/tag/v0.3.1, as well as GRCh38 human reference genome.
```
#Download sample training datasets
wget -qO- https://github.com/WGLab/DeepMod2/files/14673227/data_download.tar.gz| tar xzf - -C ${INPUT_DIR}

#Download reference genome
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.gz -O -| gunzip -c > ${INPUT_DIR}/GRCh38.fa
wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GRCh38_major_release_seqs_for_alignment_pipelines/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna.fai -O ${INPUT_DIR}/GRCh38.fa.fai
```

**Download Dorado basecaller**

We will download [Dorado](https://github.com/nanoporetech/dorado) basecaller (v0.5.3) and `dna_r10.4.1_e8.2_400bps_hac@v4.3.0` basecalling model. Make sure to select the appropriate basecalling model for your sample depending upon flowcell chemistry. Use the following method to install Dorado if you are using a Linux OS:

```
#Download Dorado (Linux version) if you are using Linux
wget -qO- https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.5.3-linux-x64.tar.gz | tar xzf - -C ${INPUT_DIR}
${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado download --model  dna_r10.4.1_e8.2_400bps_hac@v4.3.0 --directory ${INPUT_DIR}/dorado-0.5.3-linux-x64/models/

DORADO_PATH=${INPUT_DIR}/dorado-0.5.3-linux-x64
```

Otherwise, if you are using macOS, the use following method to install Dorado:
```
#Download Dorado (macOS version) if you are using macOS
 wget https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.5.3-osx-arm64.zip
 unzip dorado-0.5.3-osx-arm64.zip -d ${INPUT_DIR}
 ${INPUT_DIR}/dorado-0.5.3-linux-x64/bin/dorado download --model  dna_r10.4.1_e8.2_400bps_hac@v4.3.0 --directory ${INPUT_DIR}/dorado-0.5.3-osx-arm64/models/

DORADO_PATH=${INPUT_DIR}/dorado-0.5.3-osx-arm64
```

`DORADO_PATH` variable stores the path to Dorado installation.

## 2.2 Dorado basecalling with move tables
In order to generate features for our model, we will first basecall our sample with Dorado using `--emit-moves` to create move tables in addition to basecalls. We will also provide reference genome to Dorado produced aligned BAM file, but you can also skip providing reference genome to Dorado and align yourself later using minimap2.

```
#basecall the sample
${DORADO_PATH}/bin/dorado basecaller --emit-moves --recursive --reference ${INPUT_DIR}/GRCh38.fa ${DORADO_PATH}/models/dna_r10.4.1_e8.2_400bps_hac@v4.3.0  ${INPUT_DIR}/data_download/signal_files/  > ${INPUT_DIR}/bam_files/all_reads.bam
```

The above command will create `${INPUT_DIR}/bam_files/all_reads.bam` BAM file from the basecalling of sample `${INPUT_DIR}/data_download/signal_files/`. We will use this sample for training and validation of model, and then again for running DeepMod2 on the trained model.

## 2.3 Generate Training Features
We will use `generate_features.py` module under `train` folder of DeepMod2 repository to generate features for model training. You can find a list of options using `python ${DeepMod2_DIR}/train/generate_features.py --help` command. We will generate features separately for the sample by providing signal POD5 file as `--input`, BAM file as `--bam` and a list of positions with modified/unmodified labels as `--pos_list`. In this case, we will use a window size of 10, which means how many bases before and after each base position of interest (from pos_list) to include in feature generation. At this stage, you do not need to specify a motif to `generate_features.py`, as you are providing a list of coordinates directly. You can use `--chrom  CHROM` to generate features for just a single chromosome and use `--threads NUM_THREADS` to speed up feature generation.

```
#generate features for modified sample
python ${DeepMod2_DIR}/train/generate_features.py --bam ${INPUT_DIR}/bam_files/all_reads.bam --input  ${INPUT_DIR}/data_download/signal_files/ --ref ${INPUT_DIR}/GRCh38.fa --file_type pod5 --threads 4 --output ${OUTPUT_DIR}/features/mixed/ --pos_list  ${INPUT_DIR}/data_download/label_files/mixed_list  --window 10
```

This will allow us to create features under `${OUTPUT_DIR}/features/mixed/` directory that contains both modified and unmodified instances. We will supply this folder to model training module.

## 2.4 Train DeepMod2 Model
We will train a deep learning model using `train_models.py` module under `train` folder of DeepMod2 repository. You can find a full list of options using `python ${DeepMod2_DIR}/train/train_models.py --help` command. In this demo, we will train a small model:

- using BiLSTM architecture (`--model_type bilstm`)
- model will have 2 recurrent layers (`--num_layers 2`) 
- model will have 32 hidden nodes in LSTM layers (`--dim_feedforward 32`)
- model will have 32 hidden nodes in fully connected classifier layer (`--num_fc 32`)
- trained for 10 epochs (`--epochs 10`)
- trained using a learning rate of 0.01 (`--lr 0.01`)
- trained using L2 regularization of 0.01 (`--l2_coef 0.01`)
- the model will include reference sequence as a features (`--include_ref`)
- the model will be validated using a random 50% validation split of training datasets (`--validation_type split --validation_fraction 0.5`)
- the model training will be reproducible using the same fixed seed (`--seed 0`)

We will provide the training features using `--mixed_training_dataset` parameter.

```
#train model using mixed sample containinf modified and canonical instances
python ${DeepMod2_DIR}/train/train_models.py --mixed_training_dataset ${OUTPUT_DIR}/features/mixed/ --validation_type split --validation_fraction 0.5 --model_save_path ${OUTPUT_DIR}/mixed_bilstm/ --epochs 10 --batch_size 128 --model_type bilstm --num_layers 2 --num_fc 32 --dim_feedforward 32 --lr 0.01 --window 10 --include_ref --l2_coef 0.01 --seed 0
```

This example uses a split of training dataset for validation, but you can generate features for a different genome or sample or chromosome using `generate_features.py`, and provide those features for validation by specifying validation type and a path to the validation dataset as follows `--validation_type dataset --validation_dataset path_to_validation_features`.

Since we fixed a random seed, we should be able to reproduce the following training and validation log:

<details>
  <summary>
    Training Log
  </summary>

```
mixed_training_dataset: ['training_data/features/mixed/']
can_training_dataset: None
mod_training_dataset: None
validation_type: split
validation_fraction: 0.5
validation_dataset: None
prefix: model
weights: equal
model_save_path: training_data/mixed_bilstm/
epochs: 10
batch_size: 128
retrain: None
fc_type: all
model_type: bilstm
window: 10
num_layers: 2
dim_feedforward: 32
num_fc: 32
embedding_dim: 4
embedding_type: one_hot
pe_dim: 16
pe_type: fixed
nhead: 4
include_ref: True
train_w_wo_ref: False
lr: 0.01
l2_coef: 0.01
seed: 0

Starting training.
Number of Modified Instances=783.0
Number of Un-Modified Instances=774.0
Positive Label Weight=1.0

BiLSTM(
  (emb): SeqEmbed(
    (seq_emb): RefandReadEmbed(
      (read_emb): OneHotEncode()
      (ref_emb): OneHotEncode()
    )
  )
  (bilstm): LSTM(19, 32, num_layers=2, batch_first=True, bidirectional=True)
  (classifier): ClassifierAll(
    (fc): Linear(in_features=1344, out_features=32, bias=True)
    (out): Linear(in_features=32, out_features=1, bias=True)
  )
)
# Parameters= 81729
..

Epoch 1: #Train=778  #Test=779  Time=0.4083
Training  Loss:  Total: 0.6904
Training Accuracy:  Total: 0.5373
Training Precision:  Total: 0.5560
Training Recall:  Total: 0.3821
Training F1:  Total: 0.4529

Testing  Loss:  Total: 0.7076
Testing Accuracy:  Total: 0.5045
Testing Precision:  Total: 0.5045
Testing Recall:  Total: 1.0000
Testing F1:  Total: 0.6706
..

Epoch 2: #Train=778  #Test=779  Time=0.0381
Training  Loss:  Total: 0.7017
Training Accuracy:  Total: 0.5193
Training Precision:  Total: 0.5139
Training Recall:  Total: 0.7590
Training F1:  Total: 0.6128

Testing  Loss:  Total: 0.6864
Testing Accuracy:  Total: 0.5687
Testing Precision:  Total: 0.5580
Testing Recall:  Total: 0.6972
Testing F1:  Total: 0.6199
..

Epoch 3: #Train=778  #Test=779  Time=0.0382
Training  Loss:  Total: 0.6805
Training Accuracy:  Total: 0.5990
Training Precision:  Total: 0.5707
Training Recall:  Total: 0.8077
Training F1:  Total: 0.6688

Testing  Loss:  Total: 0.6832
Testing Accuracy:  Total: 0.5469
Testing Precision:  Total: 0.5287
Testing Recall:  Total: 0.9389
Testing F1:  Total: 0.6764
..

Epoch 4: #Train=778  #Test=779  Time=0.0379
Training  Loss:  Total: 0.6724
Training Accuracy:  Total: 0.5977
Training Precision:  Total: 0.5672
Training Recall:  Total: 0.8333
Training F1:  Total: 0.6750

Testing  Loss:  Total: 0.6620
Testing Accuracy:  Total: 0.5944
Testing Precision:  Total: 0.5612
Testing Recall:  Total: 0.8982
Testing F1:  Total: 0.6908
..

Epoch 5: #Train=778  #Test=779  Time=0.0379
Training  Loss:  Total: 0.6386
Training Accuracy:  Total: 0.6517
Training Precision:  Total: 0.6024
Training Recall:  Total: 0.8974
Training F1:  Total: 0.7209

Testing  Loss:  Total: 0.5991
Testing Accuracy:  Total: 0.6842
Testing Precision:  Total: 0.6491
Testing Recall:  Total: 0.8142
Testing F1:  Total: 0.7223
..

Epoch 6: #Train=778  #Test=779  Time=0.0378
Training  Loss:  Total: 0.5673
Training Accuracy:  Total: 0.7301
Training Precision:  Total: 0.6765
Training Recall:  Total: 0.8846
Training F1:  Total: 0.7667

Testing  Loss:  Total: 0.5319
Testing Accuracy:  Total: 0.7279
Testing Precision:  Total: 0.6843
Testing Recall:  Total: 0.8550
Testing F1:  Total: 0.7602
..

Epoch 7: #Train=778  #Test=779  Time=0.0389
Training  Loss:  Total: 0.4900
Training Accuracy:  Total: 0.7661
Training Precision:  Total: 0.7537
Training Recall:  Total: 0.7923
Training F1:  Total: 0.7725

Testing  Loss:  Total: 0.5002
Testing Accuracy:  Total: 0.7484
Testing Precision:  Total: 0.6913
Testing Recall:  Total: 0.9059
Testing F1:  Total: 0.7841
..

Epoch 8: #Train=778  #Test=779  Time=0.0382
Training  Loss:  Total: 0.4430
Training Accuracy:  Total: 0.8123
Training Precision:  Total: 0.7961
Training Recall:  Total: 0.8410
Training F1:  Total: 0.8180

Testing  Loss:  Total: 0.4391
Testing Accuracy:  Total: 0.7856
Testing Precision:  Total: 0.7415
Testing Recall:  Total: 0.8830
Testing F1:  Total: 0.8060
..

Epoch 9: #Train=778  #Test=779  Time=0.0377
Training  Loss:  Total: 0.3857
Training Accuracy:  Total: 0.8303
Training Precision:  Total: 0.8131
Training Recall:  Total: 0.8590
Training F1:  Total: 0.8354

Testing  Loss:  Total: 0.3735
Testing Accuracy:  Total: 0.8203
Testing Precision:  Total: 0.7768
Testing Recall:  Total: 0.9033
Testing F1:  Total: 0.8353
..

Epoch 10: #Train=778  #Test=779  Time=0.0378
Training  Loss:  Total: 0.3244
Training Accuracy:  Total: 0.8548
Training Precision:  Total: 0.8386
Training Recall:  Total: 0.8795
Training F1:  Total: 0.8586

Testing  Loss:  Total: 0.3084
Testing Accuracy:  Total: 0.8678
Testing Precision:  Total: 0.8519
Testing Recall:  Total: 0.8931
Testing F1:  Total: 0.8720
Time taken=2.4622
```
</details>

At the end of the 10th epoch, we get 86.78% model validation accuracy. The model is saved after each epoch under `${OUTPUT_DIR}/mixed_bilstm/` directory specified by `--model_save_path`, with each saved model checkpoint having the following naming format: "prefix.epochN.validation_accuracy", which can be see by listing the contents of model_save_path folder using the following command:

```
ls -1trh ${OUTPUT_DIR}/mixed_bilstm/
```

<details>
  <summary>
Folder contents:
  </summary>

```
model.cfg
model.epoch1.0.5045
model.epoch2.0.5687
model.epoch3.0.5469
model.epoch4.0.5944
model.epoch5.0.6842
model.epoch6.0.7279
model.epoch7.0.7484
model.epoch8.0.7856
model.epoch9.0.8203
model.epoch10.0.8678
model.log
```
</details>

The folder contains a long file `model.log`, 10 saved model checkpoints `model.epochY.0.XXXX`, and model configuration file `model.cfg`. When we want to use this model, we have to provide a saved checkpoint and the model configuration file to DeepMod2. For now, we will use the last checkpoint file for model inference:

```
model_path=`ls -1tr ${OUTPUT_DIR}/mixed_bilstm/model.epoch*|tail -1`
```

## 2.5 Test DeepMod2 Model
We will now use the model `model.epoch10.0.8678` and `model.cfg` on the test dataset using DeepMod2's `detect` module. We will provide the model as `--model PATH_TO_MODEL_CONFIGURATION_FILE,PATH_TO_MODEL_CHECKPOINT` where we provide the paths to model configuration file and model checkpoint separated by a comma to `--model` parameter. 

For inference, we need to provide a motif using `--motif` parameter as `--motif CG 0` by specifying the motif and 0-based indices of the bases of interest within the motif separated by whitespace. This parameter tells which base in the motif to call modification on. The default behaviour is to call modifications on all loci of a read that either match the motif, or map to a reference position that matches the motif. You can choose to call modifications only on those read loci that map to a reference position with motif match by using `--reference_motif_only` parameter. You can further narrow down the loci where the modification is called if you are only interested in a select few reference motif positions by specifying a file with a list of coordinates of interest using `--mod_positions` parameter. This file is whitespace separated and has the following format: "contig position strand" on each line, where position should be zero-based and strand should be "+" or "-".

For now, we will just use `--motif CG 0` to call modifications, and provide BAM file, signal files and the reference genome to DeepMod2.

```
# Call modifications on the test dataset
python ${DeepMod2_DIR}/deepmod2 detect --model ${OUTPUT_DIR}/mixed_bilstm/model.cfg,${model_path} --file_type pod5 --bam ${INPUT_DIR}/bam_files/all_reads.bam --input ${INPUT_DIR}/data_download/signal_files/ --output ${PREDICTION_DIR} --ref ${INPUT_DIR}/GRCh38.fa --threads 4 --motif CG 0
```

The modification calling results will be under `${PREDICTION_DIR}` folder specified by `--output` parameter. For a detailed description of the output files, see [Example.md](https://github.com/WGLab/DeepMod2/blob/main/docs/Example.md#113-methylation-calling-with-deepmod2). The output folder will have the following files:

```
args -> Shows the arguments and command use to run DeepMod2
output.bam -> Unsorted modification tagged BAM file
output.per_read -> Per-read modification calls in sorted BED file
output.per_site -> Per-site modification calls for +- strands separately in sorted BED file.
output.per_site.aggregated -> Per-site modification calls for with counts for +- strands combined. Produces only for CG motif. 
```

We can sort the modification tagged BAM file using samtools to view modification in IGV:

```
samtools sort ${PREDICTION_DIR}/output.bam -o ${PREDICTION_DIR}/sorted.bam --write-index
```

Open the BAM file ${PREDICTION_DIR}/sorted.bam in IGV, select Color alignments by base modification. Go to the region chr11:2699000-2702000 to see the following methylation tags:

<PNG>


--------------------------------------
# 3. Train model from samples with mixed labels of modified and unmodified (canonical) bases, as well as samples with only modified or unmodified (canonical) bases
If you have a sample with modified bases or a sample with unmodified bases as described in section 1, or a sample with mixed labels of modified and unmodified labels as descrbied in section 2, or any combination of such samples, you can generate features for each sample using `generate_features.py` and then supply these features to `train_models.py` to train a model. This situation is described in the figure below or as a subset of the figure below:

<p align="center"> <img src="https://github.com/WGLab/DeepMod2/assets/35819083/59de6365-2a91-47e7-b854-bb3e30c65eee"  width="75%" > </p>
<p align="center"> <img src="https://github.com/WGLab/DeepMod2/assets/35819083/49a4fb31-88e1-48f4-bcb3-9020454fcd93"  width="75%" > </p>

**Case 1**

For example, if you have a sample with mixed labels called MIXED_SAMPLE, a sample with modified bases called MOD_SAMPLE and a sample with only canonical bases called CAN_SAMPLE, then you can use `generate_features.py` to generate features for each sample. If you want to use them all for training, then you can provide them to `train_models.py` as:

```
python DeepMod2/train/train_models.py --mixed_training_dataset MIXED_SAMPLE --mod_training_dataset MOD_SAMPLE --can_training_dataset CAN_SAMPLE ...
```

**Case 2**

Consider another example where you have two samples with mixed labels called MIXED_SAMPLE_1 and MIXED_SAMPLE_2, and a sample with only canonical bases called CAN_SAMPLE, then you can use `generate_features.py` to generate features for each sample. If you want to use them all for training, then you can provide them to `train_models.py` as:

```
python DeepMod2/train/train_models.py --mixed_training_dataset MIXED_SAMPLE_1 MIXED_SAMPLE_2 --can_training_dataset CAN_SAMPLE ...
```

**Case 3**

Consider yet another example where you have a sample with mixed labels called MIXED_SAMPLE, and a sample with only canonical bases called CAN_SAMPLE. Let's say you also have another sample with mixed labels or modified labels only called VALIDATION_SAMPLE, then you can use `generate_features.py` to generate features for each sample. Then you can provide them to `train_models.py` to train the model using MIXED_SAMPLE and CAN_SAMPLE and validate it using VALIDATION_SAMPLE as follows:

```
python DeepMod2/train/train_models.py --mixed_training_dataset MIXED_SAMPLE_1 MIXED_SAMPLE_2 --can_training_dataset CAN_SAMPLE --validation_dataset VALIDATION_SAMPLE
```
