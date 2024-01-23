# DeepMod2 Usage


DeepMod2 has two run options `detect` and `merge` which can be shown using `python deepmod2 --help`
```
usage: deepmod2 [-h] [--print_models] {detect,merge} ...

options:
  -h, --help      show this help message and exit
  --print_models  Print details of models available (default: False)

Options:
  {detect,merge}
    detect        Call methylation from Guppy or Dorado basecalled POD5/FAST5 files using move tables for signal alignment.
    merge         Merge per-read calls into per-site calls

```

In short, `detect` is for methylation calling using Dorado and Guppy generated POD5/FAST5 files, respectively, whereas `merge` is for merging per-read methylation calls from several runs into per-site methylation calls.

`python deepmod2 --print_models` shows names of available models as well as their description, such as datasets used for training. Names of these models can be used with `--model` argument to specify which model to use. Please use the appropriate DeepMod2 model for the flowcell you are using, e.g. R9.4.1 vs R10.4.1.

## detect-guppy
DeepMod2 can detect 5mC methylation from Dorado and Guppy basecalled POD5/FAST5 signal files using `detect` option. In addition to signal files, you are required to provide a BAM file containing read sequences. It is recommended that the reads in the BAM file are aligned to a reference genome for more accurate results, in which case you should also provide a reference genome FASTA file. You are required to provide move tables for the signal files, so make sure you output move tables when you run the basecaller. The move table information can be part of the BAM file as mv and ts tags, or it can be part of FAST5 file (which is deprecated now).

```
usage: deepmod2 detect [-h] [--prefix PREFIX] [--output OUTPUT] [--qscore_cutoff QSCORE_CUTOFF] [--length_cutoff LENGTH_CUTOFF] [--mod_t MOD_T] [--unmod_t UNMOD_T] [--include_non_cpg_ref]
                       [--threads THREADS] [--ref REF] --model MODEL --bam BAM --file_type {fast5,pod5} --input INPUT [--guppy_group GUPPY_GROUP] [--chrom [CHROM ...]] [--fast5_move]
                       [--skip_per_site] [--device DEVICE] [--disable_pruning] [--exclude_ref_features] [--batch_size BATCH_SIZE] [--bam_threads BAM_THREADS] [--skip_unmapped]

options:
  -h, --help            show this help message and exit
  --prefix PREFIX       Prefix for the output files (default: output)
  --output OUTPUT       Path to folder where intermediate and final files will be stored, default is current working directory (default: None)
  --qscore_cutoff QSCORE_CUTOFF
                        Minimum cutoff for mean quality score of a read (default: 0)
  --length_cutoff LENGTH_CUTOFF
                        Minimum cutoff for read length (default: 0)
  --mod_t MOD_T         Probability threshold for a per-read prediction to be considered modified. Only predictiond with probability >= mod_t will be considered as modified for
                        calculation of per-site modification levels. (default: 0.5)
  --unmod_t UNMOD_T     Probability threshold for a per-read prediction to be considered unmodified. Only predictiond with probability < unmod_t will be considered as unmodified for
                        calculation of per-site modification levels. (default: 0.5)
  --include_non_cpg_ref
                        Include non-CpG reference loci in per-site output where reads have CpG motif. (default: False)
  --threads THREADS     Number of threads to use for processing signal and running model inference. If a GPU is used for inference, then --threads number of threads will be running on GPU
                        concurrently. The total number of threads used by DeepMod2 is equal to --threads plus --bam_threads. It is recommended to run DeepMod2 with mutliple cores, and use
                        at least 4 bam_threads for compressing BAM file. (default: 4)
  --ref REF             Path to reference FASTA file to anchor methylation calls to reference loci. If no reference is provided, only the motif loci on reads will be used. (default: None)
  --guppy_group GUPPY_GROUP
                        Name of the guppy basecall group (default: Basecall_1D_000)
  --chrom [CHROM ...]   A space/whitespace separated list of contigs, e.g. chr3 chr6 chr22. If not list is provided then all chromosomes in the reference are used. (default: None)
  --fast5_move          Use move table from FAST5 file instead of BAM file. If this flag is set, specify a basecall group for FAST5 file using --guppy_group parameter and ensure that the
                        FAST5 files contains move table. (default: False)
  --skip_per_site       Skip per site output (default: False)
  --device DEVICE       Device to use for running pytorch models. you can set --device=cpu for cpu, or --device=cuda for GPU. You can also specify a particular GPU device such as
                        --device=cuda:0 or --device=cuda:1 . If --device paramater is not set by user, then GPU will be used if available otherwise CPU will be used. (default: None)
  --disable_pruning     Disable model pruning (not recommended for CPU inference). By default models are pruned to remove some weights with low L1 norm in linear layers. Pruning has
                        little effect on model accuracy, it can signifcantly improve CPU inference time but not GPU inference time. (default: False)
  --exclude_ref_features
                        Exclude reference sequence from feature matrix. By default, if a reference FASTA file is provided via --ref parameter, then the reference sequence is added as a
                        feature for aligned reads, but not if a read is unmapped or if no reference is provided. (default: False)
  --batch_size BATCH_SIZE
                        Batch size to use for GPU inference. For CPU inference, batch size is fixed at 512. (default: 1024)
  --bam_threads BAM_THREADS
                        Number of threads to use for compressed BAM output. Setting it lower than 4 can significantly lower the runtime. (default: 4)
  --skip_unmapped       Skip unmapped reads from methylation calling (default: False)

Required Arguments:
  --model MODEL         Name of the model. Recommended model for R9.4.1 flowcells is "bilstm_r9.4.1", for R10.4.1 flowcell (4kHz sampling) it is "bilstm_r10.4.1_4khz". Use --print_models
                        to display all models available. (default: None)
  --bam BAM             Path to aligned or unaligned BAM file. It is ideal to have move table in BAM file but move table from FAST5 fies can also be used. Aligned BAM file is required for
                        reference anchored methylation calls, otherwise only the motif loci on reads will be called. (default: None)
  --file_type {fast5,pod5}
                        Specify whether the signal is in FAST5 or POD5 file format. If POD5 file is used, then move table must be in BAM file. (default: None)
  --input INPUT         Path to POD5/FAST5 file or folder containing POD5/FAST5 files. If folder provided, then POD5/FAST5 files will be recusrviely searched (default: None)

```

## merge
For large datasets, it can be useful to split the data and run multiple instances of DeepMod2 for speedup. In this case, you can run `detect` with `--skip_per_site` option and use `deepmod2 merge` to merge per-read calls into per-site calls.

```
usage: deepmod2 merge [-h] [--prefix PREFIX] [--output OUTPUT] [--qscore_cutoff QSCORE_CUTOFF] [--length_cutoff LENGTH_CUTOFF] [--mod_t MOD_T] [--unmod_t UNMOD_T] [--include_non_cpg_ref]
                      [--input [INPUT ...]] [--list LIST]

options:
  -h, --help            show this help message and exit
  --prefix PREFIX       Prefix for the output files (default: output)
  --output OUTPUT       Path to folder where intermediate and final files will be stored, default is current working directory (default: None)
  --qscore_cutoff QSCORE_CUTOFF
                        Minimum cutoff for mean quality score of a read (default: 0)
  --length_cutoff LENGTH_CUTOFF
                        Minimum cutoff for read length (default: 0)
  --mod_t MOD_T         Probability threshold for a per-read prediction to be considered modified. Only predictiond with probability >= mod_t will be considered as modified for
                        calculation of per-site modification levels. (default: 0.5)
  --unmod_t UNMOD_T     Probability threshold for a per-read prediction to be considered unmodified. Only predictiond with probability < unmod_t will be considered as unmodified for
                        calculation of per-site modification levels. (default: 0.5)
  --include_non_cpg_ref
                        Include non-CpG reference loci in per-site output where reads have CpG motif. (default: False)
  --input [INPUT ...]   List of paths of per-read methylation calls to merge. File paths should be separated by space/whitespace. Use either --input or --list argument, but not both.
                        (default: None)
  --list LIST           A file containing paths to per-read methylation calls to merge (one per line). Use either --inputs or --list argument, but not both. (default: None)
```