# DeepMod2 Usage


DeepMod2 has three run options `detect-guppy, detect-tombo, merge` which can be shown using `python DeepMod2/deepmod2 --help`
```
usage: deepmod2 [-h] [--print_models] {detect-guppy,detect-tombo,merge} ...

optional arguments:
  -h, --help            show this help message and exit
  --print_models        Print details of models available (default: False)

Options:
  {detect-guppy,detect-tombo,merge}
    detect-guppy        Call methylation from Guppy FAST5 files
    detect-tombo        Call methylation from Tombo FAST5 files
    merge               Merge per-read calls into per-site calls
```

In short, `detect-guppy` and `detect-tombo` are for methylation calling using Guppy and Tombo generated FAST5 files, respectively, whereas `merge` is for merging methylation calls from several runs.

`deepmod2 --print_models` shows names of available models as well as their description, such as datasets used for training. Names of these models can be used with `--model` argument to specify which model to use. Please DeepMod2 run options `detect-guppy, detect-tombo`, and model that are compatible with type of FAST5 files you are using. Use Guppy models and `detect-guppy` with Guppy basecalled FAST5 files, and Tombo models and `detect-tombo` with Tombo resquiggled FAST5 files. Note that Tombo FAST5 files should contain event standard deviation information, which can be obtained by using `--include-event-stdev` during `tombo resquiggle`.

## detect-guppy
DeepMod2 can detect 5mC methylation from Guppy basecalled FAST5 files using `detect-guppy` option. In addition to FAST5 files, this option requires a BAM file containing alignments of reads from FAST5 files, as well as a reference genome FASTA file. This option support both single and multi FAST5 formats.

```
usage: deepmod2 detect-guppy [-h] [--file_name FILE_NAME] [--output OUTPUT] [--threads THREADS] --fast5 FAST5 [--skip_per_site] [--model MODEL]
                                [--chrom [CHROM [CHROM ...]]] --bam BAM --ref REF [--guppy_group GUPPY_GROUP]

Guppy basecaller options

optional arguments:
  -h, --help            show this help message and exit
  --file_name FILE_NAME
                        Name of the output file (default: output)
  --output OUTPUT       Path to folder where intermediate and final files will be stored, default is current working directory (default: None)
  --threads THREADS     Number of processors to use (default: 1)
  --fast5 FAST5         Path to folder containing tombo requiggle Fast5 files. Fast5 files will be recusrviely searched (default: None)
  --skip_per_site       Skip per site detection and stop after per-read methylation calling. (default: False)
  --model MODEL         Name of the model. Default model is "guppy_na12878" but you can provide a path to your own model. (default: guppy_na12878)
  --chrom [CHROM [CHROM ...]]
                        A space/whitespace separated list of contigs, e.g. chr3 chr6 chr22. If not list is provided then all chromosomes in the reference
                        are used. (default: None)
  --bam BAM             Path to bam file if Guppy basecaller is user. BAM file is not needed with Tombo fast5 files. (default: None)
  --ref REF             Path to reference file (default: None)
  --guppy_group GUPPY_GROUP
                        Name of the guppy basecall group (default: Basecall_1D_000)
```

## detect-tombo
DeepMod2 can detect 5mC methylation from Tombo resquiggled FAST5 files using `detect-tombo` option. This option requires only FAST5 files produced with Tombo resquiggle and can only support single FAST5 format.

```
usage: deepmod2 detect-tombo [-h] [--file_name FILE_NAME] [--output OUTPUT] [--threads THREADS] --fast5 FAST5 [--skip_per_site]
                                [--tombo_group TOMBO_GROUP] [--model MODEL]

Tombo basecaller options

optional arguments:
  -h, --help            show this help message and exit
  --file_name FILE_NAME
                        Name of the output file (default: output)
  --output OUTPUT       Path to folder where intermediate and final files will be stored, default is current working directory (default: None)
  --threads THREADS     Number of processors to use (default: 1)
  --fast5 FAST5         Path to folder containing tombo requiggle Fast5 files. Fast5 files will be recusrviely searched (default: None)
  --skip_per_site       Skip per site detection and stop after per-read methylation calling. (default: False)
  --tombo_group TOMBO_GROUP
                        Name of the tombo group (default: RawGenomeCorrected_000)
  --model MODEL         Name of the model. Default model is "tombo_na12878" but you can provide a path to your own model. (default: tombo_na12878)
```

## merge
For large datasets, it can be useful to split the data and run multiple instances of DeepMod2 for speedup. In this case, you should run `detect-guppy` or `detect-tombo` with `--skip_per_site` option and use `deepmod2 merge` to merge per-read calls into per-site calls.

```
usage: deepmod2 merge [-h] [--file_name FILE_NAME] [--output OUTPUT] [--threads THREADS] [--inputs [INPUTS [INPUTS ...]]] [--list LIST]

Merge per-read calls into per-site calls

optional arguments:
  -h, --help            show this help message and exit
  --file_name FILE_NAME
                        Name of the output file (default: output)
  --output OUTPUT       Path to folder where intermediate and final files will be stored, default is current working directory (default: None)
  --threads THREADS     Number of processors to use (default: 1)
  --inputs [INPUTS [INPUTS ...]]
                        List of paths of per-read methylation calls to merge. File paths should be separated by space/whitespace. Use either --inputs or
                        --list argument, but not both. (default: None)
  --list LIST           A file containing paths to per-read methylation calls to merge (one per line). Use either --inputs or --list argument, but not
                        both. (default: None)
```