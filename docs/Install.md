## Conda Installation
First, install Miniconda, a minimal installation of Anaconda, which is much smaller and has a faster installation:
```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Go through all the prompts (installation in `$HOME` is recommended). After Anaconda is installed successfully, simply run:
```
git clone https://github.com/WGLab/DeepMod2.git
conda env create -f DeepMod2/environment.yml
conda activate deepmod2
```

After installing, run `python DeepMod2/deepmod2 --help` to see the run options.

### Installation for GPU
If you want to use GPU to accelerate DeepMod2, make sure to install cuda enabled version of pytorch. Details for GPU accelerated pytorch can be found here https://pytorch.org/get-started/locally/ and you can select the installation command best suited to your system. One xample of installation command is shown below for CUDA 11.8:

`conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`
