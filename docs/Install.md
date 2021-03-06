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

The above method installs CPU version of tensorflow for DeepMod2. Using a GPU version of Tensorflow with a GPU can allow significantly speed-up. You can install tensorflow GPU version (>v2.0) that works with your GPU driver.