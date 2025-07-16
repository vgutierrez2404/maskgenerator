## Installation 
Create a conda environment: 

```
conda create --name maskgenerator python=3.10

conda activate maskgenerator
```
Install pytorch>=2.5.1 torch>=2.5.1 and torchvision>=0.20.1 (requiered by Sam2): 

```
> git clone https://github.com/facebookresearch/sam2.git
> cd sam2
> pip install -e .
```
And downlowad SAM2.1 checkpoints: 

```
> cd checkpoints 
> ./download_ckpts.sh 
> cd ..
```
 
From a virtual environment
* Installed cuda version 12.1 https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local 
* cd sam2 and installed sam2 requirements: https://github.com/facebookresearch/sam2 
```
    pip install -e . 
```
## TODO 
- Add feature to select the points that we don't want in the segmentation mask 
- Update readme with the correct info to install and run the tool.
- Make able to select a remote location for the video via ssh. 

- Generate a fork of sam2 into the repository for easier installation (with the 
modifications made to the original code).

### Usage 

After installation, run: 
```
    python3 app.py # o el nombre que acabe teniendo 
``` 
and select the desired video. 

