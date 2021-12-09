# 6888project
Garrett Christian

---
## Description

Audio fuzzing and mutation tool to test mozilla's deepspeech speech to text model created for my 6888 final project.

---

## Setup

### System Requirements
- ffmpeg
- python3
- GCC
- cuda 10.1 (if using gpu enhancements for the model)
- cudnn 7.6 (if using gpu enhancements for the model)


### Set up Steps for General CPU Machine (validated on macOS)
- pip3 install virtualenv
- virtualenv -p python3 $HOME/tmp/deepspeech-gpu-venv/
- source $HOME/tmp/deepspeech-gpu-venv/bin/activate
- pip3 install deepspeech
- pip3 install python-Levenshtein
- pip3 install scipy


### Set up Steps on the UVA GPU Server
- module load cudnn-7.6
- module load cuda-toolkit-10.1
- module load python3
- module load ffmpeg-4.4.0
- module load gcc-7.1.0  
- virtualenv -p python3 $HOME/tmp/deepspeech-gpu-venv/
- source $HOME/tmp/deepspeech-gpu-venv/bin/activate
- pip3 install deepspeech-gpu
- pip3 install python-Levenshtein
- pip3 install scipy

### Get seeds (or create them yourself see scripts below)
- test-clean seeds in wav format in zip file https://drive.google.com/file/d/1AG6ihO3g4YQEbvnYRHP3Dr3l__oIuX87/view?usp=sharing
- yt-audio real world noise seeds used in wav format in zip file https://drive.google.com/file/d/1SRmAEKw-SxFHnavM0u9S223Q6pdCdyQS/view?usp=sharing

### Get the Model and Scorer
- https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3

---

## Usage

### Running the Tool
- modify toolRun.sh with your specific parameters
- bash toolRun.sh
- NOTE everytime you run the tool the output directory will be reset

### Arguments 
#### Required Arguments 
- --model [path to the deep speech model] 
- --scorer [path to the deep speech scorer]
- --seeds [path to the wav seed files] 
- --realWorldNoise [path to the wav real world noise files] 

#### Optional Arguments
- --mutations [mutations to attempt in a comma separated list] (default all mutations)
  
  Accepted mutations: PITCH,SPEED,VOLUME,LOOP,CONCAT,SUBSECTION,CUT_SECTION,REARRANGE,REMOVE_BELOW_DECIBEL,WHITE_NOISE,REAL_WORLD_NOISE,VIBRATO,TREBLE,BASE

- --threads [number of threads] (default 1 thread)
- --save [number of mutation for every type each thread should save] (default is 10)
- --saveAll (default is false)

---

## Other Information

### Other Directories
- gcAudioResults - where the results of a run are stored
- samples - includes samples of the mutations
- scripts - test scripts for the tool and other utility scripts
- seeds1 - a sample seed used for testing purposes
- test-clean - wav seed files from openSLR
- yt-audio - wav randomly selected 10 second youtube clips from google's labeled AudioDataset
- deepspeech-0.9.3-models.pbmm deep speech model
- deepspeech-0.9.3-models.scorer deep speech scorer

### Files Included in Scripts
- scripts/balanced_train_segments.csv used by googleAudioSources to randomly select real world noise from 
- scripts/client.py sample code for the deep speech model on python
- scripts/flacConversion.py conversion script to change the format of flac audio files to wav while retaining organizational file structure
- scripts/gdown.py a script to download files off of google drive used to move the seed file to the UVA server
- scripts/googleAudioSources a script to download files off of youtube based on googles labeled AudioSet datasource 
- scripts/modules.sh a script that included all of the UVA dependencies 
- scripts/multithreaded.py a test script for the multithreading python approach used
- scripts/ontology.json used by googleAudioSources to map the names of the ids 

### Other Relevant Links
- Deepspeech documentation https://deepspeech.readthedocs.io/en/r0.9/
- Deepspeech github https://github.com/mozilla/DeepSpeech
- OpenSLR https://openslr.org/
- Google's AudioDataset https://research.google.com/audioset/index.html



