#!/bin/bash

module load cudnn-7.6

module load cuda-toolkit-10.1

module load python3

module load ffmpeg-4.4.0

module load gcc-7.1.0  

virtualenv -p python3 $HOME/tmp/deepspeech-gpu-venv/
source $HOME/tmp/deepspeech-gpu-venv/bin/activate

pip3 install deepspeech-gpu

pip3 install python-Levenshtein

pip3 install scipy