#!/bin/bash


model=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/deepspeech-0.9.3-models.pbmm
scorer=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/deepspeech-0.9.3-models.scorer
# audio=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/6888project/audio/2830-3980-0043.wav
audio=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/6888project/output_001.wav

python3 customAttempt1.py --model $model --scorer $scorer --audio $audio