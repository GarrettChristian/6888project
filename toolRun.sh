#!/bin/bash

# Hard coded required arguments
model=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/deepspeech-0.9.3-models.pbmm
scorer=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/deepspeech-0.9.3-models.scorer
seeds=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/6888project/test-clean
realWorldNoise=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/6888project/yt-audio
# seeds=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/6888project/seeds1

# Optional mutations
# Accepted Mutations: PITCH,SPEED,VOLUME,LOOP,CONCAT,SUBSECTION,CUT_SECTION,REARRANGE,REMOVE_BELOW_DECIBEL,WHITE_NOISE,REAL_WORLD_NOISE
mutations=SPEED

# Run the tool 
python3 sttFuzzer.py --model $model --scorer $scorer --seeds $seeds --realWorldNoise $realWorldNoise --threads 5 --save 1


# Run with optional
# python3 sttFuzzer.py --model $model --scorer $scorer --seeds $seeds --realWorldNoise $realWorldNoise --mutations $mutations 

