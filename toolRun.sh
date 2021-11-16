#!/bin/bash

# Hard coded arguments
model=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/deepspeech-0.9.3-models.pbmm
scorer=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/deepspeech-0.9.3-models.scorer
seeds=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/6888project/test-clean

# Accepted Mutations: PITCH,SPEED,VOLUME,LOOP,CONCAT,SUBSECTION,CUT_SECTION,REARRANGE,REMOVE_BELOW_DECIBLE,WHITE_NOISE,REAL_WORLD_NOISE

# Optional 
mutations=SPEED

# Run the tool 
python3 sttFuzzer.py --model $model --scorer $scorer --seeds $seeds


# Run with optional
# python3 sttFuzzer.py --model $model --scorer $scorer --seeds $seeds --mutations $mutations

