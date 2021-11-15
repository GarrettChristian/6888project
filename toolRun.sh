#!/bin/bash

# Hard coded arguments
model=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/deepspeech-0.9.3-models.pbmm
scorer=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/deepspeech-0.9.3-models.scorer
seeds=/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/6888project/seeds


# Run the tool 
python3 sttFuzzer.py --model $model --scorer $scorer --seeds $seeds