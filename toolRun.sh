#!/bin/bash

# Required arguments for the tool to run
model=deepspeech-0.9.3-models.pbmm
scorer=deepspeech-0.9.3-models.scorer
seeds=test-clean
realWorldNoise=yt-audio
# seeds=seeds1

# Optional mutations
# Accepted Mutations: PITCH,SPEED,VOLUME,LOOP,CONCAT,SUBSECTION,CUT_SECTION,REARRANGE,REMOVE_BELOW_DECIBEL,WHITE_NOISE,REAL_WORLD_NOISE,VIBRATO,TREBLE,BASE
mutations=TREBLE,BASE

# Run the tool 
python3 sttFuzzer.py --model $model --scorer $scorer --seeds $seeds --realWorldNoise $realWorldNoise --threads 4 --save 1

# Save all version
# python3 sttFuzzer.py --model $model --scorer $scorer --seeds $seeds --realWorldNoise $realWorldNoise --threads 5 --saveAll

# Run with optional mutations
# python3 sttFuzzer.py --model $model --scorer $scorer --seeds $seeds --realWorldNoise $realWorldNoise --mutations $mutations --threads 1 --save 1

