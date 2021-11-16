import os
import uuid
import random
import shutil
import argparse
from pathlib import Path
import json
import wave
import nltk
import sys
import numpy as np

import scipy.io.wavfile as wav

from timeit import default_timer as timer
from deepspeech import Model, version
from enum import Enum

try:
    from shhlex import quote
except ImportError:
    from pipes import quote

# Enum of the different types of mutations supported
class Mutation(Enum):
    PITCH = "PITCH"
    SPEED = "SPEED"
    VOLUME = "VOLUME"
    LOOP = "LOOP"
    CONCAT = "CONCAT"            # takes the seed and adds a ranom seed as the second half
    SUBSECTION = "SUBSECTION"    # picks a random subsection of the seed from of length at least 1 second
    CUT_SECTION = "CUT_SECTION"  # removes a section of the clip of a least 1 second in length
    REARRANGE = "REARRANGE"
    REMOVE_BELOW_DECIBLE = "REMOVE_BELOW_DECIBLE"
    WHITE_NOISE = "WHITE_NOISE"
    REAL_WORLD_NOISE = "REAL_WORLD_NOISE"

# Output directory constants
RESULTS_DIR = "gcAudioResults"
AUDIO_OUTPUT_DIR = RESULTS_DIR + "/output"
SUCCESS_DIR = RESULTS_DIR + "/success"
FAILURE_DIR = RESULTS_DIR + "/failure"

# Global seeds and model
seeds = []
ds = None
mutantsEnabled = []

# Global results
numMutations = 0
failures = 0
mutationCount = {}
mutationFailureCount = {}

# Output json dictionary constants
ID = "id"
OUTPUT_FILE = "outputFile"
MUTATION = "mutation"
MUTATION_DETAILS = "mutationDetails"
ORIGINAL_TEXT = "originalText"
ORIGINAL_CONFIDENCE = "originalConfidence"
MUTATION_TEXT = "mutantText"
MUTATION_CONFIDENCE = "mutantConfidence"
COMMAND = "command"
SEED_FILE = "seedFile"

# --------------------------------------------------------

def printMutants(mutants):
    for i in range(len(mutants)):
        if len(mutants) == i + 1:
            print("%s" % (mutants[i].name), end='')
        else:
            print("%s, " % (mutants[i].name), end='')
    print()


# --------------------------------------------------------

def setup():
    global ds
    global seeds
    global mutationCount
    global mutationFailureCount
    global mutantsEnabled

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Running Garrett Christian\'s Audio Fuzzing Tool')
    parser.add_argument('--model', required=True, 
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', required=True,
                        help='Path to the external scorer file')
    parser.add_argument('--seeds', required=True,
                        help='Path to the seed files, NOTE these seeds must be .wav files')
    parser.add_argument('--mutations', required=False,
                        help='Mutations to perform comma seperated example: PITCH,SPEED')
    args = parser.parse_args()

    print("\n--------------------------------------------------------")
    print("Running Setup\n")

    # Set up the output folders
    # Remove the old output folder
    try:
        shutil.rmtree(RESULTS_DIR)
        print("Resetting results folder")
    except FileNotFoundError:
        print("Setting up new results folder")
    # Create the new result folders
    os.mkdir(RESULTS_DIR)
    print()

    # Set up the model
    print('Loading model from file {}'.format(args.model))
    model_load_start = timer()
    # sphinx-doc: python_ref_model_start
    ds = Model(args.model)
    # sphinx-doc: python_ref_model_stop
    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end))
    print()

    # Set up the scorer
    print('Loading scorer from files {}'.format(args.scorer))
    scorer_load_start = timer()
    ds.enableExternalScorer(args.scorer)
    scorer_load_end = timer() - scorer_load_start
    print('Loaded scorer in {:.3}s.'.format(scorer_load_end))
    print()

    # Get the file names of the seeds
    print("Seed Directory provided: {0}".format(args.seeds))
    for path in Path(args.seeds).rglob('*.wav'):
        seeds.append(str(path))

    print("seeds provided {0}".format(len(seeds)))
    print()

    # Get mutations to use
    mutantsEnabled = []
    if (args.mutations != None):
        for enableMutation in args.mutations.split(","):
            try:
                mutantToAdd = Mutation[enableMutation]
            except KeyError:
                print("%s is not a valid option" % (enableMutation))
                print("Accepted Mutations: ", end='')
                printMutants(list(Mutation))
                sys.exit(1)
            mutantsEnabled.append(mutantToAdd)
    else:
        for mutant in Mutation:
            mutantsEnabled.append(mutant)
    
    print("Mutations in use: ", end='')
    printMutants(mutantsEnabled)

    # Set up the results
    for mutant in mutantsEnabled:
        mutationCount[mutant.name] = 0
        mutationFailureCount[mutant.name] = 0
        makeDirSuccess = SUCCESS_DIR + "/" + mutant.name.lower()
        makeDirFailure = FAILURE_DIR + "/" + mutant.name.lower()
        makeDirOutput = AUDIO_OUTPUT_DIR + "/" + mutant.name.lower()
        os.makedirs(makeDirSuccess, exist_ok=True)
        os.makedirs(makeDirFailure, exist_ok=True)
        os.makedirs(makeDirOutput, exist_ok=True)

# --------------------------------------------------------
# MODEL SPECIFIC METHODS

def runModel(audioFile):
    fin = wave.open(audioFile, 'rb')
    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    fin.close()

    metadata = ds.sttWithMetadata(audio, 1).transcripts[0]

    text = ''.join(token.text for token in metadata.tokens)
    
    return (text, metadata.confidence)

# --------------------------------------------------------

def createMutant(seedfile):
    #  Select random Mutation
    mutation = random.choice(list(mutantsEnabled))

    id = str(uuid.uuid4())
    outputFile = "{audioOutDir}/{id}.wav".format(audioOutDir = AUDIO_OUTPUT_DIR + "/" + mutation.name.lower(), id = id)
    command = ""
    mutationDetails = ""

    # print('Mutant Selected %s %s' % (mutation.name, seedfile))

    if (Mutation.PITCH == mutation):
        command = "ffmpeg -i {seed} -af \"asetrate=44100*0.9\" -y {output} 2> /dev/null".format(
            seed = seedfile, output = outputFile)

    elif (Mutation.SPEED == mutation):
        speed = random.uniform(.5, .8)
        if random.randint(1, 2) % 2 == 0:
            speed = random.uniform(1.2, 1.5)
        
        command = "ffmpeg -i {seed} -filter:a \"atempo={speed}\" -vn {output}  2> /dev/null".format(
            seed = seedfile, speed = speed, output = outputFile)
        mutationDetails = "Speed %.2f" % (speed)

    elif (Mutation.VOLUME == mutation):
        volume = random.uniform(.3, .8)
        if random.randint(1, 2) % 2 == 0:
            volume = random.uniform(1.2, 1.7)
        
        command = "ffmpeg -i {seed} -af \"volume={volume}\" {output}  2> /dev/null".format(
            seed = seedfile, volume = volume, output = outputFile)
        mutationDetails = "Volume %.2f" % (volume)

    elif (Mutation.LOOP == mutation):
        command = "ffmpeg -i {seed} -filter_complex \
            \"[0:a]afifo[a0];[0:a]afifo[a1];[a0][a1]concat=n=2:v=0:a=1[a]\" \
            -map \"[a]\" {output} 2> /dev/null".format(
            seed = seedfile, output = outputFile)

    elif (Mutation.SUBSECTION == mutation):
        # Get length
        (sourceRate, sourceSig) = wav.read(seedfile)
        seedDuration = len(sourceSig) / float(sourceRate)

        print(seedDuration)
        # Pick start and end
        # Choose random number between 1 & len - 1
        # Leaves at minimum a second and max length - a second
        lenOfSection = random.uniform(1, seedDuration - 1)
        # Choose random number between 0 & length - length of the cut
        start = random.uniform(0, seedDuration - lenOfSection)
        end = start + lenOfSection
        command = "ffmpeg -ss {start} -i {seed} -to {end} -c copy {output} 2> /dev/null".format(
            start = start, seed = seedfile, end = end, output = outputFile)
        mutationDetails = "Start %f, End %f, Length %f, Length of Subsection %f" % (start, end, seedDuration, lenOfSection)

    elif (Mutation.CUT_SECTION == mutation):
        # Get length
        (sourceRate, sourceSig) = wav.read(seedfile)
        seedDuration = len(sourceSig) / float(sourceRate)

        # Pick start and end
        # Choose random number between 1 & len - 1
        # Leaves at minimum a second and max length - a second
        lenOfCut = random.uniform(1, seedDuration - 1)
        # Choose random number between 0 & length - length of the cut
        startOfCut = random.uniform(0, seedDuration - lenOfCut)
        endOfCut = startOfCut + lenOfCut
        command = "ffmpeg -i {seed} -filter_complex \
            \"[0]atrim=duration={start}[a];[0]atrim=start={end}[b];[a][b]concat=n=2:v=0:a=1\" \
            {output} 2> /dev/null".format(
            seed = seedfile, start = startOfCut, end = endOfCut, output = outputFile)
        
        mutationDetails = "Start %f, End %f, Length %f, Length of Cut %f" % (startOfCut, endOfCut, seedDuration, lenOfCut)

    elif (Mutation.REARRANGE == mutation):
        # Get length
        (sourceRate, sourceSig) = wav.read(seedfile)
        seedDuration = len(sourceSig) / float(sourceRate)

        print(seedDuration)
        # Pick start and end
        # Choose random number between 1 & len - 1
        # Leaves at minimum a second and max length - a second
        lenOfSection = random.uniform(1, seedDuration - 1)
        # Choose random number between 0 & length - length of the cut
        start = random.uniform(0, seedDuration - lenOfSection)
        end = start + lenOfSection

        if random.randint(1, 2) % 2 == 0:
            command = "ffmpeg -ss {start} -i {seed} -to {end} -i {seed} -filter_complex \
            \"[1]atrim=duration={start}[X];[1]atrim=start={end}[Y];[X][Y]concat=n=2:v=0:a=1[B]; \
            [0][B]concat=n=2:v=0:a=1\" \
            {output} 2> /dev/null".format(
            start = start, seed = seedfile, end = end, output = outputFile)
            mutationDetails = "Start %f, End %f, Length %f, Length of Cut %f, Added to the end" % (start, end, seedDuration, lenOfSection)
        else:
            command = "ffmpeg -ss {start} -i {seed} -to {end} -i {seed} -filter_complex \
            \"[1]atrim=duration={start}[X];[1]atrim=start={end}[Y];[X][Y]concat=n=2:v=0:a=1[B]; \
            [B][0]concat=n=2:v=0:a=1\" \
            {output} 2> /dev/null".format(
            start = start, seed = seedfile, end = end, output = outputFile)
            mutationDetails = "Start %f, End %f, Length %f, Length of Cut %f, Added to the front" % (start, end, seedDuration, lenOfSection)

    elif (Mutation.CONCAT == mutation):
        secondSeed = random.choice(seeds)
        command = "ffmpeg -i {seed} -i {seed2} -filter_complex [0:a][1:a]concat=n=2:v=0:a=1 \
            {output} 2> /dev/null".format(
            seed = seedfile, seed2 = secondSeed, output = outputFile)
        mutationDetails = "Combined with %s" % (secondSeed)

    elif (Mutation.REMOVE_BELOW_DECIBLE == mutation):
        belowDb = random.randint(-20, -10)
        command =  "ffmpeg -i {seed} -af silenceremove=stop_periods=-1:stop_duration=1:stop_threshold={below}dB \
            {output} 2> /dev/null".format(
            seed = seedfile, below = belowDb, output = outputFile)
        mutationDetails = "Below %d" % (belowDb)

    elif (Mutation.WHITE_NOISE == mutation):
        # Get length
        (sourceRate, sourceSig) = wav.read(seedfile)
        seedDuration = len(sourceSig) / float(sourceRate)

        sampleRate = random.randint(18000, 48000)
        amplitude = 0.1

        command = "ffmpeg -i {seed} -filter_complex \
            \"anoisesrc=d={seedDuration}:c=white:r={sampleRate}:a={amplitude}[A]; \
            [A]volume=.1[B]; \
            [0][B]amix=inputs=2:duration=shortest,volume=2\" \
            {output} 2> /dev/null".format(
            seed = seedfile, seedDuration = seedDuration, sampleRate = sampleRate, amplitude = amplitude, output = outputFile)
        mutationDetails = "Sample Rate %d, Amplitude %.2f" % (sampleRate, amplitude)

    elif (Mutation.REAL_WORLD_NOISE == mutation):
        # Get length
        (sourceRate, sourceSig) = wav.read(seedfile)
        seedDuration = len(sourceSig) / float(sourceRate)

        realNoise = "yt-audio/0SdAVK79lg.wav"

        command = "ffmpeg -i {realNoise} -to {seedDuration} -i {seed} -filter_complex \
            \"[0:a]volume=.1[A]; \
            [1][A]amix=inputs=2:duration=shortest,volume=2\" \
            {output} 2> /dev/null".format(
            seed = seedfile, seedDuration = seedDuration, realNoise = realNoise, output = outputFile)

    else:
        print('Mutant not supported')

    # print(command)
    # print()
    os.system(command)
    # print()

    return {ID: id, MUTATION: mutation, 
        OUTPUT_FILE: outputFile, 
        MUTATION_DETAILS: mutationDetails, 
        COMMAND: command, 
        SEED_FILE: seedfile}

# --------------------------------------------------------

def oracle(originalText, mutantText, mutation):

    if (Mutation.PITCH == mutation 
        or Mutation.SPEED == mutation 
        or Mutation.VOLUME == mutation):
        return originalText == mutantText

    elif (Mutation.LOOP == mutation):
        originalWords = originalText.split()
        newWords = mutantText.split()
        # original word count matches new word count / 2
        if (len(originalWords) != (len(newWords) / 2)):
            return False
        
        # Words match and are ordered correctly
        for i in range(len(originalWords)):
            if (originalWords[i] != newWords[i] or originalWords[i] != newWords[i + len(originalWords)]):
                return False
        
        return True

    elif (Mutation.CUT_SECTION == mutation 
        or Mutation.SUBSECTION == mutation 
        or Mutation.REMOVE_BELOW_DECIBLE == mutation
        or Mutation.REARRANGE == mutation):
        # All new words contained in the old words with allowed error
        originalWords = set(originalText.split())
        newWords = set(mutantText.split())
        # return newWords.issubset(originalWords)
        errors = 0
        for newWord in newWords:
            if newWord not in originalWords:
                errors += 1
        # Allow error of one word
        return errors <= 1

    elif (Mutation.CONCAT == mutation):
        originalWords = set(originalText.split())
        newWords = set(mutantText.split())
        return originalWords.issubset(newWords)

    elif (Mutation.REAL_WORLD_NOISE == mutation or Mutation.WHITE_NOISE == mutation):
        edit_distance = nltk.edit_distance(originalText, mutantText)
        return edit_distance < 10

    else:
        print('Mutant not supported')
        return False

# --------------------------------------------------------

def updateStatsSave(success, mutant):
    global failures
    global numMutations

    # Update the stats
    numMutations += 1

    dir = SUCCESS_DIR

    mutationCount[mutant[MUTATION].name] = mutationCount[mutant[MUTATION].name] + 1

    if (not success):
        failures += 1
        dir = FAILURE_DIR
        mutationFailureCount[mutant[MUTATION].name] = mutationFailureCount[mutant[MUTATION].name] + 1

    # Save mutation
    mutant[MUTATION] = mutant[MUTATION].name
    jsonFile = open("{0}/{1}.json".format(dir + "/" + mutant[MUTATION].lower(), mutant[ID]), "w")
    jsonFile.write(json.dumps(mutant, indent=4))
    jsonFile.close()

# --------------------------------------------------------

def fuzz():
    print("\n--------------------------------------------------------")
    print('Running Fuzz')
    fuzz_start = timer()

    # Select a random seed to mutate
    randomSeed = random.choice(seeds)

    # Get the resulting text of that seed
    start = timer()
    originalText = runModel(randomSeed)
    end = timer()
    print("%-20s : %4.2f | %s" % ("Original Text", end - start, originalText[0]))

    # Create a mutant from that seed
    start = timer()
    mutant = createMutant(randomSeed)
    end = timer()
    print("%-20s : %4.2f | %s %s" % ("Mutant Created", end - start, mutant[MUTATION], mutant[MUTATION_DETAILS]))

    # Run model on mutant
    start = timer()
    mutantText = runModel(mutant[OUTPUT_FILE])
    end = timer()
    print("%-20s : %4.2f | %s" % ("Mutant Text", end - start, mutantText[0]))

    success = oracle(originalText[0], mutantText[0], mutant[MUTATION])

    mutant[ORIGINAL_TEXT] = originalText[0]
    mutant[ORIGINAL_CONFIDENCE] = originalText[1]
    mutant[MUTATION_TEXT] = mutantText[0]
    mutant[MUTATION_CONFIDENCE] = mutantText[1]

    updateStatsSave(success, mutant)

    fuzz_end = timer() - fuzz_start

    # Print final Results
    print()
    print("%-20s: %s" % ("Id", mutant[ID]))
    print("%-20s: %4.2f" % ("Time", fuzz_end))
    print("%-20s: %s" % ("Mutation", mutant[MUTATION]))
    print("%-20s: %s" % ("Mutation Details", mutant[MUTATION_DETAILS]))
    print("%-20s: %s" % ("Original Text", mutant[ORIGINAL_TEXT]))
    print("%-20s: %s" % ("Mutated Text", mutant[MUTATION_TEXT]))
    print("%-20s: %s" % ("Original Confidence", mutant[ORIGINAL_CONFIDENCE]))
    print("%-20s: %s" % ("Mutant Confidence", mutant[MUTATION_CONFIDENCE]))
    print("%-20s: %s" % ("Source", mutant[SEED_FILE]))
    print("%-20s: PASSED" % ("Oracle")) if success else print("%-20s: FAILED" % ("Oracle"))

# --------------------------------------------------------

def collectFinalResults(time):
    
    print("\n\n--------------------------------------------------------\n")
    print("Stopped")
    print("Ran for: %.2f" % (time))
    print("\n--------------------------------------------------------\n\n")
    print("Final Results:")

    percentFailures = 0
    if (numMutations > 0):
        percentFailures = (failures / numMutations) * 100

    print("\t|%s|%s|" % ("-" * 23, "-" * 7))
    print("\t| %-20s: | %5d |" % ("Seeds Provided", len(seeds)))
    print("\t| %-20s: | %5d |" % ("Mutations Attempted", numMutations))
    print("\t| %-20s: | %5d |" % ("Failures", failures))
    print("\t| %-20s: | %4.2d%% |" % ("Percent of failures", percentFailures))
    print("\t|%s|%s|\n" % ("-" * 23, "-" * 7))

    print("Mutation Results:")

    print("\t|------------------------------------------------|")
    print("\t| %-20s | %6s | %6s | %5s |" % ("Mutant Name", "Errors", "Count", "%"))
    print("\t|----------------------|--------|--------|-------|")
    for mutant in mutantsEnabled:
        percentMutant = 0
        if (mutationCount[mutant.name] > 0):
            percentMutant = (mutationFailureCount[mutant.name] / mutationCount[mutant.name]) * 100
        
        print("\t| %-20s | %6d | %6d | %4.2d%% |" % (mutant.name, mutationFailureCount[mutant.name], mutationCount[mutant.name], percentMutant))
    print("\t|------------------------------------------------|")

    print("\n\n")

    finalResults = {"failures": failures, "mutations" : numMutations, "percentFailures": percentFailures, "mutantCount": mutationCount, "mutantFailCount": mutationFailureCount}
    jsonFile = open("{0}/finalResults.json".format(RESULTS_DIR), "w")
    jsonFile.write(json.dumps(finalResults, indent=4))
    jsonFile.close()

# --------------------------------------------------------

def main():
    print('Starting Garrett Christian\'s DeepSpeech Audio Fuzzing Tool')
    setup()
    run_start = timer()

    try:
        print("Starting Mutations")
        while True:
            fuzz()
        
    except KeyboardInterrupt:
        run_end = timer()
        collectFinalResults(run_end - run_start)

if __name__ == '__main__':
    main()