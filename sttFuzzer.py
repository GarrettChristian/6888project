import os
import uuid
import random
import shutil
import argparse
from pathlib import Path
import json
import wave
from Levenshtein import distance as lev
import sys
import numpy as np
import threading

import scipy.io.wavfile as wav

from timeit import default_timer as timer
from deepspeech import Model, version
from enum import Enum

# Enum of the different types of mutations supported
class Mutation(Enum):
    PITCH = "PITCH"              # changes the pitch
    SPEED = "SPEED"              # picks a new speed either slower or faster
    VOLUME = "VOLUME"            # picks a new volume either quieter or louder
    LOOP = "LOOP"                # plays the seed twice
    CONCAT = "CONCAT"            # takes the seed and adds a ranom seed as the second half
    SUBSECTION = "SUBSECTION"    # picks a random subsection of the seed from of length at least 1/4 of the seed len
    CUT_SECTION = "CUT_SECTION"  # removes a section of the seed of a least at least 1/4 of the seed len
    REARRANGE = "REARRANGE"      # moves a section of the seed of at least 1/4 of the seed len
    REMOVE_BELOW_DECIBLE = "REMOVE_BELOW_DECIBEL"   # removes audio below (-10 through -20)
    WHITE_NOISE = "WHITE_NOISE"                     # adds white noise to the audio
    REAL_WORLD_NOISE = "REAL_WORLD_NOISE"           # adds real world noise to the audio
    VIBRATO = "VIBRATO"                             # vibrato: Sinusoidal phase modulation
    BASE = "BASE"                                   # Boost or cut the bass (lower) frequencies of the audio 
    TREBLE = "TREBLE"                               # Boost or cut treble (upper) frequencies of the audio

# Output directory constants
RESULTS_DIR = "gcAudioResults"
AUDIO_OUTPUT_DIR = RESULTS_DIR + "/output"
SUCCESS_DIR = RESULTS_DIR + "/success"
FAILURE_DIR = RESULTS_DIR + "/failure"

# Global seeds and model
seeds = []
modelLocation = ""
scorerLocation = ""
mutantsEnabled = []
realWorldNoise = []
results = []

# Global results
numMutations = 0
failures = 0
mutationCount = {}
mutationFailureCount = {}
saveCount = 0
saveAll = False
threads = 0

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
PASSED = "passed"

# --------------------------------------------------------

"""
printMutants

Helper to print mutant enum values

@param mutants list of enum values to print
"""
def printMutants(mutants):
    for i in range(len(mutants)):
        if len(mutants) == i + 1:
            print("%s" % (mutants[i].name), end='')
        else:
            print("%s, " % (mutants[i].name), end='')
    print()

# --------------------------------------------------------

"""
formatSecondsToHhmmss

Helper to convert seconds to hours minutes and seconds

@param seconds
@return formatted string of hhmmss
"""
def formatSecondsToHhmmss(seconds):
    hours = seconds / (60*60)
    seconds %= (60*60)
    minutes = seconds / 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

# --------------------------------------------------------

"""
setup

Reads in the arguments from the command line and sets up all global arguments
"""
def setup():
    global results
    global seeds
    global mutationCount
    global mutationFailureCount
    global mutantsEnabled
    global saveCount
    global saveAll
    global realWorldNoise
    global threads
    global modelLocation
    global scorerLocation

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Running Garrett Christian\'s Audio Fuzzing Tool')
    parser.add_argument('--model', required=True, 
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', required=True,
                        help='Path to the external scorer file')
    parser.add_argument('--seeds', required=True,
                        help='Path to the seed files, NOTE these seeds must be .wav files')
    parser.add_argument('--realWorldNoise', required=True,
                        help='Path to the real world noise files, NOTE these files must be .wav files')
    parser.add_argument('--threads', required=False, default=1, type=int,
                        help='Path to the real world noise files, NOTE these files must be .wav files')
    parser.add_argument('--mutations', required=False,
                        help='Mutations to perform comma seperated example: PITCH,SPEED')
    parser.add_argument('--save', required=False, default=10, type=int,
                        help='Amount to save of failed runs')
    parser.add_argument('--saveAll', required=False, action='store_true', default=False,
                        help='Saves all ouput')
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
    modelLocation = args.model
    print('Loading model from file {}'.format(args.model))

    # Set up the scorer
    scorerLocation = args.scorer
    print('Loading scorer from files {}'.format(args.scorer))

    # Get the file names of the seeds
    print("Seed Directory provided: {0}".format(args.seeds))
    for path in Path(args.seeds).rglob('*.wav'):
        seeds.append(str(path))
    print("seeds provided {0}".format(len(seeds)))

    # Get the file names of the realWorldNoise
    print("Real World Audio Seed Directory provided: {0}".format(args.realWorldNoise))
    for path in Path(args.realWorldNoise).rglob('*.wav'):
        realWorldNoise.append(str(path))
    print("Real world noise provided {0}".format(len(realWorldNoise)))
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
    print()

    # How much of the results should be saved
    saveAll = args.saveAll
    saveCount = args.save
    if (saveAll):
        print("Saving all results")
    else:
        print("Saving the first %d error results for each mutation" % (saveCount))
    print()
    
    # Get number of threads to run
    threads = args.threads
    print("Starting %d threads" % (threads))
    for i in range(threads):
        results.append(None)

    # Set up the results directories
    for mutant in mutantsEnabled:
        makeDirSuccess = SUCCESS_DIR + "/" + mutant.name.lower()
        makeDirFailure = FAILURE_DIR + "/" + mutant.name.lower()
        makeDirOutput = AUDIO_OUTPUT_DIR + "/" + mutant.name.lower()
        os.makedirs(makeDirSuccess, exist_ok=True)
        os.makedirs(makeDirFailure, exist_ok=True)
        os.makedirs(makeDirOutput, exist_ok=True)

# --------------------------------------------------------

"""
runModel

Runs the provided model on the provided audio

@param audioFile string path to the audio
@param ds deepspeech model to use on the audio
@return a tuple with the text and model confidence 
"""
def runModel(audioFile, ds):
    fin = wave.open(audioFile, 'rb')
    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    fin.close()

    metadata = ds.sttWithMetadata(audio, 1).transcripts[0]

    text = ''.join(token.text for token in metadata.tokens)
    
    return (text, metadata.confidence)

# --------------------------------------------------------

"""
createMutant

Randomly selects a mutant from the enabled mutants then uses FFMPEG to create a mutant audio

@param seedfile string path to the seed audio to create the mutant from
@return a dictionary with all the new mutants information
"""
def createMutant(seedfile):
    #  Select random Mutation
    mutation = random.choice(list(mutantsEnabled))

    id = str(uuid.uuid4())
    outputFile = "{audioOutDir}/{id}.wav".format(audioOutDir = AUDIO_OUTPUT_DIR + "/" + mutation.name.lower(), id = id)
    command = ""
    mutationDetails = ""

    # print('Mutant Selected %s %s' % (mutation.name, seedfile))

    if (Mutation.PITCH == mutation):
        # Set the output sample rate. Default is 44100 Hz.
        mutiplier = random.uniform(.8, 1)
        rate = 44100 * mutiplier
        command = "ffmpeg -i {seed} -af \"asetrate={rate}\" -y {output} 2> /dev/null".format(
            seed = seedfile, rate = rate, output = outputFile)
        mutationDetails = "Rate %.2f" % (rate)

    elif (Mutation.SPEED == mutation):
        speed = random.uniform(.5, .8)
        if random.randint(1, 2) % 2 == 0:
            speed = random.uniform(1.2, 1.5)
        
        command = "ffmpeg -i {seed} -filter:a \"atempo={speed}\" -vn {output} 2> /dev/null".format(
            seed = seedfile, speed = speed, output = outputFile)
        mutationDetails = "Speed %.2f" % (speed)

    elif (Mutation.VOLUME == mutation):
        volume = random.uniform(.3, .8)
        if random.randint(1, 2) % 2 == 0:
            volume = random.uniform(1.2, 1.7)
        
        command = "ffmpeg -i {seed} -af \"volume={volume}\" {output} 2> /dev/null".format(
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

        # Pick start and end
        # Choose random number between 1 & len - 1
        # Leaves at minimum a second and max length - a second
        lenOfSection = random.uniform((seedDuration * (1/4)), seedDuration - (seedDuration * (1/4)))
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
        lenOfCut = random.uniform((seedDuration * (1/4)), seedDuration - (seedDuration * (1/4)))
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

        # Pick start and end
        # Choose random number between 1 & len - 1
        # Leaves at minimum a second and max length - a second
        lenOfSection = random.uniform((seedDuration * (1/3)), seedDuration - (seedDuration * (1/3)))
        # Choose random number between 0 & length - length of the cut

        # Add at the front
        if random.randint(1, 2) % 2 == 0:
            start = random.uniform(0, seedDuration - lenOfSection - (seedDuration * (1/8)))
            end = start + lenOfSection
            command = "ffmpeg -i {seed} -i {seed} -filter_complex \
            \"[0]atrim={start}:{end}[A]; \
            [1]atrim=duration={start}[X];[1]atrim=start={end}[Y];[X][Y]concat=n=2:v=0:a=1[B]; \
            [B][A]concat=n=2:v=0:a=1\" \
            {output} 2> /dev/null".format(
            start = start, seed = seedfile, end = end, output = outputFile)
            mutationDetails = "Start %f, End %f, Length %f, Length of Cut %f, Added to the end" % (start, end, seedDuration, lenOfSection)
        # Add at the end
        else:
            start = random.uniform((seedDuration * (1/8)), seedDuration - lenOfSection)
            end = start + lenOfSection
            command = "ffmpeg -i {seed} -i {seed} -filter_complex \
            \"[0]atrim={start}:{end}[A]; \
            [1]atrim=duration={start}[X];[1]atrim=start={end}[Y];[X][Y]concat=n=2:v=0:a=1[B]; \
            [A][B]concat=n=2:v=0:a=1\" \
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

        realNoise = random.choice(realWorldNoise)

        command = "ffmpeg -i {realNoise} -to {seedDuration} -i {seed} -filter_complex \
            \"[0:a]volume=.1[A]; \
            [1][A]amix=inputs=2:duration=shortest,volume=2\" \
            {output} 2> /dev/null".format(
            seed = seedfile, seedDuration = seedDuration, realNoise = realNoise, output = outputFile)
        mutationDetails = "Added %s" % (realNoise)

    elif (Mutation.TREBLE == mutation 
        or Mutation.BASE == mutation):
        
        # Boost or cut -20 (for a large cut) to +20 (for a large boost)
        gain = 0
        type = "cut"
        if random.randint(1, 2) % 2 == 0:
            gain = random.randint(-25, -18)
        else:
            gain = random.randint(18, 25)
            type = "boost"
        if (Mutation.TREBLE == mutation):
            command = "ffmpeg -i {seed} -af \"treble=g={gain}\" {output} 2> /dev/null".format(
                seed = seedfile, gain = gain, output = outputFile)
        else:
            command = "ffmpeg -i {seed} -af \"lowshelf=g={gain}\" {output} 2> /dev/null".format(
                seed = seedfile, gain = gain, output = outputFile)
        mutationDetails = "Gain %d (%s)" % (gain, type)

    elif (Mutation.VIBRATO == mutation):
        # Modulation frequency in Hertz. Range is 0.1 - 20000.0. Default value is 5.0 Hz.
        frequency = random.uniform(5, 8)
        # Depth of modulation as a percentage. Range is 0.0 - 1.0. Default value is 0.5.
        depth = random.uniform(.5, .8)

        command = "ffmpeg -i {seed} -filter_complex \"vibrato=f={frequency}:d={depth}\" {output} 2> /dev/null".format(
            seed = seedfile, frequency = frequency, depth = depth, output = outputFile)
        mutationDetails = "Frequency %.2f, Depth %.2f" % (frequency, depth)

    else:
        print('Mutant not supported')


    if (command != ""):
        os.system(command)
    else:
        outputFile = seedfile

    return {
        ID: id, 
        MUTATION: mutation, 
        OUTPUT_FILE: outputFile, 
        MUTATION_DETAILS: mutationDetails, 
        COMMAND: command, 
        SEED_FILE: seedfile
        }

# --------------------------------------------------------

"""
oracle

Determines if the mutant passes the oracle based on the mutation used

@param originalText string from the original audio
@param mutantText string from the mutant audio
@param mutation that was used to create the mutant text
@return pass or fail based on the oracle 
"""
def oracle(originalText, mutantText, mutation):

    if (Mutation.PITCH == mutation 
        or Mutation.SPEED == mutation 
        or Mutation.VOLUME == mutation
        or Mutation.TREBLE == mutation
        or Mutation.BASE == mutation):
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
        or Mutation.REARRANGE == mutation
        or Mutation.VIBRATO == mutation):
        # All new words contained in the old words with allowed error
        originalWords = set(originalText.split())
        newWords = set(mutantText.split())
        # return newWords.issubset(originalWords)
        errors = 0
        for newWord in newWords:
            if newWord not in originalWords:
                errors += 1
        
        # Allow error of two words for REARRANGE one for the rest
        if (Mutation.REARRANGE == mutation):
            return errors <= 2
        return errors <= 1

    elif (Mutation.CONCAT == mutation):
        originalWords = set(originalText.split())
        newWords = set(mutantText.split())
        return originalWords.issubset(newWords)

    elif (Mutation.REAL_WORLD_NOISE == mutation or Mutation.WHITE_NOISE == mutation):
        edit_distance = lev(originalText, mutantText)
        return edit_distance < 10

    else:
        print('Mutant not supported')
        return False

# --------------------------------------------------------

"""
fuzz

The function that all the threads are running
Sets up the model, scorer and results, then runs the fuzz until terminated
Fuzz: get original text, create mutant audio, get mutant text, compare with oracle, update and print results
Once terminated the thread saves it's individual results to a shared global array

@param event that will tell the thread when to conclude
@param threadId int specifier of the thread
@param mutex to protect the output
"""
def fuzz(event, threadId, mutex):
    global modelLocation
    global scorerLocation
    global saveAll
    global saveCount
    global results
    global mutantsEnabled

    # Set up this threads model & scorer
    model_load_start = timer()
    ds = Model(modelLocation)
    model_load_end = timer() - model_load_start

    scorer_load_start = timer()
    ds.enableExternalScorer(scorerLocation)
    scorer_load_end = timer() - scorer_load_start
    
    mutex.acquire()
    print("Thread %d" % (threadId))
    print('Loaded model in {:.3}s.'.format(model_load_end))
    print('Loaded scorer in {:.3}s.'.format(scorer_load_end))
    print()
    mutex.release()

    # Set up the result counters for this thread
    mutationCount = {}
    mutationFailureCount = {}
    failures = 0
    numMutations = 0

    for mutant in mutantsEnabled:
        mutationCount[mutant.name] = 0
        mutationFailureCount[mutant.name] = 0

    # Attempt different mutations until the process is killed
    while not event.is_set():
        fuzz_start = timer()

        # Select a random seed to mutate
        randomSeed = random.choice(seeds)

        # Get the resulting text of that seed
        startModel = timer()
        originalText = runModel(randomSeed, ds)
        endModel = timer()

        # Create a mutant from that seed
        startCreateMutant = timer()
        mutant = createMutant(randomSeed)
        endCreateMutant = timer()

        # Run model on mutant
        startModelMutant = timer()
        mutantText = runModel(mutant[OUTPUT_FILE], ds)
        endModelMutant = timer()

        # Check if there was an error
        success = oracle(originalText[0], mutantText[0], mutant[MUTATION])

        mutant[ORIGINAL_TEXT] = originalText[0]
        mutant[ORIGINAL_CONFIDENCE] = originalText[1]
        mutant[MUTATION_TEXT] = mutantText[0]
        mutant[MUTATION_CONFIDENCE] = mutantText[1]

        # Update stats
        numMutations += 1
        mutationCount[mutant[MUTATION].name] = mutationCount[mutant[MUTATION].name] + 1

        # update failure stats and select the location to save
        dir = SUCCESS_DIR
        if (not success):
            failures += 1
            dir = FAILURE_DIR
            mutationFailureCount[mutant[MUTATION].name] = mutationFailureCount[mutant[MUTATION].name] + 1

        # Save mutation (only if enabled)
        if ((mutationCount[mutant[MUTATION].name] <= saveCount and not success) 
            or saveAll):
            mutant[MUTATION] = mutant[MUTATION].name
            mutant[PASSED] = success
            jsonFile = open("{0}/{1}.json".format(dir + "/" + mutant[MUTATION].lower(), mutant[ID]), "w")
            jsonFile.write(json.dumps(mutant, indent=4))
            jsonFile.close()
        else:
            mutant[MUTATION] = mutant[MUTATION].name
            os.remove(mutant[OUTPUT_FILE])

        fuzz_end = timer() - fuzz_start

        # Prepare the final results
        resultText = "\n--------------------------------------------------------\n"
        resultText += "Thread %d\n" % (threadId)
        resultText += "%-20s: %s\n" % ("Id", mutant[ID])
        resultText += "%-20s: %s\n" % ("Total time", formatSecondsToHhmmss(fuzz_end))
        resultText += "%-20s: %d\n" % ("Current count", numMutations)
        resultText += "%-20s: time: %s | %s %s\n" % ("Mutant Created", formatSecondsToHhmmss(endCreateMutant - startCreateMutant), mutant[MUTATION], mutant[MUTATION_DETAILS])
        resultText += "%-20s: time: %s | %s\n" % ("Original Text", formatSecondsToHhmmss(endModel - startModel), originalText[0])
        resultText += "%-20s: time: %s | %s\n" % ("Mutant Text", formatSecondsToHhmmss(endModelMutant - startModelMutant), mutantText[0])
        resultText += "%-20s: %s\n" % ("Original Confidence", mutant[ORIGINAL_CONFIDENCE])
        resultText += "%-20s: %s\n" % ("Mutant Confidence", mutant[MUTATION_CONFIDENCE])
        resultText += "%-20s: %s\n" % ("Source", mutant[SEED_FILE])
        if success:
            resultText += "%-20s: PASSED\n" % ("Oracle")
        else: 
            resultText += "%-20s: FAILED\n" % ("Oracle")

        # Print final Results
        mutex.acquire()
        print(resultText)
        mutex.release()

    # Once the process has been signaled to end collect all results in the global results array
    mutex.acquire()
    results[threadId] = (numMutations, failures, mutationCount, mutationFailureCount)
    mutex.release()

# --------------------------------------------------------

"""
collectFinalResults

Aggregates the final results from the threads, prints the final metrics,
and saves them to an output file

@param time float of how much time the 
@param results array of individual thread fuzzing metrics 
"""
def collectFinalResults(time, results):

    # Aggregate results
    numMutations = 0
    failures = 0
    mutationCount = {}
    mutationFailureCount = {}

    for mutant in mutantsEnabled:
        mutationCount[mutant.name] = 0
        mutationFailureCount[mutant.name] = 0

    for r in results:
        numMutations += r[0]
        failures += r[1]
        for mutant in mutantsEnabled:
            mutationCount[mutant.name] += r[2][mutant.name]
            mutationFailureCount[mutant.name] += r[3][mutant.name]
    
    # Print time
    print("\n\n--------------------------------------------------------\n")
    print("Stopped")
    print("Ran for: %s" % (formatSecondsToHhmmss(time)))
    print("\n--------------------------------------------------------\n\n")
    print("Final Results:")

    percentFailures = 0
    if (numMutations > 0):
        percentFailures = (failures / numMutations) * 100

    # Print general information
    print("\t|%s|%s|" % ("-" * 23, "-" * 7))
    print("\t| %-20s: | %5d |" % ("Seeds Provided", len(seeds)))
    print("\t| %-20s: | %5d |" % ("Mutations Attempted", numMutations))
    print("\t| %-20s: | %5d |" % ("Failures", failures))
    print("\t| %-20s: | %4.2d%% |" % ("Percent of Failures", percentFailures))
    print("\t|%s|%s|\n" % ("-" * 23, "-" * 7))

    # Print mutant information
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

    # Write out the final results
    finalResults = {"failures": failures, "mutations" : numMutations, "percentFailures": percentFailures, "mutantCount": mutationCount, "mutantFailCount": mutationFailureCount}
    jsonFile = open("{0}/finalResults.json".format(RESULTS_DIR), "w")
    jsonFile.write(json.dumps(finalResults, indent=4))
    jsonFile.close()

# --------------------------------------------------------

"""
main

Runs the set up, starts all the threads to perform their fuzzing, 
waits for the kill signal, joins the threads, then outputs the results
"""
def main():
    global results

    # Collect arguments set up mutex
    print('Starting Garrett Christian\'s DeepSpeech Audio Fuzzing Tool')
    setup()
    run_start = timer()
    mutex = threading.Lock()
    print("\n--------------------------------------------------------")
    print("Starting threads and perfoming setup\n")

    # Start the threads providing them with an event to be triggered when the process is signaled to end
    try:
        event = threading.Event()
        threadList = []
        for i in range(threads):
            thread = threading.Thread(target=fuzz, args=(event, i, mutex, ))
            threadList.append(thread)
            thread.start()
        event.wait()  # wait forever but without blocking KeyboardInterrupt exceptions
            
    except KeyboardInterrupt:
        event.set()  # inform the child thread that it should exit
        mutex.acquire()
        print("\n--------------------------------------------------------")
        print("Ctrl+C pressed...")
        print("Waiting for other processes to conclude then collecting results\n")
        mutex.release()
        for thread in threadList:
            thread.join()

        run_end = timer()
        collectFinalResults(run_end - run_start, results)


if __name__ == '__main__':
    main()
