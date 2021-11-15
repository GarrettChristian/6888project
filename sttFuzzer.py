import os
import uuid
import random
import time
import shutil
import argparse
import glob
import json
import wave
import shlex
import subprocess
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
    WHITE_NOISE = "WHITE_NOISE"
    LOOP = "LOOP"
    SUBSECTION = "SUBSECTION"    # picks a random subsection of the seed from of length at least 1 second
    CUT_SECTION = "CUT_SECTION"  # removes a section of the clip of a least 1 second in length
    COMBINE = "COMBINE"
    REARANGE = "REARANGE"
    REMOVE_BELOW_DECIBLE = "REMOVE_BELOW_DECIBLE"
    REAL_WORLD_NOISE = "REAL_WORLD_NOISE"

# Output directory constants
RESULTS_DIR = "gcAudioResults"
AUDIO_OUTPUT_DIR = RESULTS_DIR + "/output"
SUCCESS_DIR = RESULTS_DIR + "/success"
FAILURE_DIR = RESULTS_DIR + "/failure"

# Global seeds and model
seeds = []
ds = None

# Global results
mutations = 0
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

# --------------------------------------------------------

def setup():
    global ds
    global seeds
    global mutationCount
    global mutationFailureCount

    # Parse the arguments
    parser = argparse.ArgumentParser(description='Running Garrett Christian\'s Audio Fuzzing Tool')
    parser.add_argument('--model', required=True, 
                        help='Path to the model (protocol buffer binary file)')
    parser.add_argument('--scorer', required=True,
                        help='Path to the external scorer file')
    parser.add_argument('--seeds', required=True,
                        help='Path to the seed files')
    args = parser.parse_args()

    print("\n--------------------------------------------------------")
    print("Setup")

    # Set up the output folders
    # Remove the old output folder
    try:
        shutil.rmtree(RESULTS_DIR)
        print("Resetting results folder")
    except FileNotFoundError:
        print("Setting up new results folder")
    # Create the new result folders
    os.mkdir(RESULTS_DIR)
    os.mkdir(AUDIO_OUTPUT_DIR)
    os.mkdir(SUCCESS_DIR)
    os.mkdir(FAILURE_DIR)

    # Set up the model
    print('Loading model from file {}'.format(args.model))
    model_load_start = timer()
    # sphinx-doc: python_ref_model_start
    ds = Model(args.model)
    # sphinx-doc: python_ref_model_stop
    model_load_end = timer() - model_load_start
    print('Loaded model in {:.3}s.'.format(model_load_end))

    # Set up the scorer
    print('Loading scorer from files {}'.format(args.scorer))
    scorer_load_start = timer()
    ds.enableExternalScorer(args.scorer)
    scorer_load_end = timer() - scorer_load_start
    print('Loaded scorer in {:.3}s.'.format(scorer_load_end))

    desired_sample_rate = ds.sampleRate()

    # Get the file names of the seeds
    print("Seed Directory provided: {0}".format(args.seeds))
    seeds = glob.glob(args.seeds + '/*.wav')

    # Set up the results
    for mutant in Mutation:
        mutationCount[mutant.name] = 0
        mutationFailureCount[mutant.name] = 0

# --------------------------------------------------------
# MODEL SPECIFIC METHODS

# def convert_samplerate(audio_path, desired_sample_rate):
#     sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(quote(audio_path), desired_sample_rate)
#     try:
#         output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
#     except subprocess.CalledProcessError as e:
#         raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
#     except OSError as e:
#         raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))

#     return desired_sample_rate, np.frombuffer(output, np.int16)

# def metadata_to_string(metadata):
#     return ''.join(token.text for token in metadata.tokens)

def runModel(audioFile):
    # print("running model")

    fin = wave.open(audioFile, 'rb')
    # fs_orig = fin.getframerate()
    # if fs_orig != ds.sampleRate():
    #     print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(fs_orig, ds.sampleRate()))
    #     fs_new, audio = convert_samplerate(audioFile, ds.sampleRate())
    # else:
    audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    # audio_length = fin.getnframes() * (1/fs_orig)
    fin.close()

    # print(ds.sttWithMetadata(audio, 1).transcripts[0])
    # print(ds.stt(audio))

    metadata = ds.sttWithMetadata(audio, 1).transcripts[0]

    text = ''.join(token.text for token in metadata.tokens)
    
    return (text, metadata.confidence)
    # sphinx-doc: python_ref_inference_stop
    # inference_end = timer() - inference_start
    # print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))

# --------------------------------------------------------

def createMutant(seedfile):
    #  Select random Mutation
    # mutation = random.choice(list(Mutation))
    mutation = Mutation.CUT_SECTION

    id = str(uuid.uuid4())
    outputFile = "{audioOutDir}/{id}.wav".format(audioOutDir = AUDIO_OUTPUT_DIR, id = id)
    command = ""
    mutationDetails = ""

    print('Mutant Selected %s %s' % (mutation.name, seedfile))

    if (Mutation.PITCH == mutation):
        command = "ffmpeg -i {seed} -af \"asetrate=44100*0.9\" -y {output} 2> /dev/null".format(seed = seedfile, output = outputFile)

    elif (Mutation.SPEED == mutation):
        command = "ffmpeg -i {seed} -filter:a \"atempo=1.5\" -vn {output}  2> /dev/null".format(seed = seedfile, output = outputFile)

    elif (Mutation.VOLUME == mutation):
        command = "ffmpeg -i {seed} -af \"volume=0.5\" {output}  2> /dev/null".format(seed = seedfile, output = outputFile)

    elif (Mutation.LOOP == mutation):
        command = "ffmpeg -i {seed} -filter_complex \"[0:a]afifo[a0];[0:a]afifo[a1];[a0][a1]concat=n=2:v=0:a=1[a]\" -map \"[a]\" {output} 2> /dev/null".format(seed = seedfile, output = outputFile)

    elif (Mutation.SUBSECTION == mutation):
        # Get length
        (sourceRate, sourceSig) = wav.read(seedfile)
        seedDuration = len(sourceSig) / float(sourceRate)

        # Pick start and end
        # Choose random number between 1 & len - 1
        # Leaves at minimum a second and max length - a second
        lenOfSection = random.uniform(1, seedDuration - 1)
        # Choose random number between 0 & length - length of the cut
        start = random.uniform(0, seedDuration - lenOfSection)
        end = start + lenOfSection
        command = "ffmpeg -ss {start} -i {seed} -to {end} -c copy {output}".format(start = start, seed = seedfile, end = end, output = outputFile)
        mutationDetails = "Start %d, End %d, Length %d, Length of Subsection %d" % (start, end, seedDuration, lenOfSection)

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
        command = "ffmpeg -i {seed} -filter_complex \"[0]atrim=duration={start}[a];[0]atrim=start={end}[b];[a][b]concat=n=2:v=0:a=1\" {output} 2> /dev/null".format(seed = seedfile, start = startOfCut, end = endOfCut, output = outputFile)
        mutationDetails = "Start %d, End %d, Length %d, Length of Cut %d" % (startOfCut, endOfCut, seedDuration, lenOfCut)

    elif (Mutation.COMBINE == mutation):
        # TODO
        secondSeed = random.choice(seeds)
        print('Mutant Selected COMBINE')
        # command = "ffmpeg -filter_complex \"{seed} [a0]; amovie={seed2} [a1]; [a0][a1] amix=inputs=2:duration=shortest [aout]\" -map [aout] -acodec {output}".format(seed = seedfile, seed2 = secondSeed, output = outputFile)
        command = "ffmpeg -i \"concat:{seed}|{seed2}\" -codec copy {output}".format(seed = seedfile, seed2 = secondSeed, output = outputFile)

    elif (Mutation.WHITE_NOISE == mutation):
        # TODO taking a long time last attempt:
        # ffmpeg -i /Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/6888project/seeds/4507-16021-0012.wav -filter_complex \ "aevalsrc=-2+random(0)" -t 5 gcAudioResults/output/2dcdb5d6-fd5b-4cab-b5d9-18774a0dd486.wav
        # Produced silent file
        command = "ffmpeg -i {seed} -filter_complex aevalsrc=\'-2+random(0)\'  {output}".format(seed = seedfile, output = outputFile)
        # print('Mutant Selected WHITE_NOISE')

    elif (Mutation.REARANGE == mutation):
        # TODO
        command = "ffmpeg -i {seed} -af \"asetrate=44100*0.9\" -y {output} 2> /dev/null".format(seed = seedfile, output = outputFile)
        
    elif (Mutation.REMOVE_BELOW_DECIBLE == mutation):
        # TODO
        command = "ffmpeg -i {seed} -af \"asetrate=44100*0.9\" -y {output} 2> /dev/null".format(seed = seedfile, output = outputFile)

    elif (Mutation.REAL_WORLD_NOISE == mutation):
        # TODO
        command = "ffmpeg -i {seed} -af \"asetrate=44100*0.9\" -y {output} 2> /dev/null".format(seed = seedfile, output = outputFile)

    else:
        print('Mutant not supported')

    print(command)
    os.system(command)

    return {ID: id, MUTATION: mutation, OUTPUT_FILE: outputFile, MUTATION_DETAILS: mutationDetails, COMMAND: command}

# --------------------------------------------------------

def oracle(originalText, mutantText, mutation):

    if (Mutation.PITCH == mutation or Mutation.SPEED == mutation or Mutation.VOLUME == mutation):
        return originalText == mutantText

    elif (Mutation.WHITE_NOISE == mutation):
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

    elif (Mutation.CUT_SECTION == mutation or Mutation.SUBSECTION == mutation):
        # All new words contained in the old words
        originalWords = set(originalText.split())
        newWords = set(mutantText.split())
        return newWords.issubset(originalWords)

    elif (Mutation.COMBINE == mutation):
        return originalText == mutantText

    elif (Mutation.REARANGE == mutation):
        return originalText == mutantText

    elif (Mutation.REMOVE_BELOW_DECIBLE == mutation):
        return originalText == mutantText

    elif (Mutation.REAL_WORLD_NOISE == mutation):
        return originalText == mutantText

    else:
        print('Mutant not supported')
        return False

# --------------------------------------------------------

def updateStatsSave(success, mutant):
    global failures
    global mutations

    mutations += 1

    dir = SUCCESS_DIR

    mutationCount[mutant[MUTATION].name] = mutationCount[mutant[MUTATION].name] + 1

    if (not success):
        failures += 1
        dir = FAILURE_DIR
        mutationFailureCount[mutant[MUTATION].name] = mutationFailureCount[mutant[MUTATION].name] + 1

    mutant[MUTATION] = mutant[MUTATION].name
    jsonFile = open("{0}/{1}.json".format(dir, mutant[ID]), "w")
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
    originalText = runModel(randomSeed)

    # Create a mutant from that seed
    mutant = createMutant(randomSeed)

    mutantText = runModel(mutant[OUTPUT_FILE])

    success = oracle(originalText[0], mutantText[0], mutant[MUTATION])

    mutant[ORIGINAL_TEXT] = originalText[0]
    mutant[ORIGINAL_CONFIDENCE] = originalText[1]
    mutant[MUTATION_TEXT] = mutantText[0]
    mutant[MUTATION_CONFIDENCE] = mutantText[1]

    updateStatsSave(success, mutant)

    fuzz_end = timer() - fuzz_start
    print('Time: %4.2f | Original: %-30s | Mutated: %-30s | Mutation: %-15s |  success=%d |  Id: %s' % (fuzz_end, mutant[ORIGINAL_TEXT], mutant[MUTATION_TEXT], mutant[MUTATION], success, mutant[ID]))

# --------------------------------------------------------

def collectFinalResults():
    print("\n\n--------------------------------------------------------")
    print("Final Results:")

    percentFailures = 0
    if (mutations > 0):
        percentFailures = (failures / mutations) * 100

    print("%-30s | %4d" % ("Number of seeds provided", len(seeds)))
    print("%-30s | %4d" % ("Mutations Attempted", mutations))
    print("%-30s | %4d" % ("Failures", failures))
    print("%-30s | %4.2d%%" % ("Percent of failures:", percentFailures))

    print("\n--------------------------------------------------------")
    print("Mutation Results:")

    print("\t|------------------------------------------------|")
    print("\t| %-20s | %-6s | %-6s | %-5s |" % ("Mutant Name", "Errors", "Count", "%"))
    print("\t|----------------------|--------|--------|-------|")
    for mutant in Mutation:
        percentMutant = 0
        if (mutationCount[mutant.name] > 0):
            percentMutant = (mutationFailureCount[mutant.name] / mutationCount[mutant.name]) * 100
        
        print("\t| %-20s | %6d | %6d | %4.2d%% |" % (mutant.name, mutationFailureCount[mutant.name], mutationCount[mutant.name], percentMutant))
    print("\t|------------------------------------------------|")

    print("\n\n")

    finalResults = {"failures": failures, "mutations" : mutations, "percentFailures": percentFailures, "mutantCount": mutationCount, "mutantFailCount": mutationFailureCount}
    jsonFile = open("{0}/finalResults.json".format(RESULTS_DIR), "w")
    jsonFile.write(json.dumps(finalResults, indent=4))
    jsonFile.close()

# --------------------------------------------------------

def main():
    print('Starting Garrett Christian\'s Audio Fuzzing Tool')
    setup()

    try:
        print("Starting Mutations")
        while True:
            fuzz()
        
    except KeyboardInterrupt:
        collectFinalResults()

if __name__ == '__main__':
    main()