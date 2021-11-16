import os
import shutil
from pathlib import Path


seedPath = "/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/TestClean/test-clean"
newPath = "test-clean"

try:
    shutil.rmtree(newPath)
    print("Resetting Seed Folder")
except FileNotFoundError:
    print("Setting up new results folder")
os.mkdir(newPath)

for seed in Path(seedPath).rglob('*.flac'):
    newSeed = str(seed).replace(seedPath, newPath)
    newSeed = newSeed.replace("flac", "wav")
    makeDir = str(seed).replace(seedPath, newPath).replace(seed.name, "")

    os.makedirs(makeDir, exist_ok=True)
    print("%s %s" % (str(seed), newSeed))
    command = "ffmpeg -i {flac} {output}".format(flac = str(seed), output = newSeed)
    print(command)
    print()
    os.system(command)


