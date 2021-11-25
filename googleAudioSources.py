import os
import shutil
import youtube_dl
import time
import random
import csv
import json

ytid = "-0SdAVK79lg"
start = 30.00
length = 10.00

ytCsv = "/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/balanced_train_segments.csv"
labelJson = "/Users/garrettchristian/DocumentsDesktop/uva21/classes/softwareAna/project/6888project/ontology.json"

path = "yt-audio"

try:
    shutil.rmtree(path)
    print("Resetting Seed Folder")
except FileNotFoundError:
    print("Setting up new results folder")
os.mkdir(path)

# Get the labels
# Opening JSON file
f = open(labelJson)
 
# returns JSON object as
# a dictionary
ontology = json.load(f)
 
ontologyNames = {}

# Iterating through the json
# list
for item in ontology:
    id = item["id"]
    name = item["name"].replace(" ", "")
    name = name.replace(",", "")
    ontologyNames[id] = name
 
# Closing file
f.close()


# Handle the ytids
ytids = {}

i = 0

with open(ytCsv, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        if (i > 3):
            ytidCsv = row[0].replace(',', '')
            startCsv = row[1].replace(',', '')
            labels = row[3].replace('"', '').split(',')
            name = ytidCsv
            for label in labels:
                # print(label)
                name += ontologyNames[label]
            ytids[ytidCsv] = (startCsv, name)
        i += 1

randomKeys = random.sample(list(ytids), 500)

print(randomKeys)

ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s.%(ext)s'})

# Download the audio

i = 0

for ytid in randomKeys:
    i += 1
    try:
        url = "https://www.youtube.com/watch?v=" + ytid

        t0 = time.time()

        with ydl:
            result = ydl.extract_info(
                url,
                download=False # We just want to extract the info
            )

        if 'entries' in result:
            # Can be a playlist or a list of videos
            video = result['entries'][0]
        else:
            # Just a video
            video = result

        url = video['formats'][-1]['url']
        print(url)
        print()

        mp4 = path + "/" + ytid + ".mp4"

        command = "ffmpeg -ss {start} -i '{url}' -t {length} -c copy {mp4}".format(
            start = ytids[ytid][0], length = length, url = url, mp4 = mp4)
        print(command)
        os.system(command)
        print()

        wav = path + "/" + ytids[ytid][1] + ".wav"

        command = "ffmpeg -i {mp4} -vn {wav}".format(
            mp4 = mp4, wav = wav)
        print(command)
        os.system(command)
        print()

        os.remove(mp4)

        t1 = time.time()
        total = t1-t0
        print("\n\n%d) Completed in %f\n\n" % (i, total))

    except:
        print("Error downloading: ", ytid)

