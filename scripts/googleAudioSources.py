import os


ytid = "-0SdAVK79lg"
start = 30.00
length = 10.00

# command = "ffmpeg $(youtube-dl -g \'https://www.youtube.com/watch?v={ytid}\' | sed \"s/.*/-ss {start} -i &/\") -t {length} -c copy test3.wav".format(
#     ytid=ytid, start=start, length=length)

# print(command)
# print()
# os.system(command)


# from youtube_dl import YoutubeDL

# audio_downloader = YoutubeDL({'format':'bestaudio'})

# audio_downloader.extract_info("https://www.youtube.com/watch?v=" + ytid)


import shutil

path = "yt-audio"

try:
    shutil.rmtree(path)
    print("Resetting Seed Folder")
except FileNotFoundError:
    print("Setting up new results folder")
os.mkdir(path)


url = "https://www.youtube.com/watch?v=" + ytid

import youtube_dl
import json

ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s.%(ext)s'})


import time

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

# print(json.dumps(video, indent=4, sort_keys=True))
# video_url = video['formats']['acodec']
# print(video_url)


# urls = [f['url'] for f in video['formats'] if f['acodec'] != 'none'][0]
# print(urls)

# print()


url = video['formats'][-1]['url']
print(url)
print()

mp4 = path + "/" + ytid + ".mp4"

command = "ffmpeg -ss {start} -i '{url}' -t {length} -c copy {mp4}".format(
    start = start, length = length, url = url, mp4 = mp4)
print(command)
os.system(command)
print()

mp3 = path + "/" + ytid + ".mp3"

command = "ffmpeg -i {mp4} -vn {mp3}".format(
    mp4 = mp4, mp3 = mp3)
print(command)
os.system(command)
print()

os.remove(mp4)

t1 = time.time()

total = t1-t0

print("\n\n!!! %f" % (total))