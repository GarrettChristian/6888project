import gdown

id = '1SRmAEKw-SxFHnavM0u9S223Q6pdCdyQS'

url = 'https://drive.google.com/uc?id=' + id
output = 'yt-audio.zip'
gdown.download(url, output, quiet=False)