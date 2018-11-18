import requests
import re
import os
import urllib.request

from bs4 import BeautifulSoup

url = 'https://www.minetegn.no/Tegnordbok-HTML/video_/'
testData = open('./testHtml.txt')
filesToDownload = 10

print('[info] downloading video list from %s' % url)
req = requests.get(url)

print('[info] creating parse tree for video list')
soup = BeautifulSoup(req.text, 'html.parser')

isMp4 = re.compile('.*\.mp4')
videoCount =1

if not os.path.exists('./data/test'):
	os.makedirs('./data/test')

links = soup.findAll('a')
numberOfLinks = len(links)

print('[info] downloading all %d mp4 (approximately)' % numberOfLinks)
for linknr, link in enumerate(soup.findAll('a'), 1):
	if videoCount > filesToDownload:
		break

	if isMp4.match(link.get('href')):
		className = link.get('href')[:-4]
		print('./data/test/%s' % className)
		if not os.path.exists('./data/test/%s' % className):
			os.makedirs('./data/test/%s' % className)
			print('[info] (%d - %d) downloading %s' % (linknr, numberOfLinks, link.get('href')))
			urllib.request.urlretrieve(url + link.get('href'), './data/test/%s/%s' % (className, link.get('href')))
			videoCount = videoCount+1