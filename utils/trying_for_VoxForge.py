import urllib
import urllib.request
import re
import zipfile
import logging
import tqdm
import os
import sys

#logging.basicConfig(filename="voxforge.log", \
#					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \
#					datefmt='%d-%b-%y %H:%M:%S', \
#					level=logging.DEBUG, filemode="a")

"""
Important folders and files for VoxForge Dataset:
<foldername>/wav/: contains the wavfiles
<foldername>/etc/prompts-original: <wav filename> <text>
<foldername>/etc/README: folder description, contains the sampling rate
bash command to extract sampling rate: grep ^Sampling <README path> | tr -d '[:space:]' | cut -d":" -f2
"""


def VoxForge_urls(url):
	""" Creates a list of urls for fetching the relevant _Audio_ Data
	:param url: the url of the original VoxForge Dataset website
				"http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/"
	:output useful_urls: the url links to the audio zipfiles
	"""
	# url_segments = url.split("/")[:-2]
	req = urllib.request.Request(url)
	resp = urllib.request.urlopen(req)
	respData = resp.read().decode("utf8")
	links = re.findall(r'href=(.*?)>',respData)
	#print(links)
	useful_urls = set()
	for link in links:
		if link.replace('"','').endswith(".tgz"):
			#print(link)
			useful_urls.add(url+link.replace('"',''))
	return useful_urls


def download_zip(url_list, download_path="datasets/VoxForge"):
	""" Uses the VoxForge_urls() functions output to download the audio zip files.
		 If the path already exists, it assumes the dataset to be present and hence
		 does not download the data again.
	:param urls: the list of relevant urls from which the data needs to be fetched
	:param path: the path where the zipfiles needs to be stored
				 (default: "datasets/VoxForge")
	"""
	if os.path.exists(download_path):
		print("Dataset already exists.")
	else:
		os.mkdir(download_path)
		zip_path = os.path.join(download_path,"zip_files")
		os.mkdir(zip_path)
		# print(zip_path)
		for i in tqdm.tqdm(url_list, desc="Downloading the Audio zip files: "):
			# takes the name of the zipfile being downloaded to save
			zip_name = i.split("/")[-1]
			urllib.request.urlretrieve(i, zip_path + "/" + zip_name)
		print("\nZip Files downloaded")


def unzip_audio_files(zip_path="dataset/VoxForge/zip_files", \
					  audio_path="dataset/VoxForge/audio_files"):
	""" This function is used to unzip all the zipfiles downloaded by the
		download_zip() function and place it to the path provided.
	:param zip_path: the path where the zipfiles are downloaded and kept
	:param path: the path where the audio files are to be downloaded
	"""
	if os.path.exists(audio_path):
		print("Audio files are already extracted.")
	else:
		#os.mkdir(os.path.join(pwd, path))
		os.mkdir(audio_path)
		#for zips in tqdm.tqdm(os.listdir(os.path.join(pwd, zip_path)), desc="Unzipping the Audio files: "):
		for zips in tqdm.tqdm(os.listdir(zip_path), desc="Unzipping the Audio files: "):
			name = zips.split(".")[-2]
			#with zipfile.ZipFile(os.path.join(os.path.join(pwd, zip_path), zips), 'r') as zip_ref:
			with zipfile.ZipFile(os.path.join(zip_path, zips), 'r') as zip_ref:
				#zip_ref.extractall(os.path.join(os.path.join(pwd, path), name))
				zip_ref.extractall(os.path.join(audio_path, name))
		print("\nFinished unzipping.")


if __name__=="__main__":
	home_url="http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/"
	# print(VoxForge_urls(home_url))
	urls = VoxForge_urls(home_url)
	download_zip(urls, \
				 download_path="../dataset/VoxForge")
	unzip_audio_files(zip_path="../dataset/VoxForge/zip_files", \
					  audio_path="../dataset/VoxForge/audio_files")
