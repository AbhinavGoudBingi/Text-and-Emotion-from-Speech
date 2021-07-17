import urllib
import urllib.request
import re
import zipfile
import logging
import tqdm
import os
import sys

logging.basicConfig(filename="runtime.log", \
					format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', \
					datefmt='%d-%b-%y %H:%M:%S', \
					level=logging.DEBUG, filemode="a")


def RAVDESS_urls(url):
    """ Creates a list of urls for fetching the relevant _Audio_ Data
    :param url: the url of the original RAVDESS Dataset website
                "https://zenodo.org/record/1188976"
    :output useful_urls: the url links to the audio zipfiles
    """
    url_segments = url.split("/")[:-2]
    req = urllib.request.Request(url)
    resp = urllib.request.urlopen(req)
    respData = resp.read()
    links = re.findall(r'href=(.*?)>',str(respData))
    useful_urls = set()
    for link in links:
        if "download=1" in link and "Audio_Speech" in link:
            #print(link)
            useful_urls.add(("/".join(url_segments)+link[1:].split(">")[0]).strip('"'))
    return useful_urls


def download_zip(urls, path="dataset/RAVDESS"):
    """ Uses the RAVDESS_url() function output to download the audio zip files.
        If the path already exists, it assumes the dataset to be present and hence
        does not download the data again.
    :param urls: the list of relevant urls from which the data needs to be fetched
    :param path: the path where the zipfiles needs to be stored
                 (default: "RAVDESS")
    """
    #pwd = os.getcwd()
    #if os.path.exists(os.path.join(pwd, path)):
    if os.path.exists(path):
        logging.debug("Directory already exists. Seeking for zip files.")
    else:
        #os.mkdir(os.path.join(pwd, path))
        os.mkdir(path)
        #zip_path = os.path.join(pwd,path+"/zip_files")
        zip_path = os.path.join(path,"zip_files")
        os.mkdir(zip_path)
        # print(zip_path)
        for i in tqdm.tqdm(urls, desc="Downloading the Audio zip files: "):
            # takes the name of the zipfile being downloaded to save
            zip_name = i.split("?")[0].split("/")[-1]
            urllib.request.urlretrieve(i, zip_path + "/" + zip_name)
        logging.debug("\nZip Files downloaded")


def unzip_audio_files(zip_path="dataset/RAVDESS/zip_files" ,path="dataset/RAVDESS/audio_files"):
    """ This function is used to unzip all the zipfiles downloaded by the
        download_zip() function and place it to the path provided.
    :param zip_path: the path where the zipfiles are downloaded and kept
    :param path: the path where the audio files are to be downloaded
    """
    #pwd = os.getcwd()
    #if os.path.exists(os.path.join(pwd, path)):
    if os.path.exists(path):
        logging.debug("Audio files are already extracted.")
    else:
        #os.mkdir(os.path.join(pwd, path))
        os.mkdir(path)
        #for zips in tqdm.tqdm(os.listdir(os.path.join(pwd, zip_path)), desc="Unzipping the Audio files: "):
        for zips in tqdm.tqdm(os.listdir(zip_path), desc="Unzipping the Audio files: "):
            name = zips.split(".")[-2]
            #with zipfile.ZipFile(os.path.join(os.path.join(pwd, zip_path), zips), 'r') as zip_ref:
            with zipfile.ZipFile(os.path.join(zip_path, zips), 'r') as zip_ref:
                #zip_ref.extractall(os.path.join(os.path.join(pwd, path), name))
                zip_ref.extractall(os.path.join(path, name))
        logging.debug("\nFinished unzipping.")



if __name__=="__main__":
	RAVDESS_URL = "https://zenodo.org/record/1188976"
	urls = RAVDESS_urls(RAVDESS_URL)
	download_zip(urls, path="../dataset/RAVDESS")
	unzip_audio_files(zip_path="../dataset/RAVDESS/zip_files" ,path="../dataset/RAVDESS/audio_files")
