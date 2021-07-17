## CS-753 Project Abstract
### Title : speech to sign-language(with emotions) for the hearing-impaired

1. Project Abstract

	With the aim of converting speech to sign-language for the hearing-impaired, we first convert the speech to text and then from text to sign-language. In addition to this we also would like to extract the emotion conveyed by the speech signal and display it to them as well because the sign language itself by the hand does not convey any sort of emotion.
	For this we would first try to train two different networks. The two networks are to train on the speech data, with one network converting the speech signal to text and the other detecting the emotion. Then, depending on the detected text we output the corresponding sign-language, along with the emotion conveyed with it.

2. Datasets

	- For emotion detection network: [RAVDESS speech dataset](https://doi.org/10.1371/journal.pone.0196391)
	- For speech to text network: [Google Audioset](https://research.google.com/audioset/)

3. References
	- https://imatge.upc.edu/web/projects/speech2signs-spoken-sign-language-translation-using-neural-networks
	- https://heartbeat.fritz.ai/the-3-deep-learning-frameworks-for-end-to-end-speech-recognition-that-power-your-devices-37b891ddc380


<hr>
#### Installing PyAudio
<hr>
In order to install pyaudio in linux environments, one needs to first install `portaudio19-dev`, and then installing pyaudio using pip3. In order to do that perform the following:
```bash
sudo apt-get install portaudio19-dev python3-pyaudio
```
