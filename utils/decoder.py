import ctcdecode
import torch

labels = [
    "'",  # 0
    " ",  # 1
    "a",  # 2
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",  # 27
    "_",  # 28, blank
]

class BeamDecoder:
	""" Implements the CTCBeamDecoder utility and
		uses the above defined labels list.

		Init Params:
			beam_size: number of beams to consider for the beam decoding
			blank_id: letting the decoder know about the blank id("_") index
			lm_path: path to the language model if any (can use kenlm models)

		Output:
			The decoded words or sentences from the speech signal.
	"""
	def __init__(self, beam_size=8, blank_id= labels.index("_"), lm_path=None):
		self.__decoder = ctcdecode.CTCBeamDecoder(labels=labels,
												beam_width=beam_size,
												blank_id=blank_id,
												model_path=lm_path)
		print("CTCBeamDecoder loaded from ctcdecode module.")

	def __call__(self, output):
		beam_result, beam_scores, timesteps, out_seq_len = self.__decoder.decode(output)
		return self.convert_to_string(beam_result[0][0], labels, out_seq_len[0][0])

	def convert_to_string(self, beam_result, labels, out_seq_len):
		return "".join(label[x] for x in beam_result[0:out_seq_len])
