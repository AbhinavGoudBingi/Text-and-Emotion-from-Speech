import torch
import ctcdecode
import numpy as np
from transform_text import TextTransform
import warnings
warnings.filterwarnings("ignore")

class BeamCTCDecoder(object):
	"""
	Basic decoder class from which all other decoders inherit. Implements several
	helper functions. Subclasses should implement the decode() method.
	Arguments:
		labels (string): mapping from integers to characters.
		blank_index (int, optional): index for the blank '_' character, inferred from labels
		space_index (int, optional): index for the space ' ' character. Defaults to 28.
	"""

	def __init__(self, labels, lm_path=None, alpha=0, beta=0, cutoff_top_n=20, \
				cutoff_prob=1.0,  beam_width=4, num_processes=4):
		self.labels = labels
		self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
		self.blank_index = self.labels.index("_")
		self.decoder = ctcdecode.CTCBeamDecoder(labels = self.labels,
												model_path = lm_path,
												alpha = alpha,
												beta = beta,
												cutoff_top_n = len(self.labels),
												cutoff_prob = cutoff_prob,
												beam_width = beam_width,
												num_processes = num_processes,
												blank_id = self.blank_index,
												log_probs_input=False)
		self.text_transform = TextTransform()

	def avg_wer(self, wer_scores, combined_ref_len):
		return float(sum(wer_scores)) / float(combined_ref_len)

	def _levenshtein_distance(self, ref, hyp):
		m = len(ref)
		n = len(hyp)

		# special case
		if ref == hyp:
			return 0
		if m == 0:
			return n
		if n == 0:
			return m

		if m < n:
			ref, hyp = hyp, ref
			m, n = n, m

		# use O(min(m, n)) space
		distance = np.zeros((2, n + 1), dtype=np.int32)

		# initialize distance matrix
		for j in range(0,n + 1):
			distance[0][j] = j

		# calculate levenshtein distance
		for i in range(1, m + 1):
			prev_row_idx = (i - 1) % 2
			cur_row_idx = i % 2
			distance[cur_row_idx][0] = i
			for j in range(1, n + 1):
				if ref[i - 1] == hyp[j - 1]:
					distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
				else:
					s_num = distance[prev_row_idx][j - 1] + 1
					i_num = distance[cur_row_idx][j - 1] + 1
					d_num = distance[prev_row_idx][j] + 1
					distance[cur_row_idx][j] = min(s_num, i_num, d_num)

		return distance[m % 2][n]

	def word_errors(self, reference, hypothesis, ignore_case=False, delimiter=' '):
		"""Compute the levenshtein distance between reference sequence and
		hypothesis sequence in word-level.
		:param reference: The reference sentence.
		:type reference: basestring
		:param hypothesis: The hypothesis sentence.
		:type hypothesis: basestring
		:param ignore_case: Whether case-sensitive or not.
		:type ignore_case: bool
		:param delimiter: Delimiter of input sentences.
		:type delimiter: char
		:return: Levenshtein distance and word number of reference sentence.
		:rtype: list
		"""
		if ignore_case == True:
			reference = reference.lower()
			hypothesis = hypothesis.lower()

		ref_words = reference.split(delimiter)
		hyp_words = hypothesis.split(delimiter)

		edit_distance = self._levenshtein_distance(ref_words, hyp_words)
		return float(edit_distance), len(ref_words) if len(hyp_words)<len(ref_words) else len(hyp_words)

	def char_errors(self, reference, hypothesis, ignore_case=False, remove_space=False):
		"""Compute the levenshtein distance between reference sequence and
		hypothesis sequence in char-level.
		:param reference: The reference sentence.
		:type reference: basestring
		:param hypothesis: The hypothesis sentence.
		:type hypothesis: basestring
		:param ignore_case: Whether case-sensitive or not.
		:type ignore_case: bool
		:param remove_space: Whether remove internal space characters
		:type remove_space: bool
		:return: Levenshtein distance and length of reference sentence.
		:rtype: list
		"""
		if ignore_case == True:
			reference = reference.lower()
			hypothesis = hypothesis.lower()

		join_char = ' '
		if remove_space == True:
			join_char = ''

		reference = join_char.join(filter(None, reference.split(' ')))
		hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

		edit_distance = self._levenshtein_distance(reference, hypothesis)
		return float(edit_distance), len(reference) if len(reference)>len(hypothesis) else len(hypothesis)

	def wer(self, reference, hypothesis, ignore_case=False, delimiter=' '):
		"""Calculate word error rate (WER). WER compares reference text and
		hypothesis text in word-level. WER is defined as:
		.. math::
			WER = (Sw + Dw + Iw) / Nw
		where
		.. code-block:: text
			Sw is the number of words subsituted,
			Dw is the number of words deleted,
			Iw is the number of words inserted,
			Nw is the number of words in the reference
		We can use levenshtein distance to calculate WER. Please draw an attention
		that empty items will be removed when splitting sentences by delimiter.
		:param reference: The reference sentence.
		:type reference: basestring
		:param hypothesis: The hypothesis sentence.
		:type hypothesis: basestring
		:param ignore_case: Whether case-sensitive or not.
		:type ignore_case: bool
		:param delimiter: Delimiter of input sentences.
		:type delimiter: char
		:return: Word error rate.
		:rtype: float
		:raises ValueError: If word number of reference is zero.
		"""
		edit_distance, ref_len = self.word_errors(reference, hypothesis, ignore_case,
											delimiter)

		if ref_len == 0:
			raise ValueError("Reference's word number should be greater than 0.")

		wer = float(edit_distance) / ref_len
		return wer

	def cer(self, reference, hypothesis, ignore_case=False, remove_space=False):
		"""Calculate charactor error rate (CER). CER compares reference text and
		hypothesis text in char-level. CER is defined as:
		.. math::
			CER = (Sc + Dc + Ic) / Nc
		where
		.. code-block:: text
			Sc is the number of characters substituted,
			Dc is the number of characters deleted,
			Ic is the number of characters inserted
			Nc is the number of characters in the reference
		We can use levenshtein distance to calculate CER. Chinese input should be
		encoded to unicode. Please draw an attention that the leading and tailing
		space characters will be truncated and multiple consecutive space
		characters in a sentence will be replaced by one space character.
		:param reference: The reference sentence.
		:type reference: basestring
		:param hypothesis: The hypothesis sentence.
		:type hypothesis: basestring
		:param ignore_case: Whether case-sensitive or not.
		:type ignore_case: bool
		:param remove_space: Whether remove internal space characters
		:type remove_space: bool
		:return: Character error rate.
		:rtype: float
		:raises ValueError: If the reference length is zero.
		"""
		edit_distance, ref_len = self.char_errors(reference, hypothesis, ignore_case,
											remove_space)

		if ref_len == 0:
			raise ValueError("Length of reference should be greater than 0.")

		cer = float(edit_distance) / ref_len
		return cer

	def decode_(self, probs, sizes=None):
		"""
		Given a matrix of character probabilities, returns the decoder's
		best guess of the transcription
		Arguments:
			probs: Tensor of character probabilities, where probs[c,t]
							is the probability of character c at time t
			sizes(optional): Size of each sequence in the mini-batch
		Returns:
			string: sequence of the model's best guess for the transcription
		"""
		#beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(probs, sizes)
		beam_result, beam_scores, timesteps, out_seq_len = self.decoder.decode(probs)
		#return text_transform.int_to_text(beam_result[:,0][:out_seq_len[:,0]])
		return self.convert_to_string(beam_result, out_seq_len)

	def convert_to_string(self, out, seq_len):
		results = []
		for i in range(len(out)):
			results.append(self.text_transform.int_to_text(out[i][0][:seq_len[i][0]].tolist()))
			#results.append("".join([self.int_to_char[n.item()] for n in out[i][0][:seq_len[i][0]]]))
		#text_transform.int_to_text(labels[i][:label_lengths[i]].tolist())
		# self.int_to_char[n.item()]
		return results
