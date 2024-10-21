import os
import sys
import numpy as np
import argparse
import json
from os.path import join, dirname
import logging

import pickle
from sklearn.decomposition import PCA, IncrementalPCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encoding_utils import *
from feature_spaces import _FEATURE_CONFIG, get_feature_space

from config import REPO_DIR, EM_DATA_DIR, DATA_DIR, SAVE_DIR

# Print paths for debugging
print(f"REPO_DIR from config: {REPO_DIR}")
print(f"EM_DATA_DIR from config: {EM_DATA_DIR}")
print(f"DATA_DIR from config: {DATA_DIR}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--subject", type=str, required=True)
	parser.add_argument("--feature", type=str, required=False)
	parser.add_argument("--x", type=int)
	parser.add_argument("--k", type=float)
	parser.add_argument("--sessions", nargs='+', type=int, default=[1, 2, 3, 4, 5])
	parser.add_argument("--trim", type=int, default=5)
	parser.add_argument("--ndelays", type=int, default=4)
	parser.add_argument("--nboots", type=int, default=50)
	parser.add_argument("--chunklen", type=int, default=40)
	parser.add_argument("--nchunks", type=int, default=125)
	parser.add_argument("--singcutoff", type=float, default=1e-10)
	parser.add_argument("-use_corr", action="store_true")
	parser.add_argument("-single_alpha", action="store_true")
	logging.basicConfig(level=logging.INFO)

	args = parser.parse_args()
	globals().update(args.__dict__)

	fs = " ".join(_FEATURE_CONFIG.keys())
	assert feature in _FEATURE_CONFIG.keys(), "Available feature spaces:" + fs
	assert np.amax(sessions) <= 5 and np.amin(sessions) >=1, "1 <= session <= 5"

	sessions = list(map(str, sessions))
	with open(join(EM_DATA_DIR, "sess_to_story.json"), "r") as f:
		sess_to_story = json.load(f) 
	train_stories, test_stories = [], []
	for sess in sessions:
		stories, tstory = sess_to_story[sess][0], sess_to_story[sess][1]
		train_stories.extend(stories)
		if tstory not in test_stories:
			test_stories.append(tstory)

	assert len(set(train_stories) & set(test_stories)) == 0, "Train - Test overlap!"

	sys.stdout.flush()

	# all stories is encoded into features
	allstories = list(set(train_stories) | set(test_stories))

	print(test_stories)
	print(train_stories)
	print(len(train_stories))

	allstories.remove("adollshouse")
	train_stories.remove("adollshouse")

	for story in allstories:
		stories = [story]
		Resp = get_response(stories, subject)
		print(story, "Rresp: ", Resp.shape)

		n_components = 100  # Target dimensionality

		# Using IncrementalPCA for memory efficiency
		ipca = IncrementalPCA(n_components=n_components, batch_size=200)
		X_ipca = ipca.fit_transform(Resp)
		print("array after PCA", X_ipca.shape)
		
		save_dir = os.path.join(SAVE_DIR, "100", story)
		os.makedirs(save_dir, exist_ok=True)
		save_file = os.path.join(save_dir, "vectors.pkl")
		print(save_file)
		with open(save_file, 'wb') as file:
			pickle.dump(X_ipca, file)
	
	print("Finished Saving PCA Responses")