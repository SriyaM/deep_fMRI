import os
import sys
import numpy as np
import argparse
import json
from os.path import join, dirname
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from encoding_utils import *
from feature_spaces import _FEATURE_CONFIG, get_feature_space
from ridge_utils.ridge import bootstrap_ridge

from config import REPO_DIR, EM_DATA_DIR, DATA_DIR, SAVE_DIR

# Print paths for debugging
print(f"REPO_DIR from config: {REPO_DIR}")
print(f"EM_DATA_DIR from config: {EM_DATA_DIR}")
print(f"DATA_DIR from config: {DATA_DIR}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--subject", type=str, required=True)
	parser.add_argument("--feature", type=str, required=True)
	parser.add_argument("--x", type=int)
	parser.add_argument("--k", type=int)
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
	
	allstories = list(set(train_stories) | set(test_stories))

	print(test_stories)
	print(train_stories)
	print(len(train_stories))

	allstories.remove("adollshouse")
	train_stories.remove("adollshouse")

	def get_save_location(SAVE_DIR, feature, numeric_mod, subject):
		save_location = join(SAVE_DIR, "results", feature + "_" + numeric_mod, subject)
		os.makedirs(save_location, exist_ok=True)
	
	numeric_mod = ""
	if x is not None:
		numeric_mod += str(x) + "x"
		if k is not None:
			numeric_mod += "_" + str(k) + "k"
	
	if x is None and k is not None:
		numeric_mod += str(k) + "k"
	
	save_location = get_save_location(SAVE_DIR, feature, numeric_mod, subject)

	print("Saving encoding model & results to:", save_location)
	os.makedirs(save_location, exist_ok=True)

	downsampled_feat = get_feature_space(feature, allstories, x, k, subject)
	print("Stimulus & Response parameters:")
	print("trim: %d, ndelays: %d" % (trim, ndelays))

	# Delayed stimulus
	delRstim = apply_zscore_and_hrf(train_stories, downsampled_feat, trim, ndelays)
	print("delRstim: ", delRstim.shape)
	delPstim = apply_zscore_and_hrf(test_stories, downsampled_feat, trim, ndelays)
	print("delPstim: ", delPstim.shape)

	# Response
	zRresp = get_response(train_stories, subject)
	print("zRresp: ", zRresp.shape)
	zPresp = get_response(test_stories, subject)
	print("zPresp: ", zPresp.shape)

	# Ridge
	alphas = np.logspace(1, 3, 10)

	print("Ridge parameters:")
	print("nboots: %d, chunklen: %d, nchunks: %d, single_alpha: %s, use_corr: %s" % (
		nboots, chunklen, nchunks, single_alpha, use_corr))

	wt, corrs, valphas, bscorrs, valinds = bootstrap_ridge(
		delRstim, zRresp, delPstim, zPresp, alphas, nboots, chunklen, 
		nchunks, singcutoff=singcutoff, single_alpha=single_alpha, 
		use_corr=use_corr)
	
	print("FINISHED ROUND")
	sys.stdout.flush()

	# Save regression results.
	np.savez("%s/weights" % save_location, wt)
	np.savez("%s/corrs" % save_location, corrs)
	np.savez("%s/valphas" % save_location, valphas)
	np.savez("%s/bscorrs" % save_location, bscorrs)
	np.savez("%s/valinds" % save_location, np.array(valinds))
	print("Total r2: %d" % sum(corrs * np.abs(corrs)))
