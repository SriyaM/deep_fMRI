import os
import sys
import numpy as np
import json
from os.path import join, dirname
from tqdm import tqdm
import pickle
from huggingface_hub import login

from ridge_utils.interpdata import lanczosinterp2D
from ridge_utils.SemanticModel import SemanticModel
from ridge_utils.dsutils import *
from ridge_utils.stimulus_utils import load_textgrids, load_simulated_trfiles
from ridge_utils.stimulus_utils import load_textgrids, load_simulated_trfiles

from config import REPO_DIR, EM_DATA_DIR, DATA_DIR, SAVE_DIR

import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel, BertForMaskedLM, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA, IncrementalPCA

# NEED TO CHANGE THE PATH OF THIS TO EUNJIS FILE
from mech_lm.experiments.inference_neuro import setup_model, generate_embeddings

def get_save_location(SAVE_DIR, feature, numeric_mod, subject):
	save_location = join(SAVE_DIR, "results", feature + "_" + numeric_mod, subject)
	os.makedirs(SAVE_DIR, exist_ok=True)

def get_story_wordseqs(stories):
	grids = load_textgrids(stories, DATA_DIR)
	with open(join(DATA_DIR, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	# a dictionary of words listened to between the start and end time
	trfiles = load_simulated_trfiles(respdict)
	# returns a dictionary with dataseq stories that have a transcript with filtered words and metadata
	wordseqs = make_word_ds(grids, trfiles)
	return wordseqs

def get_story_phonseqs(stories):
	grids = load_textgrids(stories, DATA_DIR)
	with open(join(DATA_DIR, "ds003020/derivative/respdict.json"), "r") as f:
		respdict = json.load(f)
	trfiles = load_simulated_trfiles(respdict)
	wordseqs = make_phoneme_ds(grids, trfiles)
	return wordseqs

def downsample_word_vectors(stories, word_vectors, wordseqs):
	"""Get Lanczos downsampled word_vectors for specified stories.

	Args:
		stories: List of stories to obtain vectors for.
		word_vectors: Dictionary of {story: <float32>[num_story_words, vector_size]}

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	downsampled_semanticseqs = dict()
	for story in stories:
		downsampled_semanticseqs[story] = lanczosinterp2D(
			word_vectors[story], wordseqs[story].data_times, 
			wordseqs[story].tr_times, window=3)
	return downsampled_semanticseqs

######################################
########## ENG1000 Features ##########
######################################

def get_eng1000_vectors(allstories, x, k):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}

	# for each story, create embeddings using the wordseqs
	for story in allstories:
		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		vectors[story] = sm.data
	
	for story in allstories:
		print("LENGTHS", "time", story, len(wordseqs[story].data), len(wordseqs[story].data_times), len(vectors[story]), len(wordseqs[story].split_inds))
	return downsample_word_vectors(allstories, vectors, wordseqs)

######################################
########## Llama Features ############
######################################

def get_llama_vectors(allstories, x, k):
	"""Get llama vectors (300-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		word_vectors = get_last_token_embedding((wordseqs[story]).data, k)
		vectors[story] = word_vectors
	return downsample_word_vectors(allstories, vectors, wordseqs)

def get_last_token_embedding(words, window_size, model, tokenizer,  batch_size=64):

	# Check if ROCm is available and move model to GPU
	if torch.cuda.is_available():  # ROCm uses the same torch.cuda API
		device = torch.device('cuda')
		print(f"Using GPU: {torch.cuda.get_device_name(0)}")
		sys.stdout.flush()
	else:
		device = torch.device('cpu')
		print("ROCm not available, using CPU.")
		sys.stdout.flush()

	# Manually set padding token if it's not set
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	# Record indices of empty words and remove them from the list
	empty_word_indices = [idx for idx, word in enumerate(words) if not word]
	trimmed_words = [word for word in words if word]

	# Tokenize the entire text once
	encoded_input = tokenizer(trimmed_words, return_tensors='pt', padding=True, truncation=False, is_split_into_words=True)
	input_ids = encoded_input['input_ids'][0]
	attention_mask = encoded_input['attention_mask'][0]

	# Move to device
	input_ids = input_ids.to(device)
	attention_mask = attention_mask.to(device)

	# Map word indices to token indices
	word_to_token_indices = []
	current_token_idx = 0
	for word in trimmed_words:
		tokenized_word = tokenizer.tokenize(word)
		token_indices = list(range(current_token_idx, current_token_idx + len(tokenized_word)))
		word_to_token_indices.append(token_indices)
		current_token_idx += len(tokenized_word)

	all_embeddings = []

	# Prepare context windows based on word indices
	context_indices = []
	for idx, token_indices in enumerate(word_to_token_indices):
		start_idx = max(0, idx - window_size + 1)
		end_idx = idx + 1
		context_window_tokens = [tok for word_idx in range(start_idx, end_idx) for tok in word_to_token_indices[word_idx]]
		context_indices.append((context_window_tokens, token_indices[-1]))

	# Process context windows in batches
	for i in tqdm(range(0, len(context_indices), batch_size)):
		batch_contexts = context_indices[i:i + batch_size]

		# Ensure no out-of-bounds tokens before creating input batches
		valid_batch_contexts = []
		for tokens, token_idx in batch_contexts:
			if all(0 <= t < len(input_ids) for t in tokens):
				valid_batch_contexts.append((tokens, token_idx))
			else:
				print(f"Skipping out-of-bounds context: tokens={tokens}, token_idx={token_idx}")
				print(len(input_ids))

		if not valid_batch_contexts:
			continue

		# Create input batches
		input_ids_batch = torch.nn.utils.rnn.pad_sequence([input_ids[torch.tensor(tokens)].to(device) for tokens, _ in valid_batch_contexts], batch_first=True)
		attention_mask_batch = torch.nn.utils.rnn.pad_sequence([attention_mask[torch.tensor(tokens)].to(device) for tokens, _ in valid_batch_contexts], batch_first=True)

		with torch.no_grad():
			outputs = model(input_ids=input_ids_batch, attention_mask=attention_mask_batch)

		# Get the last hidden state
		last_hidden_state = outputs.last_hidden_state

		# Extract the embeddings of the current words (last token in each window)
		for j, (_, token_idx) in enumerate(valid_batch_contexts):
			relative_token_idx = token_idx - valid_batch_contexts[j][0][0]
			if 0 <= relative_token_idx < last_hidden_state.size(1):
				current_word_embedding = last_hidden_state[j, relative_token_idx, :].unsqueeze(0)
				all_embeddings.append(current_word_embedding)
			else:
				print(f"Skipping out-of-bounds embedding: relative_token_idx={relative_token_idx}, last_hidden_state.size(1)={last_hidden_state.size(1)}")

	# Concatenate all batches to get the final embeddings
	if all_embeddings:
		all_embeddings = torch.cat(all_embeddings, dim=0)

		# Convert the final tensor to a NumPy array
		all_embeddings_np = all_embeddings.cpu().numpy()
		print(len(all_embeddings_np))
	else:
		all_embeddings_np = np.array([])

	# Ensure the output has the same number of embeddings as the original number of words
	embedding_dim = all_embeddings_np.shape[1] if all_embeddings_np.size > 0 else model.config.hidden_size
	zero_vector = np.zeros((1, embedding_dim))

	final_embeddings_np = []
	trimmed_word_idx = 0

	for idx in range(len(words)):
		if idx in empty_word_indices:
			final_embeddings_np.append(zero_vector)
		else:
			final_embeddings_np.append(all_embeddings_np[trimmed_word_idx])
			trimmed_word_idx += 1

	final_embeddings_np = np.vstack(final_embeddings_np)

	return final_embeddings_np

##########################################################
########## Infinigram Features with induction ############
##########################################################

def get_pca_tr_infini_general(allstories, x, k, vec_location_suffix, model_type, tokenizer, model_checkpoint, batch_size=64):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.
		x: Integer value to determine the embedding path.
		k: Top percent of vectors to keep.
		vec_location_suffix: The suffix to append to the vector location for different data variations.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	vectors = {}
	save_location = get_save_location(SAVE_DIR, vec_location_suffix, "", "UTS03")

	if "llama" in model_checkpoint:
		base_checkpoint = "llama2"
	else:
		base_checkpoint = "gpt2"

	# Setup models for both infini-gram and specific model
	lm, tokenizer = setup_model(save_location, model_type, seed=1, tokenizer_checkpoint=tokenizer, checkpoint=model_checkpoint)
	lm_b, tokenizer_b = setup_model(save_location, 'infini-gram', seed=1, tokenizer_checkpoint="gpt2", checkpoint=base_checkpoint)

	# Process each story in the list
	for story in allstories:
		words = (wordseqs[story]).data
		
		# Initialize an empty list to store the vectors for this story
		story_vectors = []
		
		# Process words in batches
		for i in range(0, len(words), batch_size):
			batch = words[i:i + batch_size]
			batch_vectors = generate_embeddings(batch, lm, tokenizer, lm_b, tokenizer_b, save_location, story, model_type)
			story_vectors.append(batch_vectors)
		
		# Concatenate all batch vectors for this story
		story_vectors = np.concatenate(story_vectors, axis=0)

		# Replace NaN values with zero
		story_vectors = np.nan_to_num(story_vectors, nan=0.0)

		# Apply Incremental PCA
		n_components = x  # Target dimensionality
		ipca = IncrementalPCA(n_components=n_components, batch_size=1000)
		story_vectors_pca = ipca.fit_transform(story_vectors)
		print(f"{story} array after PCA", story_vectors_pca.shape)

		# Save the PCA-transformed vectors to an output file
		os.makedirs(os.path.join(save_location, story), exist_ok=True)
		pickle_file_path = join(save_location, story, 'vectors.pkl')

		with open(pickle_file_path, 'wb') as pickle_file:
			pickle.dump(np.array(story_vectors_pca), pickle_file)

	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	top_percent = k

	vectors = {}
	resp = {}

	# for each story, create embeddings using the wordseqs
	for story in allstories:
		# Load Infinigram Vectors
		cur_vec_file = os.path.join(save_location, story, "vectors.pkl")
		with open(cur_vec_file, 'rb') as file:
			infini_vectors = pickle.load(file)

		# Load PCA responses
		pca_file = os.path.join(SAVE_DIR, "100", story)
		with open(pca_file, 'rb') as file:
			resp_vectors = pickle.load(file)
			resp[story] = resp_vectors

		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		eng_vectors = sm.data
		fin_vectors = pca_tr_infini_prep(wordseqs[story], eng_vectors, infini_vectors, resp_vectors)
		vectors[story] = fin_vectors

	# Downsample Vectors
	downsampled = downsample_word_vectors(allstories, vectors, wordseqs)

	new_features = dict()
	for story in allstories:
		# Induction over TR times
		new_vector = interp_induction(wordseqs[story], eng_vectors, infini_vectors, resp[story], downsampled[story], top_percent)
		# Validate the dimension
		print("dimension of new TR", story, new_vector.shape)
		new_features[story] = new_vector

	return new_features


def get_pca_tr_infini(allstories, x, k):
    """Calls the general PCA TR function with infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, "infinigram", 'infini-gram', "gpt2", "hf_openwebtext_gpt2")

def get_pca_tr_infini_w_cont(allstories, x, k):
    """Calls the general PCA TR function with infinigram_w_cont vectors."""
    return get_pca_tr_infini_general(allstories, x, k, "infinigram_w_cont", 'infini-gram-w-incontext', "gpt2", "hf_openwebtext_gpt2")

def get_pca_tr_incont_infini(allstories, x, k):
    """Calls the general PCA TR function with incont_infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, "incont_infinigram", 'incontext-infini-gram', "gpt2", "hf_openwebtext_gpt2")

def get_pca_tr_incont_fuzzy_gpt(allstories, x, k):
    """Calls the general PCA TR function with incont_infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, "incont_fuzzy_gpt", 'incontext-fuzzy', "gpt2", "hf_openwebtext_gpt2")

def get_pca_tr_incont_fuzzy_llama(allstories, x, k):
    """Calls the general PCA TR function with incont_infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, "incont_fuzzy_llama", 'incontext-fuzzy', "llama2", "hf_openwebtext_llama")

def get_pca_tr_incont_dist_gpt(allstories, x, k):
    """Calls the general PCA TR function with incont_infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, "incont_fuzzy_gpt", 'incontext-fuzzy', "gpt2_dist", "hf_openwebtext_gpt2")

def get_pca_tr_incont_dist_llama(allstories, x, k):
    """Calls the general PCA TR function with incont_infinigram vectors."""
    return get_pca_tr_infini_general(allstories, x, k, "incont_fuzzy_llama", 'incontext-fuzzy', "llama2_dist", "hf_openwebtext_llama")


def get_pca_tr_random(allstories, x, k):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	top_percent = k

	vectors = {}
	resp = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		# Load PCA responses
		pca_file = os.path.join(SAVE_DIR, "100", story)
		with open(pca_file, 'rb') as file:
			resp_vectors = pickle.load(file)
			resp[story] = resp_vectors

		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		eng_vectors = sm.data
		vectors[story] = eng_vectors
	
	# Downsample Vectors
	downsampled = downsample_word_vectors(allstories, vectors, wordseqs)
	# Adjust them in the correct way
	new_features = dict()
	for story in allstories:
		new_vector = interp_induction_random(wordseqs[story], eng_vectors, resp[story], downsampled[story], top_percent)
		# Validate the dimension
		print("dimension of new TR", story, new_vector.shape)
		new_features[story] = new_vector

	return new_features

def get_pca_tr_exact(allstories, x, k):
	"""Get Eng1000 vectors (985-d) for specified stories.

	Args:
		allstories: List of stories to obtain vectors for.

	Returns:
		Dictionary of {story: downsampled vectors}
	"""
	wordseqs = get_story_wordseqs(allstories)
	eng1000 = SemanticModel.load(join(EM_DATA_DIR, "english1000sm.hf5"))
	top_percent = k

	vectors = {}
	resp = {}
	# for each story, create embeddings using the wordseqs
	for story in allstories:
		# Load PCA responses
		pca_file = os.path.join(SAVE_DIR, "100", story)
		with open(pca_file, 'rb') as file:
			resp_vectors = pickle.load(file)
			resp[story] = resp_vectors

		sm = make_semantic_model(wordseqs[story], [eng1000], [985])
		eng_vectors = sm.data
		vectors[story] = eng_vectors
	
	# Downsample Vectors
	downsampled = downsample_word_vectors(allstories, vectors, wordseqs)
	# Adjust them in the correct way
	new_features = dict()
	for story in allstories:
		new_vector = interp_induction_exact(wordseqs[story], eng_vectors, resp[story], downsampled[story], top_percent)
		# Validate the dimension
		print("dimension of new TR", story, new_vector.shape)
		new_features[story] = new_vector

	return new_features

############################################
########## Feature Space Creation ##########
############################################

_FEATURE_CONFIG = {
	"eng1000": get_eng1000_vectors,
	"llama_16_words": get_llama_vectors,
	"pca_tr_infini" : get_pca_tr_infini,
	"pca_tr_infini_w_cont" : get_pca_tr_infini_w_cont,
	"pca_tr_incont_infini" : get_pca_tr_incont_infini,
	"pca_tr_incont_dist_gpt" : get_pca_tr_incont_dist_gpt,
	"pca_tr_incont_dist_llama" : get_pca_tr_incont_dist_llama,
	"pca_tr_incont_fuzzy_gpt" : get_pca_tr_incont_fuzzy_gpt,
	"pca_tr_incont_fuzzy_llama" : get_pca_tr_incont_fuzzy_llama,
	"pca_tr_random" : get_pca_tr_random,
	"pca_tr_exact" : get_pca_tr_exact,
}

def get_feature_space(feature, *args):
	return _FEATURE_CONFIG[feature](*args)