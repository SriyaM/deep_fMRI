import numpy as np
from ridge_utils.DataSequence import DataSequence
import random

DEFAULT_BAD_WORDS = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])

import numpy as np
from ridge_utils.DataSequence import DataSequence
import random

DEFAULT_BAD_WORDS = frozenset(["sentence_start", "sentence_end", "br", "lg", "ls", "ns", "sp"])

def interp_induction(ds, eng_vectors, infini_vectors, resp_vectors, tr_vectors, top_percent=1.0):
    """
    This function processes vectors and selects top percent values from `tr_vectors`.
    It uses these to compute differences in response vectors and appends those differences 
    as additional features.

    Args:
        ds: DataSequence object containing word information.
        eng_vectors: Unused in this function (kept for interface consistency).
        infini_vectors: Unused in this function (kept for interface consistency).
        resp_vectors: Response vectors to be used for feature generation.
        tr_vectors: Trained vectors for comparison and top percent selection.
        top_percent: The top percentage of vectors to consider for feature extraction.

    Returns:
        final_array: A new array with concatenated features and previous timestep information.
    """
    
    print("shape of down vec", tr_vectors.shape)
    print("shape of response vec", resp_vectors.shape)

    # Padding zeros before and after the response vectors for boundary cases
    zeros_before = np.zeros((10, 100))
    zeros_after = np.zeros((5, 100))
    resp_vectors = np.concatenate((zeros_before, resp_vectors, zeros_after), axis=0)

    newdata = []
    num_words = len(ds.data)
    sim_length = 3218

    # Create a map from word index to tr_time
    ind = range(0, num_words)
    ind_chunks = np.split(ind, ds.split_inds)
    ind_to_tr = {}

    for chunk_index, chunk in enumerate(ind_chunks):
        for i in chunk:
            ind_to_tr[i] = chunk_index

    # Find the maximum value in the last `sim_length` of each row
    max_values = []
    max_indices = []
    
    for row in tr_vectors:
        last_values = row[-sim_length:]
        max_value = np.max(last_values)
        max_index = np.argmax(last_values)
        max_values.append(max_value)
        max_indices.append(max_index)
    
    # Determine the threshold for the top percent
    threshold = np.percentile(max_values, 100 * (1 - top_percent))
    
    new_array = []
    for i, row in enumerate(tr_vectors):
        current_max = max_values[i]
        if current_max >= threshold:
            match_tr_ind = ind_to_tr[max_indices[i]]
            print("cur_tr", i, "match", match_tr_ind)
            if match_tr_ind + 3 < len(resp_vectors):
                feature_1 = resp_vectors[match_tr_ind + 1] - resp_vectors[match_tr_ind]
                feature_2 = resp_vectors[match_tr_ind + 2] - resp_vectors[match_tr_ind]
                feature_3 = resp_vectors[match_tr_ind + 3] - resp_vectors[match_tr_ind]
            else:
                feature_1 = np.zeros(100)
                feature_2 = np.zeros(100)
                feature_3 = np.zeros(100)
        else:
            print("cur_tr", i, "match", "none")
            feature_1 = np.zeros(100)
            feature_2 = np.zeros(100)
            feature_3 = np.zeros(100)
        
        new_row = np.concatenate((row[:985], feature_1, feature_2, feature_3))
        new_array.append(new_row)
    
    # Append previous time steps' information to the current vector
    final_array = []
    for i, row in enumerate(new_array):
        prev_row_1 = new_array[i-1][-300:] if i > 0 else np.zeros(300)
        prev_row_2 = new_array[i-2][-300:] if i > 1 else np.zeros(300)
        final_row = np.concatenate((row, prev_row_1, prev_row_2))
        final_array.append(final_row)
    
    return np.array(final_array)

def interp_induction_random(ds, eng_vectors, resp_vectors, tr_vectors, top_percent=0.2):
    """
    This function randomly selects indices and computes differences in response vectors 
    to concatenate as features.

    Args:
        ds: DataSequence object containing word information.
        eng_vectors: Unused in this function (kept for interface consistency).
        resp_vectors: Response vectors to be used for feature generation.
        tr_vectors: Trained vectors for comparison and feature extraction.
        top_percent: Percentage of top vectors (not used here).

    Returns:
        final_array: A new array with concatenated random features and previous timestep information.
    """
    
    print("shape of down vec", tr_vectors.shape)
    print("shape of response vec", resp_vectors.shape)

    # Padding zeros before and after the response vectors for boundary cases
    zeros_before = np.zeros((10, 100))
    zeros_after = np.zeros((5, 100))
    resp_vectors = np.concatenate((zeros_before, resp_vectors, zeros_after), axis=0)

    num_words = len(ds.data)
    
    # Create map from index to tr_time
    ind = range(0, num_words)
    ind_chunks = np.split(ind, ds.split_inds)
    ind_to_tr = {}

    for chunk_index, chunk in enumerate(ind_chunks):
        for i in chunk:
            ind_to_tr[i] = chunk_index

    # Randomly select indices with replacement
    random_indices = random.choices(range(len(tr_vectors)), k=len(tr_vectors))

    new_array = []
    for i, row in enumerate(tr_vectors):
        random_ind = random_indices[i]
        match_tr_ind = ind_to_tr.get(random_ind, None)
        
        if match_tr_ind is not None and match_tr_ind + 3 < len(resp_vectors):
            feature_1 = resp_vectors[match_tr_ind + 1] - resp_vectors[match_tr_ind]
            feature_2 = resp_vectors[match_tr_ind + 2] - resp_vectors[match_tr_ind]
            feature_3 = resp_vectors[match_tr_ind + 3] - resp_vectors[match_tr_ind]
            print("cur_tr", i, "random match", match_tr_ind)
        else:
            print("cur_tr", i, "random match", "none")
            feature_1 = np.zeros(100)
            feature_2 = np.zeros(100)
            feature_3 = np.zeros(100)

        new_row = np.concatenate((row[:985], feature_1, feature_2, feature_3))
        new_array.append(new_row)

    final_array = []
    for i, row in enumerate(new_array):
        prev_row_1 = new_array[i-1][-300:] if i > 0 else np.zeros(300)
        prev_row_2 = new_array[i-2][-300:] if i > 1 else np.zeros(300)
        final_row = np.concatenate((row, prev_row_1, prev_row_2))
        final_array.append(final_row)

    return np.array(final_array)

def find_matching_index(word_list, current_idx, n):
    """
    Finds the index of the closest match of a sequence of `n` words before the current index.

    Args:
        word_list: List of words in the story.
        current_idx: Index of the word to start matching.
        n: Number of words to match.

    Returns:
        The index of the matching sequence if found, otherwise None.
    """
    if current_idx < n:
        return None

    current_sequence = word_list[current_idx - n:current_idx]

    for i in range(current_idx - n):
        if word_list[i:i + n] == current_sequence:
            return i

    return None

def interp_induction_exact(ds, eng_vectors, resp_vectors, tr_vectors, top_percent=0.2):
    """
    This function finds exact matching word sequences and uses their respective 
    response vector differences as features.

    Args:
        ds: DataSequence object containing word information.
        eng_vectors: Unused in this function (kept for interface consistency).
        resp_vectors: Response vectors to be used for feature generation.
        tr_vectors: Trained vectors for comparison and exact matching.
        top_percent: Percentage of top vectors (not used here).

    Returns:
        final_array: A new array with concatenated features based on matched sequences 
        and previous timestep information.
    """
    
    print("shape of down vec", tr_vectors.shape)
    print("shape of response vec", resp_vectors.shape)

    zeros_before = np.zeros((10, 100))
    zeros_after = np.zeros((5, 100))
    resp_vectors = np.concatenate((zeros_before, resp_vectors, zeros_after), axis=0)

    num_words = len(ds.data)
    word_list = ds.data
    
    ind = range(0, num_words)
    ind_chunks = np.split(ind, ds.split_inds)
    ind_to_tr = {}

    for chunk_index, chunk in enumerate(ind_chunks):
        for i in chunk:
            ind_to_tr[i] = chunk_index

    new_array = []
    for i, row in enumerate(tr_vectors):
        match_word_ind = None
        for n in range(4, 0, -1):
            match_word_ind = find_matching_index(word_list, i, n)
            if match_word_ind is not None:
                break

        if match_word_ind is not None:
            match_tr_ind = ind_to_tr.get(match_word_ind, None)
        else:
            match_tr_ind = None
        
        if match_tr_ind is not None and match_tr_ind + 3 < len(resp_vectors):
            feature_1 = resp_vectors[match_tr_ind + 1] - resp_vectors[match_tr_ind]
            feature_2 = resp_vectors[match_tr_ind + 2] - resp_vectors[match_tr_ind]
            feature_3 = resp_vectors[match_tr_ind + 3] - resp_vectors[match_tr_ind]
            print(f"cur_tr {i} matched {n}-word sequence at index {match_word_ind} (tr index {match_tr_ind})")
        else:
            print(f"cur_tr {i} no match")
            feature_1 = np.zeros(100)
            feature_2 = np.zeros(100)
            feature_3 = np.zeros(100)

        new_row = np.concatenate((row[:985], feature_1, feature_2, feature_3))
        new_array.append(new_row)

    final_array = []
    for i, row in enumerate(new_array):
        prev_row_1 = new_array[i-1][-300:] if i > 0 else np.zeros(300)
        prev_row_2 = new_array[i-2][-300:] if i > 1 else np.zeros(300)
        final_row = np.concatenate((row, prev_row_1, prev_row_2))
        final_array.append(final_row)

    return np.array(final_array)

def pca_tr_infini_prep(ds, eng_vectors, infini_vectors, resp_vectors):
    """
    Prepares a PCA-compatible DataSequence object by appending similarity 
    scores between the current word vector and previous eligible words.

    Args:
        ds: DataSequence object containing word information.
        eng_vectors: Encoded word vectors (e.g., PCA-reduced vectors).
        infini_vectors: Infinite-gram embeddings for words.
        resp_vectors: Response vectors for comparison and similarity calculation.

    Returns:
        newdata: Array with concatenated similarity vectors and original word vectors.
    """
    newdata = []
    num_words = len(ds.data)
    sim_vec_length = 3218

    ind = range(0, num_words)
    ind_chunks = np.split(ind, ds.split_inds)
    ind_to_tr = {}

    for chunk_index, chunk in enumerate(ind_chunks):
        for i in chunk:
            ind_to_tr[i] = chunk_index

    for ind, w in enumerate(ds.data):
        cur_tr_ind = ind_to_tr[ind]

        sim_vector = np.zeros(sim_vec_length)
        
        if cur_tr_ind == 0 or len(ind_chunks[cur_tr_ind - 3]) == 0 or cur_tr_ind < 3:
            eligible_ind = []  
        else:
            final = ind_chunks[cur_tr_ind - 3][0]
            eligible_ind = [i for i in range(0, final)]
        
        if eligible_ind:
            eligible_infinigram = [infini_vectors[idx] for idx in eligible_ind]
            cur_infini = infini_vectors[ind]
            
            for i, idx in enumerate(eligible_ind):
                similarity = 1 - cosine(cur_infini, eligible_infinigram[i])
                if idx < sim_vec_length:
                    sim_vector[idx] = similarity
        
        eng1000_vec = eng_vectors[ind]
        final_vector = np.concatenate((eng1000_vec, sim_vector))
        newdata.append(final_vector)

    return np.array(newdata)

def make_word_ds(grids, trfiles, bad_words=DEFAULT_BAD_WORDS):
    """Creates DataSequence objects containing the words from each grid, with any words appearing
    in the [bad_words] set removed.
    """
    ds = dict()
    stories = list(set(trfiles.keys()) & set(grids.keys()))
    for st in stories:
        grtranscript = grids[st].tiers[1].make_simple_transcript()
        ## Filter out bad words
        goodtranscript = [x for x in grtranscript
                          if x[2].lower().strip("{}").strip() not in bad_words]
        d = DataSequence.from_grid(goodtranscript, trfiles[st][0])
        ds[st] = d

    return ds

def make_phoneme_ds(grids, trfiles):
    """Creates DataSequence objects containing the phonemes from each grid.
    """
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[0].make_simple_transcript()
        d = DataSequence.from_grid(grtranscript, trfiles[st][0])
        ds[st] = d

    return ds

phonemes = ['AA', 'AE','AH','AO','AW','AY','B','CH','D', 'DH', 'EH', 'ER', 'EY', 
            'F', 'G', 'HH', 'IH', 'IY', 'JH','K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 
            'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']

def make_character_ds(grids, trfiles):
    ds = dict()
    stories = grids.keys()
    for st in stories:
        grtranscript = grids[st].tiers[2].make_simple_transcript()
        fixed_grtranscript = [(s,e,map(int, c.split(","))) for s,e,c in grtranscript if c]
        d = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
        ds[st] = d
    return ds

def make_dialogue_ds(grids, trfiles):
    ds = dict()
    for st, gr in grids.iteritems():
        grtranscript = gr.tiers[3].make_simple_transcript()
        fixed_grtranscript = [(s,e,c) for s,e,c in grtranscript if c]
        ds[st] = DataSequence.from_grid(fixed_grtranscript, trfiles[st][0])
    return ds

def histogram_phonemes(ds, phonemeset=phonemes):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = ds.data
    N = len(ds.data)
    newdata = np.zeros((N, len(phonemeset)))
    phind = dict(enumerate(phonemeset))
    for ii,ph in enumerate(olddata):
        try:
            #ind = phonemeset.index(ph.upper().strip("0123456789"))
            ind = phind[ph.upper().strip("0123456789")]
            newdata[ii][ind] = 1
        except Exception as e:
            pass

    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def histogram_phonemes2(ds, phonemeset=phonemes):
    """Histograms the phonemes in the DataSequence [ds].
    """
    olddata = np.array([ph.upper().strip("0123456789") for ph in ds.data])
    newdata = np.vstack([olddata==ph for ph in phonemeset]).T
    return DataSequence(newdata, ds.split_inds, ds.data_times, ds.tr_times)

def make_semantic_model(ds: DataSequence, lsasms: list, sizes: list):
    """
    ds
        datasequence to operate on
    lsasms
        semantic models to use
    sizes
        sizes of resulting vectors from each semantic model
    """
    newdata = []
    num_lsasms = len(lsasms)
    for w in ds.data:
        v = []
        for i in range(num_lsasms):
            lsasm = lsasms[i]
            size = sizes[i]
            try:
                v = np.concatenate((v, lsasm[str.encode(w.lower())]))
            except KeyError as e:
                v = np.concatenate((v, np.zeros((size)))) #lsasm.data.shape[0],))
        newdata.append(v)
    return DataSequence(np.array(newdata), ds.split_inds, ds.data_times, ds.tr_times)

def make_character_model(dss):
    """Make character indicator model for a dict of datasequences.
    """
    stories = dss.keys()
    storychars = dict([(st,np.unique(np.hstack(ds.data))) for st,ds in dss.iteritems()])
    total_chars = sum(map(len, storychars.values()))
    char_inds = dict()
    ncharsdone = 0
    for st in stories:
        char_inds[st] = dict(zip(storychars[st], range(ncharsdone, ncharsdone+len(storychars[st]))))
        ncharsdone += len(storychars[st])

    charmodels = dict()
    for st,ds in dss.iteritems():
        charmat = np.zeros((len(ds.data), total_chars))
        for ti,charlist in enumerate(ds.data):
            for char in charlist:
                charmat[ti, char_inds[st][char]] = 1
        charmodels[st] = DataSequence(charmat, ds.split_inds, ds.data_times, ds.tr_times)

    return charmodels, char_inds

def make_dialogue_model(ds):
    return DataSequence(np.ones((len(ds.data),1)), ds.split_inds, ds.data_times, ds.tr_times)

def modulate(ds, vec):
    """Multiplies each row (each word/phoneme) by the corresponding value in [vec].
    """
    return DataSequence((ds.data.T*vec).T, ds.split_inds, ds.data_times, ds.tr_times)

def catmats(*seqs):
    keys = seqs[0].keys()
    return dict([(k, DataSequence(np.hstack([s[k].data for s in seqs]), seqs[0][k].split_inds)) for k in keys])
