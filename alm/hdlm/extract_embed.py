import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel

def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs

# encoder = AutoModel.from_pretrained("sentence-transformers/gtr-t5-base").encoder

# tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/gtr-t5-base")

encoder = AutoModel.from_pretrained("gpt2-xl")

tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")



vocab_tokens = tokenizer.get_vocab().keys()

# Convert vocabulary tokens to a list
vocab_tokens_list = list(vocab_tokens)

# Get the embeddings for each vocabulary token
embeddings_list = []
with torch.no_grad():
  for token in tqdm(vocab_tokens_list):
      tokens = tokenizer(token, return_tensors="pt")
      model_output = encoder(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
      hidden_state = model_output.last_hidden_state
      embeddings = mean_pool(hidden_state, tokens['attention_mask'])
      embeddings_list.append(embeddings)

print(len(embeddings_list))

final_embeddings = torch.cat(embeddings_list,dim=0)

print(final_embeddings.shape)

torch.save(final_embeddings, 'gpt2xl_embed.pt')

