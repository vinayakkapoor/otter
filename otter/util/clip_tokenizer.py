import torch
from typing import Union, List, OrderedDict
from packaging import version
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()
# Save the original bpe function
_original_tokenizer_bpe = _tokenizer.bpe
_tokenizer_lru_cache = OrderedDict()

def get_tokenizer():
    return _tokenizer

def create_lru_cache_bpe_wrapper(lru_cache_maxsize):    
    def _lru_cache_bpe_wrapper(token):
        if token in _tokenizer_lru_cache:
            # Cache hit: REturn the cached bpe
            _tokenizer_lru_cache.move_to_end(token)
            return _tokenizer_lru_cache[token]
        else:
            # Cache miss: Use the original bpe function to generate the bpe
            result = _original_tokenizer_bpe(token)
            _tokenizer_lru_cache[token] = result

            # Remove the least recently used item to maintain cache max size
            if len(_tokenizer_lru_cache) > lru_cache_maxsize:
                _tokenizer_lru_cache.popitem(last=False) 

            return result
    
    return _lru_cache_bpe_wrapper

def enable_lru_caching(lru_cache_maxsize):
    # Monkey patch the byte pair encoder
    _tokenizer.bpe = create_lru_cache_bpe_wrapper(lru_cache_maxsize)

# Unchanged from https://github.com/openai/CLIP/blob/main/clip/clip.py
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result