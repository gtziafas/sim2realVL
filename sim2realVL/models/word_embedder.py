from ..types import *
import torch

WordEmbedder = Map[List[str], array]


# pre-trained GloVe embeddings with glove_dim=300
def glove_embeddings(version: str = 'md') -> WordEmbedder:
    import spacy
    _glove = spacy.load(f'en_core_web_{version}') 
    print('Loaded spacy word embeddings...')
    def embedd(sent: List[str]) -> array:
        sent_proc = _glove(sent)
        return array([word.vector for word in sent_proc])
    def embedd_many(sents: List[str]) -> List[array]:
        return list(map(embedd, sents))
    return embedd_many


# make word embedder function
def make_word_embedder(embeddings: str = 'glove_md', **kwargs) -> WordEmbedder:
    if embeddings.startswith('glove_'):
        version = embeddings.split('_')[1]
        if version not in ['md', 'lg']:
            raise ValueError('See utils/embeddings.py for valid embedding options')
        embedder = glove_embeddings(version)

    else:
        raise ValueError('See utils/embeddings.py for valid embedding options')

    return embedder