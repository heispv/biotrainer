# Changed version from the original one from Konstantin Schütze (konstin, https://github.com/konstin) from
# bio_embeddings repository (https://github.com/sacdallago/bio_embeddings)
# Original file: https://github.com/sacdallago/bio_embeddings/blob/efb9801f0de9b9d51d19b741088763a7d2d0c3a2/bio_embeddings/embed/one_hot_encoding_embedder.py

import numpy as np

from .embedder_interfaces import EmbedderInterface

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWXY"
index_map = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
template = np.eye(len(AMINO_ACIDS), dtype=np.float32)


class OneHotEncodingEmbedder(EmbedderInterface):
    """
    Implements one-hot encoding for amino acid sequences as a baseline embedder.

    This class provides a naive baseline implementation for one-hot encoding of amino acids, used
    to compare various input types or training methodologies in bioinformatics. The class supports
    optional parameters for consistency with more complex models, though options like device
    specifications are not utilized here.

    Attributes:
    -----------
    number_of_layers : int
        Number of encoding layers; here it is always 1 as only a single layer is used for encoding.
    embedding_dimension : int
        Dimension of the embedding, equal to the number of recognized amino acids.
    name : str
        Name of the embedder, set to 'one_hot_encoding'.

    Methods:
    --------
    _embed_single(sequence: str) -> ndarray
        Encodes a single amino acid sequence into a one-hot encoded numpy array.
    reduce_per_protein(embedding: ndarray) -> ndarray
        Computes the average one-hot encoding across a sequence to represent its composition.
    """

    number_of_layers = 1
    embedding_dimension = len(AMINO_ACIDS)
    name = "one_hot_encoding"

    def _embed_single(self, sequence: str) -> np.ndarray:
        """
        Generates a one-hot encoded array for a given amino acid sequence.

        Parameters:
        -----------
        sequence : str
            A string representing a valid amino acid sequence.

        Returns:
        --------
        ndarray
            A numpy array where each row corresponds to an amino acid in the sequence,
            represented as a one-hot vector.
        """
        indices = [index_map[aa] for aa in sequence]
        return template[indices]

    @staticmethod
    def reduce_per_protein(embedding: np.ndarray) -> np.ndarray:
        """
        Calculates the mean of one-hot encoded vectors of a protein to get its average composition.

        Parameters:
        -----------
        embedding : ndarray
            A numpy array of one-hot encoded vectors for a sequence.

        Returns:
        --------
        ndarray
            A single vector that represents the average presence of each amino acid in the sequence.

        Notes:
        ------
        - This function is useful for reducing the sequence representation to a fixed-size vector,
          irrespective of the sequence length.
        """
        return embedding.mean(axis=0)
