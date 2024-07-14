"""
Abstract interface for Embedder.
File adapted from bio_embeddings (https://github.com/sacdallago/bio_embeddings/blob/efb9801f0de9b9d51d19b741088763a7d2d0c3a2/bio_embeddings/embed/embedder_interfaces.py)
Original Authors:
  Christian Dallago (sacdallago)
  Konstantin Schuetze (konstin)
"""

import abc
import torch
import logging
import regex as re

from typing import List, Generator, Optional, Iterable, Any, Union
from numpy import ndarray

logger = logging.getLogger(__name__)


class EmbedderInterface(abc.ABC):
    name: str
    _device: Union[None, str, torch.device]

    @abc.abstractmethod
    def _embed_single(self, sequence: str) -> ndarray:
        """
        Generate an embedding for a given amino acid sequence.

        Parameters:
        -----------
        sequence : str
            A string representing a valid amino acid sequence.

        Returns:
        --------
        ndarray
            A NumPy array representing the embedding of the sequence.

        Notes:
        ------
        - This method should be implemented by subclasses.
        """
        pass

    @staticmethod
    def _preprocess_sequences(sequences: Iterable[str]) -> List[str]:
        """
        Preprocess a list of amino acid sequences by replacing rare amino acids.

        Parameters:
        -----------
        sequences : Iterable[str]
            An iterable collection of strings, each representing an amino acid sequence.

        Returns:
        --------
        List[str]
            A list of strings where rare amino acids (U, Z, O, B) have been replaced with 'X'.

        Notes:
        ------
        - The replacement helps standardize sequences for further processing.
        - Rare amino acids U (Selenocysteine), Z (Glutamic acid or Glutamine), O (Pyrrolysine),
          and B (Asparagine or Aspartic acid) are replaced because they may not be recognized
          by some processing functions or databases.
        """
        sequences_cleaned = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
        return sequences_cleaned

    def _embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        """
        Compute embeddings for all sequences in a given batch.

        This method serves as a placeholder and is intended to be overridden by subclasses
        to implement specific batching strategies suitable for the model in use. The current
        implementation iterates through the batch and yields embeddings for each sequence
        individually using the _embed_single method.

        Parameters:
        -----------
        batch : List[str]
            A list of amino acid sequences to be embedded.

        Yields:
        -------
        Generator[ndarray, None, None]
            A generator yielding the ndarray embeddings for each sequence in the batch.
        """
        for sequence in batch:
            yield self._embed_single(sequence)

    def embed_many(
            self, sequences: Iterable[str], batch_size: Optional[int] = None
    ) -> Generator[ndarray, None, None]:
        """
        Yields embeddings for each protein sequence, processed either individually or in batches.

        This method handles sequences by either processing each one individually or in batches, depending 
        on the batch_size provided. Sequences longer than the batch_size are processed individually with a 
        warning. It preprocesses the sequences to standardize them before embedding.

        Parameters:
        -----------
        sequences : Iterable[str]
            An iterable of protein sequences represented as strings of amino acids.
        batch_size : Optional[int]
            The maximum number of amino acids per batch. If None, all sequences are processed individually.

        Yields:
        -------
        Generator[ndarray, None, None]
            A generator yielding the ndarray embeddings for each sequence. The embedding process might be 
            batched for efficiency if a batch_size is provided.
        """
        sequences = self._preprocess_sequences(sequences)

        if batch_size:
            batch = []
            length = 0
            for sequence in sequences:
                if len(sequence) > batch_size:
                    logger.warning(
                        f"A sequence is {len(sequence)} residues long, "
                        f"which is longer than your `batch_size` parameter which is {batch_size}"
                    )
                    yield from self._embed_batch([sequence])
                    continue
                if length + len(sequence) >= batch_size:
                    yield from self._embed_batch(batch)
                    batch = []
                    length = 0
                batch.append(sequence)
                length += len(sequence)
            yield from self._embed_batch(batch)
        else:
            for seq in sequences:
                yield self._embed_single(seq)

    @staticmethod
    def reduce_per_protein(embedding: ndarray) -> ndarray:
        """
        For a variable size embedding, returns a fixed size embedding encoding all information of a sequence.

        :param embedding: the embedding
        :return: A fixed size embedding (a vector of size N, where N is fixed)
        """
        return embedding.mean(axis=0)


class EmbedderWithFallback(EmbedderInterface, abc.ABC):
    """ Batching embedder that will fallback to the CPU if the embedding on the GPU failed """

    _model: Any

    @abc.abstractmethod
    def _embed_batch_implementation(
            self, batch: List[str], model: Any
    ) -> Generator[ndarray, None, None]:
        ...

    @abc.abstractmethod
    def _get_fallback_model(self):
        """Returns a (cached) cpu model.

        Note that the fallback models generally don't support half precision mode and therefore ignore
        the `half_precision_model` option (https://github.com/huggingface/transformers/issues/11546).
        """
        ...

    def _embed_batch(self, batch: List[str]) -> Generator[ndarray, None, None]:
        """Tries to get the embeddings in this order:
          * Full batch GPU
          * Single Sequence GPU
          * Single Sequence CPU

        Single sequence processing is done in case of runtime error due to
        a) very long sequence or b) too large batch size
        If this fails, you might want to consider lowering batch_size and/or
        cutting very long sequences into smaller chunks

        Returns unprocessed embeddings
        """
        # No point in having a fallback model when the normal model is CPU already
        if self._device.type == "cpu":
            yield from self._embed_batch_implementation(batch, self._model)
            return

        try:
            yield from self._embed_batch_implementation(batch, self._model.to(self._device))
        except RuntimeError as e:
            if len(batch) == 1:
                logger.warning(
                    f"RuntimeError for sequence with {len(batch[0])} residues: {e}. "
                    f"This most likely means that you don't have enough GPU RAM to embed a protein this long. "
                    f"Embedding on the CPU instead, which is very slow"
                )
                yield from self._embed_batch_implementation(batch, self._get_fallback_model())
            else:
                logger.warning(
                    f"Error processing batch of {len(batch)} sequences: {e}. "
                    f"You might want to consider adjusting the `batch_size` parameter. "
                    f"Will try to embed each sequence in the set individually on the GPU."
                )
                for sequence in batch:
                    try:
                        yield from self._embed_batch_implementation([sequence], self._model.to(self._device))
                    except RuntimeError as e:
                        logger.warning(
                            f"RuntimeError for sequence with {len(sequence)} residues: {e}. "
                            f"This most likely means that you don't have enough GPU RAM to embed a protein this long. "
                            f"Embedding on the CPU instead, which is very slow"
                        )
                        yield from self._embed_batch_implementation([sequence], self._get_fallback_model())
