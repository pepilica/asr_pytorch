import re
from collections import defaultdict
from string import ascii_lowercase

import torch
from torchaudio.models.decoder._ctc_decoder import ctc_decoder

# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = "^"

    def __init__(
        self,
        alphabet=None,
        use_torchaudio_ctc=False,
        top_k_beams=10,
        use_lm=False,
        lm_path=None,
        **kwargs,
    ):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        assert (lm_path is None) != use_lm, ""

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.use_torchaudio_ctc = use_torchaudio_ctc
        if self.use_torchaudio_ctc:
            CTCTextEncoder.decoder = ctc_decoder(
                lexicon=None,
                tokens=self.vocab,
                lm=lm_path,
                beam_size=top_k_beams,
                blank_token=self.EMPTY_TOK,
                sil_token=self.EMPTY_TOK,
            )
        self.top_k_beams = top_k_beams
        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        raw_string = self.decode(inds)
        if len(raw_string) == 0:
            return raw_string
        total_string = [raw_string[0]]
        for i in range(1, len(raw_string)):
            if raw_string[i] != raw_string[i - 1]:
                total_string.append(raw_string[i])
        return "".join(total_string)

    def beam_search_ctc_decode(self, log_probs, log_probs_length) -> str:
        if self.use_torchaudio_ctc:
            beam_search_result = CTCTextEncoder.decoder(log_probs, log_probs_length)
            hypos = [i.tokens for i in beam_search_result[0]]
        else:
            hypos = self.__beam_search_ctc_decode(
                log_probs, log_probs_length, self.top_k_beams
            )
        return hypos

    def __beam_search_ctc_decode(
        self, log_probs_batch, log_probs_length_batch, top_k_beams
    ) -> list[str]:
        if isinstance(log_probs_batch, torch.Tensor):
            log_probs_batch = log_probs_batch.detach().cpu().numpy()
        hypos = []
        for log_probs_non_normed, log_probs_length in zip(
            log_probs_batch, log_probs_length_batch
        ):
            log_probs = log_probs_non_normed[: int(log_probs_length)]
            dp = {
                (tuple(), self.EMPTY_TOK): 1.0,
            }
            for prob in log_probs:
                new_dp = defaultdict(float)
                for ind, next_token_prob in enumerate(prob):
                    cur_char = self.ind2char[ind]
                    for (prefix, last_char), v in dp.items():
                        if last_char == cur_char:
                            new_prefix = prefix
                        else:
                            if cur_char != self.EMPTY_TOK:
                                new_prefix = prefix + (ind,)
                            else:
                                new_prefix = prefix
                        new_dp[(new_prefix, cur_char)] += v * next_token_prob
                dp = dict(
                    sorted(list(new_dp.items()), key=lambda x: -x[1])[:top_k_beams]
                )
            hypos.append(max(dp.items(), key=lambda x: x[1])[0][0])
        return hypos

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
