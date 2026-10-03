"""mathematicskit.information_theory: entropy, source coding, channel capacity, and error-correcting codes.

Hartley's measure, Shannon entropy, joint/conditional entropy, mutual
information, and the Kullback-Leibler divergence via
``scipy.stats.entropy``/``scipy.special.entr``; the typical set behind
Shannon's source coding theorem, the Kraft-McMillan inequality,
Shannon-Fano and Huffman codes, Elias gamma codes, arithmetic coding,
and LZ78; channel capacity of the binary symmetric, binary erasure and
Gaussian (Shannon-Hartley) channels, the Blahut-Arimoto algorithm, a
random-coding illustration of the noisy-channel coding theorem, and
rate-distortion functions; Hamming codes, convolutional codes with
Viterbi decoding, and Gallager's LDPC codes with bit-flipping decoding
-- the coders and decoders hand-rolled, with no scipy/numpy equivalent.
Reed-Solomon codes live in :mod:`mathematicskit.abstract_algebra`.
"""

from mathematicskit import __version__
from mathematicskit.information_theory.core.base import BitFlipResult, ChannelCapacityResult, PrefixCode, TypicalSetResult
from mathematicskit.information_theory.systems.channel_codes import (
    bit_flip_decode,
    convolutional_encode,
    gallager_ldpc_matrix,
    hamming_decode,
    hamming_encode,
    hamming_parity_check_matrix,
    viterbi_decode,
)
from mathematicskit.information_theory.systems.channels import (
    awgn_capacity,
    bec_capacity,
    blahut_arimoto,
    bsc_capacity,
    minimum_ebn0,
    random_code_error_rate,
    rate_distortion_binary,
    rate_distortion_gaussian,
)
from mathematicskit.information_theory.systems.entropy import (
    binary_entropy,
    conditional_entropy,
    entropy,
    hartley_information,
    joint_entropy,
    kl_divergence,
    mutual_information,
)
from mathematicskit.information_theory.systems.source_coding import (
    arithmetic_decode,
    arithmetic_encode,
    arithmetic_intervals,
    canonical_code,
    elias_gamma_decode,
    elias_gamma_encode,
    huffman_code,
    kraft_sum,
    lz78_compressed_bits,
    lz78_decode,
    lz78_encode,
    shannon_fano_code,
    typical_set,
)
from mathematicskit.information_theory.utils.channel_noise import bsc_transmit, hamming_distance

__all__ = [
    "__version__",
    "PrefixCode",
    "ChannelCapacityResult",
    "TypicalSetResult",
    "BitFlipResult",
    "hartley_information",
    "entropy",
    "binary_entropy",
    "joint_entropy",
    "conditional_entropy",
    "mutual_information",
    "kl_divergence",
    "typical_set",
    "kraft_sum",
    "canonical_code",
    "shannon_fano_code",
    "huffman_code",
    "elias_gamma_encode",
    "elias_gamma_decode",
    "arithmetic_intervals",
    "arithmetic_encode",
    "arithmetic_decode",
    "lz78_encode",
    "lz78_decode",
    "lz78_compressed_bits",
    "bsc_capacity",
    "bec_capacity",
    "awgn_capacity",
    "minimum_ebn0",
    "blahut_arimoto",
    "random_code_error_rate",
    "rate_distortion_binary",
    "rate_distortion_gaussian",
    "hamming_parity_check_matrix",
    "hamming_encode",
    "hamming_decode",
    "convolutional_encode",
    "viterbi_decode",
    "gallager_ldpc_matrix",
    "bit_flip_decode",
    "bsc_transmit",
    "hamming_distance",
]
