import math
import itertools
import numpy as np

def total_space():
    # constants
    ENC_LAYER_CHOICES = [6]
    DEC_LAYER_CHOICES = [6,5,4,3,2,1]

    ENC_EMBED_CHOICES = [640, 512]
    DEC_EMBED_CHOICES = [640, 512]

    enc_ffn  = 3   # 2048/1024/512
    dec_ffn  = 3
    enc_sa   = 2   # 4 / 2
    dec_sa   = 2
    dec_ende = 2
    dec_arbi = 3

    total = 0
    for d in DEC_LAYER_CHOICES:
        part = (
            len(ENC_LAYER_CHOICES) * len(DEC_LAYER_CHOICES) *   # layer nums
            len(ENC_EMBED_CHOICES) * len(DEC_EMBED_CHOICES) *   # embed dims
            (enc_ffn  ** 6) *                                   # 6 encoder slots
            (dec_ffn  ** d) *
            (enc_sa   ** 6) *
            (dec_sa   ** d) *
            (dec_ende ** d) *
            (dec_arbi ** d)
        )
        total += part
    return total

TOTAL_CONFIGS = total_space()
print(f"Total discrete configs ≈ {TOTAL_CONFIGS:,}")

# Total discrete configs ≈ 2,507,080,072,034,304