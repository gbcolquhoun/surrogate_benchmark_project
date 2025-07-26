python3 surrogate_benchmark_project/make_iwslt_split.py \
  --de 2014-01/texts/de/en/de-en/IWSLT14.TED*.de-en.de.xml \
  --en 2014-01/texts/de/en/de-en/IWSLT14.TED*.de-en.en.xml \
  --n 2000 \
  --out data/text/iwslt14_de_en/dev2000


  python3 -m fairseq_cli.preprocess \
  --source-lang de --target-lang en \
  --trainpref  data/text/iwslt14_de_en/dev1000 \
  --validpref  data/text/iwslt14_de_en/dev100 \
  --destdir    data/binary/iwslt14_de_en_dev1000_v2 \
  --joined-dictionary \
  --srcdict    data/binary/iwslt14_de_en_dev1000/dict.de.txt \
  --workers 4

python3 -m fairseq_cli.preprocess \
  --source-lang de --target-lang en \
  --validpref  data/text/iwslt14_de_en/dev1000 \
  --destdir    data/binary/iwslt14_de_en_valid1000_origdict \
  --joined-dictionary \
  --srcdict    data/binary/iwslt14_de_en/dict.de.txt \
  --workers 4

ls $VIRTUAL_ENV/bin | grep fairseq

PYTHONPATH="" \
python -m fairseq_cli.preprocess  \
  --source-lang de --target-lang en \
  --validpref data/text/iwslt14_de_en/dev1000 \
  --destdir   data/binary/iwslt14_de_en_valid1000_origdict_nojoin \
  --srcdict   hardware-aware-transformers/data/binary/iwslt14_de_en/dict.de.txt \
  --workers 4

  PYTHONPATH="" \
  python -m fairseq_cli.preprocess \
  --source-lang de --target-lang en \
  --validpref  data/text/iwslt14_de_en/dev1000      \
  --destdir    data/binary/iwslt14_de_en_valid1000 \
  --srcdict    hardware-aware-transformers/data/binary/iwslt14_de_en/dict.de.txt \
  --tgtdict    hardware-aware-transformers/data/binary/iwslt14_de_en/dict.en.txt \
  --workers 4


  # ▶︎  Use upstream Fairseq CLI (for preprocess, generate, etc.)
PYTHONPATH="" python -m fairseq_cli.preprocess  [...]

# ▶︎  Run HAT code (needs the fork)
export PYTHONPATH="$HOME/Documents/hardware-aware-transformers:$PYTHONPATH"
python surrogate_benchmark_project/main.py