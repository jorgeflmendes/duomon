# Battle Transformer Experiment

Isolated sequence-modeling experiment for curated Pokemon/VGC battle data.

This is intentionally separate from the production CTDE pipeline. The goal is to
pre-train a battle-log transformer that can later be adapted into:

- next-action priors;
- trajectory scoring;
- communication/intent embeddings;
- CTDE feature augmentation.

## Research Basis

- Decision Transformer: casts RL as conditional sequence modeling over
  return/state/action trajectories.
- Trajectory Transformer: models full trajectories autoregressively and can use
  beam-style planning.
- VGC-Bench: provides human-play VGC datasets and standardized Pokemon
  evaluation context.
- Metamon/Pokemon transformer work: supports treating Pokemon battles as long
  text/trajectory sequences for offline policy learning.

## Files

- `prepare_battle_corpus.py`: converts the curated battle dataset into
  train/validation text corpora and trains a ByteLevel BPE tokenizer.
- `train_battle_lm.py`: trains a compact causal GPT-style model on the corpus.

Generated corpora, tokenizers, checkpoints, and metrics should live under
`outputs/transformer_training/` and are not committed.
