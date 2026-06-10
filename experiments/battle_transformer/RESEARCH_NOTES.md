# Battle Transformer Research Notes

## Direction

The current experiment follows the sequence-modeling line of offline RL:

- **Decision Transformer** treats RL as conditional sequence modeling over
  return/state/action histories.
- **Trajectory Transformer** models whole trajectories autoregressively and
  can support model-based planning over predicted futures.
- **VGC-Bench** establishes Pokemon VGC as a generalization-heavy benchmark
  with human battle logs and baseline agents.
- **Metamon** shows that transformer policies trained from Pokemon battle
  demonstrations can reach strong human-level play in competitive Pokemon.
- **PokeLLMon** supports text-based battle representations with consistency
  controls and knowledge/reasoning augmentation.

## Current Implementation

This folder currently trains a causal GPT-style language model over curated
battle text. It is not yet a deployed policy.

The first useful adaptation path is:

1. Train broad battle language model on all curated battle logs.
2. Fine-tune on high-precision/high-rating logs plus DuoMon winning traces.
3. Convert model activations into compact battle embeddings.
4. Add those embeddings as CTDE reranker features or a communication intent
   encoder.
5. Only then evaluate win-rate impact against the existing benchmark protocol.

## Known Limits

- VGC-Bench logs are single-match competitive VGC, not DuoMon cooperative A/B
  communication traces.
- The current model predicts battle text, not legal actions from a live
  Showdown request.
- The dataset has winners only after filtering; for value learning we still
  need losses/counterfactual candidates.
- This is a strong pretraining direction, not proof of improved win-rate until
  converted into the policy and benchmarked.
