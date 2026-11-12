# English-to-French Translation with a Transformer from Scratch

An encoder–decoder Transformer implemented in **TensorFlow/Keras** for English-to-French machine translation. The project builds the core Transformer components—including scaled dot-product attention, multi-head attention, positional encodings, encoder/decoder blocks, and masking—rather than relying on a pretrained translation model.

The project also compares **greedy decoding** and **beam search** at inference time using corpus BLEU.

## Results

| Metric | Result |
|---|---:|
| Training loss (epoch 5) | 1.3387 |
| Validation loss (epoch 5) | 1.3188 |
| Training masked token accuracy | 75.07% |
| Validation masked token accuracy | 75.72% |
| Greedy decoding BLEU | 19.81 |
| Beam search BLEU | **22.90** |
| Beam-search gain | **+3.08 BLEU points** |

**BLEU evaluation:** SacreBLEU corpus BLEU on the same randomly sampled **100 held-out test sentences** (`random_state=42`). These are preliminary sample-level results, not a full-test-set benchmark. The BLEU gain is approximately **15.6% relative to greedy decoding**.

## Project overview

The model translates English sentences into French using a sequence-to-sequence Transformer:

```text
English sentence
    ↓
English BPE tokenizer → token IDs
    ↓
Token embedding + sinusoidal positional encoding
    ↓
Transformer encoder (2 layers)
    ↓
Contextual English representations
    ↓
Transformer decoder (2 layers)
    ↑                     ↑
Previously generated     Encoder output
French tokens            (cross-attention)
    ↓
Linear vocabulary projection → next-token logits
    ↓
Greedy decoding OR beam search
    ↓
French BPE decoder → French sentence
```

During training, the decoder receives shifted ground-truth French tokens (**teacher forcing**) and uses a causal mask so each position cannot attend to future target tokens. During inference, the model generates French tokens autoregressively, starting with `[START]` and stopping at `[END]` or the maximum length.

## Dataset and preprocessing

- **Dataset:** [ManyThings French–English sentence pairs](https://www.manythings.org/anki/) (`fra-eng.zip`, `fra.txt`).
- **Split:** 80% training / 10% validation / 10% test.
- **Tokenization:** Separately trained English and French byte-pair encoding (BPE) tokenizers using Hugging Face `tokenizers`.
- **Vocabulary:** Approximately 8,000 tokens per language.
- **Special tokens:** `[PAD]`, `[UNK]`, `[START]`, `[END]`.
- **Maximum sequence length:** 32 tokens.
- **Padding:** Token ID `0` for `[PAD]`.
- **Training examples:** Approximately 192,236 after sequence-length filtering.

The English input ends with `[END]`. For French, the decoder input begins with `[START]`, while the training target is shifted one token ahead and ends with `[END]`.

## Model configuration

| Component | Configuration |
|---|---|
| Architecture | Encoder–decoder Transformer |
| Encoder layers | 2 |
| Decoder layers | 2 |
| Model dimension (`d_model`) | 128 |
| Attention heads | 4 |
| Dimension per head | 32 |
| Feed-forward hidden dimension (`dff`) | 512 |
| Dropout | 0.1 |
| Positional encoding | Fixed sinusoidal |
| Output layer | Dense projection to French vocabulary logits |
| Maximum sequence length | 32 |

### Components implemented

1. **Scaled dot-product attention:** Computes `softmax(QKᵀ / sqrt(d_k) + mask) V`.
2. **Multi-head attention:** Projects queries, keys, and values into four heads, computes attention per head, concatenates results, and applies an output projection.
3. **Encoder block:** Self-attention, residual connections, layer normalization, and a position-wise feed-forward network.
4. **Decoder block:** Masked self-attention, encoder–decoder cross-attention, residual connections, normalization, and feed-forward network.
5. **Masks:** Encoder padding masks, decoder causal/look-ahead masks, and cross-attention padding masks.
6. **Sinusoidal positional encoding:** Adds token-position information to embeddings.

## Training

- **Framework:** TensorFlow / Keras.
- **Optimizer:** Adam (`beta_1=0.9`, `beta_2=0.98`, `epsilon=1e-9`).
- **Learning rate:** Transformer inverse-square-root schedule with 4,000 warmup steps.
- **Batch size:** 32.
- **Epochs completed:** 5.
- **Loss:** Sparse categorical cross-entropy from logits, excluding padding tokens.
- **Accuracy:** Token-level accuracy excluding padding tokens.
- **Checkpoints:** Weights saved after each epoch to `checkpoints/transformer_epoch_XX.weights.h5`.

**Note:** Masked token accuracy uses the correct preceding French tokens during evaluation. It is not the percentage of fully correct translated sentences. BLEU is computed separately from autoregressively generated translations.

## Inference: greedy decoding vs. beam search

Both strategies use the **same trained Transformer weights**. They differ only in how they choose output tokens.

### Greedy decoding

At each decoding step, select the French token with the highest logit/probability (`argmax`), append it to the generated sequence, and continue. Only one partial translation is retained.

### Beam search

Maintain multiple promising partial translations and expand each using the model's next-token log probabilities. Rank candidates using cumulative log probability with length normalization and retain the top beams.

- **Beam width:** 4.
- **Length-penalty parameter:** `alpha = 0.6`.
- **Final output:** The single highest-scoring translation, not all four beams.

| Decoding method | Corpus BLEU (100-sentence sample) |
|---|---:|
| Greedy | 19.81 |
| Beam search (width 4, α = 0.6) | **22.90** |

Beam search increased BLEU by **3.08 points** on this sample, at the cost of slower inference. The initial beam-search implementation runs the full Transformer separately for each active candidate, recomputing encoder representations; caching encoder outputs and batching beams are useful future optimizations.

## Example usage

After running the notebook to initialize the model, load its trained weights if necessary, and load both tokenizers:

```python
english = "I love you."

print("Greedy:", translate_greedy(english))
print("Beam:  ", translate_beam(english, beam_width=4))
```

The `translate_greedy` and `translate_beam` functions are defined in the notebook. **The tokenizer files, architecture, and trained weights must be available to reproduce inference.** Checkpoints contain weights only; they do not replace the model definition or tokenizer files.

### Dependencies

```bash
pip install tensorflow tokenizers pandas numpy matplotlib sacrebleu
```

Run the notebook cells in order to prepare the dataset, train the BPE tokenizers, construct and train the Transformer, and evaluate translation quality. If you have already trained the model, reconstruct its architecture and load the corresponding checkpoint weights instead of retraining.

## Recommended visualizations

- Training vs. validation loss by epoch.
- Training vs. validation masked token accuracy by epoch.
- Bar chart comparing greedy and beam-search BLEU.
- Side-by-side table: English source, reference French, greedy translation, and beam-search translation.
- Optional decoder cross-attention heatmap showing alignment between generated French tokens and English input tokens (requires exposing attention weights).

## Limitations and next steps

- **Evaluation sample:** BLEU is currently based on only 100 test sentences. Evaluate on a larger, fixed held-out subset or the full test set for more stable results.
- **Dataset comparability:** Results from the ManyThings dataset are not directly comparable to WMT benchmarks.
- **Model scale:** The model has only two encoder and decoder layers, `d_model=128`, and five completed epochs; translation quality may improve with further training and tuning.
- **Inference performance:** Reuse encoder output and batch beam candidates to avoid redundant computation.
- **Translation quality:** Inspect generated examples, including errors, alongside BLEU; lexical overlap does not fully measure semantic correctness.
- **Reproducibility:** Save tokenizer JSON files, final model weights, package versions, and any preprocessing/normalization settings.

## Reference

Vaswani, A., et al. (2017). **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)**. *Advances in Neural Information Processing Systems (NeurIPS).* The original Transformer paper used beam search for translation decoding, including beam size 4 and length penalty α = 0.6. Its reported translation benchmarks were obtained on different datasets and with substantially larger models and training budgets; they should **not** be interpreted as directly comparable to this project's BLEU results.

---

**Project type:** Educational implementation and experimental evaluation of an encoder–decoder Transformer for neural machine translation.
