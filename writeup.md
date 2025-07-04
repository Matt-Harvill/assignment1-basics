# Writeup

## 2 Byte-Pair Encoding (BPE) Tokenizer

### Problem (unicode1): Understanding Unicode (1 point)
1. chr(0) returns '\x00'
2. We need the escape character for the backslash - here is the chr(0).__repr__(): "'\\x00'"
3. Printing chr(0) reveals that it appears to be an empty string

### Problem (unicode2): Unicode Encodings (3 points)
1. It's useful to use utf8 because it encodes ASCII characters with a single byte whereas utf32 uses 4 bytes for every character and utf16 dedicates 2-4 bytes per character. For other characters, utf16 can be more efficient, but not for common characters.
2. Since utf8 can have variable length (1-4) bytes per character, this incorrectly splits multi-byte characters into multiple characters. Here's a string that fails: "5 ÷ 2 = 2.5"
3. I understand the concept and won't spend my time looking this up at the moment.

### Problem (train_bpe): BPE Tokenizer Training (15 points) ✅

### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
1. It took about 4 minutes to train. The longest token is " accomplishment" with a length of 15, at index ~7000 which makes sense since it would generally make sense for the longest token to be later in the vocabulary.
2. Pretokenization takes the largest amount of time (communication syncing between all the processes).

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
1. Ran out of RAM
2. Ran out of RAM

### Problem (tokenizer): Implementing the tokenizer (15 points) ✅

### Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
1. TinyStories tokenizer compression ratio: 2.20 bytes/token
2. TinyStories compression ratio drops to 2.05 bytes/token on the OpenWeb text samples. This is because it wasn't trained to optimize for the distribution of byte pair merges in OpenWeb text.
3. It is about 1MB/s which is decent! It would take 233 hours to tokenize the pile at this rate.
4. I tokenized TinyStories valid and train text. 16 bits unsigned gives the range 0-65536 which encompasses both tokenizer vocab lengths. It's also half the storage of 32 bits per token.

## 3 Transformer Language Model Architecture

### Problem (linear): Implementing the linear module (1 point) ✅

### Problem (embedding): Implement the embedding module (1 point) ✅

### Problem (rmsnorm): Root Mean Square Layer Normalization (1 point) ✅

### Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points) ✅

### Problem (rope): Implement RoPE (2 points) ✅

### Problem (softmax): Implement softmax (1 point) ✅

### Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points) ✅

### Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points) ✅

### Problem (transformer_block): Implement the Transformer block (3 points) ✅

### Problem (transformer_lm): Implementing the Transformer LM (3 points) ✅

### Problem (transformer_accounting): Transformer LM resource accounting (5 points)
1. Number of trainable parameters = 2vd [embed + unembed] + (1 + 48(2))d [norm] + 48(4(d^2)) [attn] + 48(3(4d^2)) [ffn] = 2127057600. Since single precision is fp32 which is 4B/parameter, this equates to 8508230400 Bytes or ~8.5GB.
2. Matrix multiplications in attention are the projections: 8sd^2 FLOPs, kq scores: 2sd^2 FLOPs, value multiplication: 2s^2d FLOPs, outputs: 2sd^2 so 48(12sd^2 + 2s^2d) for all attention FLOPs. Then we also have ffn which is 2(3(4sd^2)) FLOPs for two up_proj and one down_proj. We have 48x of these too. Finally, the lm_head projection is 2sdv FLOPs. Total FLOPs is 4855591731200 or ~5TFLOPs
3. FFN take over half of total FLOPS (double attention FLOPS and 20x lm_head FLOPs)
4. The results are in calculations.py and basically the pattern that emerges with increasing total model size is the ratio of lm_head FLOPs decreases, attn FLOPs ratio decreases, and ffn FLOPs ratio increases. This makes sense since ffn and attn are mostly quadratically related to d_model and linearly with num_layers. And since lm_head is linearly related to d_model but vocab size is constant, it also makes sense why lm_head takes so many of smaller models' FLOPs.
5. Results are also in calculations.py. Since attention has a component that's quadratic w.r.t. context_length instead of all other components which are linearly related to context_length, we see that attention actually becomes the FLOPs bottleneck in this context_length and the large context regime generally.

## 4 Training a Transformer LM

### Problem (cross_entropy): Implement Cross entropy ✅

### Problem (learning_rate_tuning): Tuning the learning rate (1 point)
1. The loss decays faster and faster with higher learning rates, with lr=1e2 to be the best (convergent) lr. At lr=1e3, the loss is still low, but appears to be diverging.

### Problem (adamw): Implement AdamW (2 points) ✅

### Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)
1. Peak memory
Parameters
- Transformer Block (* num_layers)
    - RMSNorms (2 * d_model)
    - Q,K,V,O (4 * d_model^2)
    - W1,W2,W3 (12 * d_model^2)
- Final RMSNorm (d_model)
- Output embedding (d_model * vocab_size)
- Cross-entropy on logits (0)

Activations
- Transformer Block (* num_layers)
    - RMSNorms (2 * batch_size * context_length * d_model)
    - Attention
        - QKV projections (3 * batch_size * context_length * d_model)
        - Q^TK (batch_size * num_heads * context_length * context_length)
        - softmax (batch_size * num_heads * context_length * context_length)
        - weighted sum of values (batch_size * context_length * d_model)
        - output proj (batch_size * context_length * d_model)
    - W1 (batch_size * context_length * 4 * d_model)
    - SiLU (batch_size * context_length * 4 * d_model)
    - W2 (batch_size * context_length * d_model)
- Final RMSNorm (batch_size * context_length * d_model)
- Output embedding (batch_size * context_length * vocab_size)
- Cross-entropy on logits (0 extra)

Gradients
- 1 for each parameter
- 1 for each activation (for backprop as we go, so not all at once)

Optimizer State
- 1 for each parameter for "m" - first moments exponential average
- 1 for each parameter for "v" - second moments exponential average

Total Memory
- 4 * Parameters (weights, grads, m, v)
- ~1 * Activations (assuming no activation checkpointing and we need current activation grads for backprop)
- All of this multiplied by 4 bytes since fp32 has 4 bytes per scalar

In terms of all Variables:
M_bytes = 4 * (
    4 * (num_layers * (2 * d_model + 16 * d_model**2) + d_model * (1 + vocab_size)) +
    (16 * batch_size * num_layers * context_length * d_model) +
    (2 * batch_size * num_layers * context_length**2 * num_heads) +
    (batch_size * context_length * d_model) +
    (batch_size * context_length * vocab_size)
)

2. 15.311900672 * batch_size + 32.7463424 GB (from calculations.py). This means we can use batch_size = 3 to train on 80GB. This is highly cautious as we would probably not save all intermediate activations (including attention which is massive)

3. Since there are ~16 FLOPs per parameter for an AdamW step, this totals to 2127057600 * 16 = 34032921600 ~ 34GFLOPs.

4. Total FLOPS per sample is 4855591731200 so if we take 2x that in backward and add constant 34G from AdamW, with a batch size of 1024, a single step should take 14.9 petaFLOPs. Assuming 50% MFU with 19.5 teraFLOP/s as peak, this means we effectively achieve 9.75 teraFLOP/s making one step take 1568 seconds. Thus it would take ~7.2k days = 20 years on a single A100.

### Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup ✅

### Problem (gradient_clipping): Implement gradient clipping (1 point) ✅

## 5 Training Loop

### Problem (data_loading): Implement data loading (2 points) ✅

### Problem (checkpointing): Implement model checkpointing (1 point) ✅

### Problem (training_together): Put it together (4 points)

## 6 Generating text

### Problem (decoding): Decoding (3 points)

## 7 Experiments

### Problem (experiment_log): Experiment logging (3 points)

### Problem (learning_rate): Tune the learning rate (3 points) (4 H100 hrs)

### Problem (batch_size_experiment): Batch size variations (1 point) (2 H100 hrs)

### Problem (generate): Generate text (1 point)

### Problem (layer_norm_ablation): Remove RMSNorm and train (1 point) (1 H100 hr)

### Problem (pre_norm_ablation): Implement post-norm and train (1 point) (1 H100 hr)

### Problem (no_pos_emb): Implement NoPE (1 point) (1 H100 hr)

### Problem (swiglu_ablation): SwiGLU vs. SiLU (1 point) (1 H100 hr)

### Problem (main_experiment): Experiment on OWT (2 points) (3 H100 hrs)
