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

### Problem (transformer_block): Implement the Transformer block (3 points)

### Problem (transformer_lm): Implementing the Transformer LM (3 points)

### Problem (transformer_accounting): Transformer LM resource accounting (5 points)
