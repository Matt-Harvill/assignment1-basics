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

### Problem (train_bpe_expts_owt):
1. Ran out of RAM
2. Ran out of RAM
