from typing import Any
import regex
from collections import Counter


# spaces are included with words, 'not skipped', there is a difference between 'hello' and ' hello'
"""
From GPT-4 Explanation of this regex
This regular expression (regex) is a pattern used for tokenization in GPT-2, 
and it is designed to match and separate different types of tokens in a text. Let’s break down each part of the regex:

1. 's|'t|'re|'ve|'m|'ll|'d
These are contractions often used in English. The regex is designed to separate these suffixes as individual tokens.
For example, in "it’s", the regex will create two tokens: "it" and "’s".
2. ?[\p{L}]+
This part matches Unicode letter characters (alphabetic characters).
\p{L} represents any kind of letter from any language.
+ indicates one or more of the preceding element.
The optional space ? before the letters ensures that space-prefixed words are matched as separate tokens.
3. ?[\p{N}]+
This part matches Unicode number characters.
\p{N} represents any kind of numeric character in any script.
Similar to the letters, the numbers can also be optionally space-prefixed.
4. ?[^\s\p{L}\p{N}]+
This part matches sequences that are not whitespace, letters, or numbers.
^\s\p{L}\p{N} negates spaces, letters, and numbers, matching symbols and punctuation.
These characters, possibly prefixed by a space, are treated as separate tokens.
5. \s+(?!\S)|\s+
\s+ matches one or more whitespace characters.
(?!\S) is a negative lookahead assertion that ensures the matched spaces are not followed by non-space characters, preserving certain spaces as separate tokens.
Putting It All Together
This regex pattern is applied sequentially, and each part of the pattern aims to match different types of text sequences: 
contractions, words, numbers, symbols/punctuations, and spaces. By applying this regex pattern, a text can be broken down 
into a sequence of meaningful tokens, which is a common preprocessing step in natural language processing (NLP).

"""

# with special characters <PAD> and |endoftext|
# manually adding variations of special tokens to handle space being included in tokens
# GPT2_PATTERN = (
#     r"""'s|'t|'re|'ve|'m|'ll|'d|<PAD>| <PAD>|\|endoftext\|| \|endoftext\|| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# )

# we will use this approach where spaces are encoded separately
# Pattern to have no spaces for all words
# GPT2_PATTERN = (
#     r"""'s|'t|'re|'ve|'m|'ll|'d|<PAD>|\|endoftext\||[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# )


class ByteBPE:
    def __init__(self, special_tokens=None) -> None:
        
        
        if special_tokens:
            regex_special_tokens = [sp.replace("|", "\|") for sp in special_tokens]
            regex_special_tokens = "|".join(regex_special_tokens)
            
            GPT2_PATTERN = rf"""'s|'t|'re|'ve|'m|'ll|'d|{regex_special_tokens}|[\p{{L}}]+|[\p{{N}}]+|[^\s\p{{L}}\p{{N}}]+|\s+(?!\S)|\s+"""            
            
            self.special_tokens = [sp.encode("utf-8") for sp in special_tokens]
            self.ranks = {sp:idx for idx,sp in enumerate(self.special_tokens) }
            self.decode_ranks = {v:k for k,v in self.ranks.items()}
        else:
            GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            self.special_tokens = None
            self.ranks = {}
            self.decode_ranks = {}
                
            
        self.pattrn = regex.compile(GPT2_PATTERN)
        

    def train(self, filename: str, vocab_size: int):
        """
        ranks start from all bytes 0:255
        return: ranks, dict of byte_token_sequences: integer of rank capped at vocab size
        """
        if vocab_size < 256:
            raise Exception("must be greater than 256, minimum number of possible bytes")
        
        
        rank_starter = len(self.ranks)
        for i in range(256):
            self.ranks[bytes([i])] = i + rank_starter
        
        with open(filename, 'r') as f:
            text = f.read()
        
        # apply regex to text, get words, then convert each word to list of bytes
        words = self.pattrn.findall(text)
        # drop any special tokens found in text
        if self.special_tokens:
            words = [word for word in words if word.encode("utf-8") not in self.special_tokens]
            
        words_bytes = [[bytes([b]) for b in word.encode('utf-8')] for word in words]
        
        # keep iterating until vocab size is reached or no maximum sequences are found
        while len(self.ranks) < vocab_size:
            
            # compute most common pair of sequenced bytes and add it to ranks
            counter = Counter()
            for bword in words_bytes:
                # generate pairs 
                for pair in zip(bword[:-1], bword[1:]):
                    counter[pair] += 1
            
            # if counter is empty means no sequences are found, which means our vocab size are bigger than total possible tokens
            if not counter:
                break
            
            most_common_pair = max(counter, key=lambda k: counter[k])
           
            # gen new token based on merge and add it to ranks
            token = most_common_pair[0] + most_common_pair[1]
            self.ranks[token] = len(self.ranks)
            
            # merge all pairs based on new token and generate new list of words where pairs are merged
            new_words = []
            for b_word in words_bytes:
                i = 0
                new_word = []
                while i < len(b_word) - 1:
                    if (b_word[i], b_word[i + 1]) == most_common_pair:
                        # we found a pair
                        new_word.append(token)
                        # push index by 2
                        i += 2
                    else:
                        # add only byte at index i
                        new_word.append(b_word[i])
                        # push index by 1
                        i += 1
                # special check for last character at index i
                if i == len(b_word) - 1:
                    new_word.append(b_word[i])
                new_words.append(new_word)
            
            words_bytes = new_words
            
            # print(words_bytes)        
            
    
        self.decode_ranks = {v:k for k,v in self.ranks.items()}
    
    def encode(self, text: str):
        """
        # merge pairs sequentially until no more pairs found in ranks
        # pairs with lower ranks must be merged first
        """
        
        words = self.pattrn.findall(text)
        words_bytes = []
        # if a word is special token, add as is to the list
        for word in words:
            if self.special_tokens and word.encode("utf-8") in self.special_tokens:
                words_bytes.append(word.encode("utf-8"))
            else:
                words_bytes.append([bytes([b]) for b in word.encode("utf-8")])
                
        
        final_tokens = []
        for bword in words_bytes:
            if self.special_tokens and bword in self.special_tokens:
                final_tokens.append(bword)
            else:
                while True:
                    min_idx = None
                    min_rank = None
                    for i, pair in enumerate(zip(bword[:-1], bword[1:])):
                        rank = self.ranks.get((pair[0] + pair[1]))
                        if rank is not None and (min_rank is None or rank < min_rank):
                            min_rank = rank
                            min_idx = i
                        
                    
                    if min_rank is None:
                        break
                
                    bword = bword[:min_idx] + [bword[min_idx] + bword[min_idx+1]] + bword[min_idx+2:]
                
                final_tokens.extend(bword)
        
        ids = [self.ranks[token] for token in final_tokens]
        return ids
       
    
    def decode(self, ids: list[int]) -> list[str]:
    
        decoded_tokens = [self.decode_ranks[ide].decode("utf-8") for ide in ids]
        print(f"decoded tokens {decoded_tokens}")
        return "".join(decoded_tokens)
    
    
if __name__ == "__main__":
    filename_text = "data/scratch_tokenizers/hello.txt"
    b_bpe = ByteBPE(special_tokens=["<PAD>", "|endoftext|"])
    b_bpe.train(filename=filename_text, vocab_size=300)
    
    tokens_ids = b_bpe.encode("<PAD> asd hello worldf <PAD> |endoftext|")
    print(tokens_ids)
    tokens = b_bpe.decode(tokens_ids)
    print(tokens)
    
    
    filename_text = "data/scratch_tokenizers/multi_hello.txt"
    b_bpe = ByteBPE()
    b_bpe.train(filename=filename_text, vocab_size=300)
    
    tokens_ids = b_bpe.encode("مرحبا hello")
    print(tokens_ids)
    tokens = b_bpe.decode(tokens_ids)
    print(tokens)

