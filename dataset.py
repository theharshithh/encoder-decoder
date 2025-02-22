import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
import os

import requests

kan_dataset = "https://huggingface.co/Avanthika/language-translation/resolve/main/kannada.txt"
en_dataset = "https://huggingface.co/Avanthika/language-translation/resolve/main/english.txt"

def create_dataset(eng_path='eng.txt', kan_path='kan.txt', dataset_dir="dataset"):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    eng_path = os.path.join(dataset_dir, eng_path)
    kan_path = os.path.join(dataset_dir, kan_path)

    if not os.path.exists(eng_path):
        response = requests.get(en_dataset)
        with open(eng_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

    if not os.path.exists(kan_path):
        response = requests.get(kan_dataset)
        with open(kan_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

    with open(eng_path, 'r', encoding='utf-8') as f:
        eng_lines = f.read().strip().split('\n')

    with open(kan_path, 'r', encoding='utf-8') as f:
        kan_lines = f.read().strip().split('\n')

    data_pairs = []
    for e_line, k_line in zip(eng_lines, kan_lines):
        data_pairs.append(
            {"translation": {"en": e_line, "ka": k_line}}
        )
    return data_pairs

class BilingualDataset(Dataset):
    def __init__(self, ds, tokeniser_src, tokeniser_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()

        self.ds = ds
        self.tokeniser_src = tokeniser_src
        self.tokeniser_tgt = tokeniser_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokeniser_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokeniser_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokeniser_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        #seq_len = 10
        # pad_token = 0 -> tokeniser_src.token_to_id("[PAD]")
        #eos_token = 45 -> tokeniser_src.token_to_id("[EOS]")
        #sos_token = 15 -> tokeniser_src.token_to_id("[SOS]")
        src_tgt_pair = self.ds[index]
        # crude eg: src_tgt_pair = {"translation": {"en": "Hello world", "fr": "Bonjour le monde"}}
        src_text = src_tgt_pair["translation"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]

        enc_input_tokens = self.tokeniser_src.encode(src_text).ids
        # enc_input_tokens = [34, 56]
        dec_input_tokens = self.tokeniser_tgt.encode(tgt_text).ids
        
        # Account for SOS and EOS tokens in the length calculation
        enc_no_of_pad_tokens = self.seq_len - len(enc_input_tokens) - 2  # -2 for SOS and EOS
        # enc_no_of_pad_tokens = 10 - 2 - 2 = 6
        dec_no_of_pad_tokens = self.seq_len - len(dec_input_tokens) - 1  # -1 for SOS only
        # dec_no_of_pad_tokens = 10 - 1 - 1 = 8
        
        if enc_no_of_pad_tokens < 0 or dec_no_of_pad_tokens < 0:
            raise ValueError('Sentence is too long')
        
        encoder_input = torch.cat(
            [
                self.sos_token,  # [SOS], size(0) = 1
                torch.tensor(enc_input_tokens, dtype=torch.int64), # [34, 56], size(0) = 2
                self.eos_token,  # [EOS], size(0) = 1
                torch.tensor([self.pad_token] * enc_no_of_pad_tokens, dtype=torch.int64) # [0, 0, 0, 0, 0, 0], size(0) = 6
            ]
        )
        # encoder_input = [15, 34, 56, 45, 0, 0, 0, 0, 0, 0] -> encoder_input.size(0) = 10
        decoder_input = torch.cat(
            [
                self.sos_token,  # [SOS], size(0) = 1
                torch.tensor(dec_input_tokens, dtype=torch.int64), # [21, 43, 67], size(0) = 3
                torch.tensor([self.pad_token] * dec_no_of_pad_tokens, dtype=torch.int64) # [0, 0, 0, 0, 0, 0, 0, 0], size(0) = 8
            ]
        )   
        # decoder_input = [15, 21, 43, 67, 0, 0, 0, 0, 0, 0] -> decoder_input.size(0) = 10
        
        prediction = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64), # [21, 43, 67], size(0) = 3
                self.eos_token, # [45], size(0) = 1
                torch.tensor([self.pad_token] * dec_no_of_pad_tokens, dtype=torch.int64) # [0] * 7 = [0, 0, 0, 0, 0, 0, 0], size(0) = 7
            ]
        )
        # label = [21, 43, 67, 45, 0, 0, 0, 0, 0, 0] -> label.size(0) = 10
        
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert prediction.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).type(torch.int64).unsqueeze(0).unsqueeze(0), # [1, 1, 10] -> goes to attn, so need (b,t,c) dims
            "decoder_mask": (decoder_input != self.pad_token).type(torch.int64).unsqueeze(0).unsqueeze(0) & self.causal_mask(decoder_input.size(0)), # (1,1,seq_len) & (1,seq_len,seq_len) -> (1,1,seq_len),  [1, 1, 10] -> goes to attn, so need (b,t,c) dims
            "label": prediction, # (seq_len)
            "src_text": src_text, 
            "tgt_text": tgt_text
        }
    
    def causal_mask(self, size):
        mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int64)
        return mask == 0

    
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int64)
    return mask == 0