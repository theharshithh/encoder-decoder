import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import random_split
import os
from dataset import BilingualDataset, create_dataset
from torch.utils.data import DataLoader
from model import build_transformer
from config import get_config, get_weights_file_path
from torch.utils.tensorboard import SummaryWriter
from config import get_config
from tqdm import tqdm

def get_or_build_tokeniser(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer =Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentances(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer=Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_all_sentances(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_ds(config):
    ds_raw = create_dataset('eng.txt', 'kan.txt', 'dataset')

    if not ds_raw:
        raise FileNotFoundError("Dataset not found")

    tokeniser_src = get_or_build_tokeniser(config, ds_raw, config['lang_src'])
    tokeniser_tgt = get_or_build_tokeniser(config, ds_raw, config['lang_tgt'])

    train_ds_size = int(0.9*len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokeniser_src, tokeniser_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokeniser_src, tokeniser_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for idx in ds_raw:
        src_ids = tokeniser_src.encode(idx['translation'][config['lang_src']]).ids
        tgt_ids = tokeniser_tgt.encode(idx['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
    
    print(f"Max length for source sentance is: {max_len_src}")
    print(f"Max length for target sentance is: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
                

    return train_dataloader, val_dataloader, tokeniser_src, tokeniser_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'],config['seq_len'], config['d_model'])
    return model


def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokeniser_src, tokeniser_tgt = get_ds(config)
    model = get_model(config, tokeniser_src.get_vocab_size(), tokeniser_tgt.get_vocab_size()).to(device)
    
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoh = 0
    global_epoch = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"Loading model from {model_filename}")
        state = torch.load(model_filename)
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokeniser_src.token_to_id("[PAD]"), label_smoothing=0.1).to(device)
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")
        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (b, 1, 1, seq_len)
            decoder_input = batch['decoder_input'].to(device) # (b, seq_len)
            decoder_mask = batch['decoder_mask'].to(device) # (b, 1, seq_len, seq_len)

            encoder_op = model.encode(encoder_input, encoder_mask) # (b, seq_len, d_model)

            decoder_op = model.encode(decoder_input, decoder_mask) #(b, seq_len, d_model)

            projection_op = model.project(decoder_op) # (b, seq_len, tgt_vocab_szie)

            label = batch['label'].to(device) # (b, seq_len)

            #(b, seq_len, tgt_vocab_szie) --> (b*seq_len, tgt_vocab_size)
            loss = loss_fn(projection_op.view(-1, tokeniser_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        
        model_filename = get_weights_file_path(config, f'{epoch:02d}')
        torch.save({
            'epoch': epoch, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    config=get_config()
    train(config)

