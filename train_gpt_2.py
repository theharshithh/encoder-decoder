import torch
import torch.nn as nn
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
import os
import requests
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import DecoderBlock, FeedForward, MultiHeadAttention, LayerNorm, PE, InputEmbedding, ProjectionLayer

def get_gpt2_config():
    return {
        'model_folder': 'gpt2_kan_weights',
        'tokenizer_file': 'tokenizer_{0}.json',
        'experiment_name': 'runs/gpt2_kannada',
        'lang': 'ka',
        'seq_len': 128,
        'batch_size': 64,
        'd_model': 768,
        'n_heads': 12,
        'n_layers': 12,
        'd_ff': 3072,
        'dropout': 0.1,
        'lr': 1e-4,
        'num_epochs': 20,
        'preload': None
    }

class GPT2Model(nn.Module):
    def __init__(self, vocab_size, seq_len, d_model=768, n_heads=12, n_layers=12, dropout=0.1, d_ff=3072):
        super().__init__()
        self.token_embedding = InputEmbedding(d_model, vocab_size)
        self.position_embedding = PE(d_model, seq_len, dropout)
        
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                self_attention_block=MultiHeadAttention(d_model, n_heads, dropout),
                cross_attention_block=None,
                feed_forward=FeedForward(d_model, d_ff, dropout),
                dropout=dropout
            ) for _ in range(n_layers)
        ])
        
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.projection = ProjectionLayer(d_model, vocab_size)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.alpha.data.fill_(1.0)
            
    def forward(self, x, mask=None):
        batch_size, seq_len = x.size()
        
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        token_embed = self.token_embedding(x)
        position_embed = self.position_embedding(positions)
        x = self.dropout(token_embed + position_embed)
        
        for block in self.decoder_blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        x = self.projection(x)
        
        return x

def get_kannada_text(kan_path='kannada.txt', dataset_dir="dataset"):
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    kan_path = os.path.join(dataset_dir, kan_path)
    
    if not os.path.exists(kan_path):
        kan_dataset = "https://huggingface.co/Avanthika/language-translation/resolve/main/kannada.txt"
        response = requests.get(kan_dataset)
        with open(kan_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

    with open(kan_path, 'r', encoding='utf-8') as f:
        kan_lines = f.read().strip().split('\n')
        
    return kan_lines

class KannadaLMDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        self.sos_token = tokenizer.token_to_id("[SOS]")
        self.eos_token = tokenizer.token_to_id("[EOS]")
        self.pad_token = tokenizer.token_to_id("[PAD]")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        tokens = self.tokenizer.encode(text).ids
        
        if len(tokens) > self.seq_len - 2:
            tokens = tokens[:self.seq_len - 2]
            
        input_tokens = [self.sos_token] + tokens + [self.eos_token]
        
        padding_length = self.seq_len - len(input_tokens)
        if padding_length > 0:
            input_tokens = input_tokens + [self.pad_token] * padding_length
            
        input_tensor = torch.tensor(input_tokens, dtype=torch.long)
        
        target_tensor = torch.tensor(tokens + [self.eos_token] + [self.pad_token] * padding_length, dtype=torch.long)
        
        mask = (input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0)
        
        seq_len = input_tensor.size(0)
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        
        combined_mask = mask & causal_mask
        
        return {
            "input_ids": input_tensor,
            "labels": target_tensor,
            "attention_mask": mask,
            "causal_mask": combined_mask,
            "text": text
        }

def get_or_build_tokenizer(config, texts):
    tokenizer_path = Path(config['tokenizer_file'].format(config['lang']))
    
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        
        tokenizer.train_from_iterator(texts, trainer=trainer)
        
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
    return tokenizer

def get_datasets(config):
    kannada_texts = get_kannada_text()
    
    tokenizer = get_or_build_tokenizer(config, kannada_texts)
    
    train_size = int(0.9 * len(kannada_texts))
    val_size = len(kannada_texts) - train_size
    
    train_texts, val_texts = random_split(kannada_texts, [train_size, val_size])
    
    train_dataset = KannadaLMDataset(train_texts, tokenizer, config['seq_len'])
    val_dataset = KannadaLMDataset(val_texts, tokenizer, config['seq_len'])
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False
    )
    
    return train_dataloader, val_dataloader, tokenizer

def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, device='cpu'):
    model.eval()
    
    input_ids = tokenizer.encode(prompt).ids
    input_ids = [tokenizer.token_to_id("[SOS]")] + input_ids
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            seq_len = input_tensor.size(1)
            causal_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len)).to(device)
            
            logits = model(input_tensor, causal_mask)
            
            next_token_logits = logits[:, -1, :] / temperature
            
            probabilities = torch.softmax(next_token_logits, dim=-1)
            
            next_token = torch.multinomial(probabilities, 1)
            
            input_tensor = torch.cat([input_tensor, next_token], dim=1)
            
            if next_token.item() == tokenizer.token_to_id("[EOS]"):
                break
    
    generated_ids = input_tensor[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    generated_text = generated_text.replace("[SOS]", "").replace("[EOS]", "").strip()
    
    return generated_text

@torch.no_grad()
def evaluate(model, val_dataloader, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss(ignore_index=val_dataloader.dataset.pad_token)
    
    for batch in val_dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        causal_mask = batch["causal_mask"].to(device)
        
        logits = model(input_ids, causal_mask)
        
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()
    
    return total_loss / len(val_dataloader)

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer = get_datasets(config)
    
    model = GPT2Model(
        vocab_size=tokenizer.get_vocab_size(),
        seq_len=config['seq_len'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        dropout=config['dropout'],
        d_ff=config['d_ff']
    ).to(device)
    
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))
    
    global_step = 0
    initial_epoch = 0
    
    if config['preload']:
        model_filename = os.path.join(config['model_folder'], f"{config['preload']}.pt")
        if os.path.exists(model_filename):
            print(f"Loading model from {model_filename}")
            state = torch.load(model_filename)
            model.load_state_dict(state['model_state_dict'])
            optimizer.load_state_dict(state['optimizer_state_dict'])
            initial_epoch = state['epoch'] + 1
            global_step = state['global_step']
    
    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        total_loss = 0

        batch_iterator = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch in batch_iterator:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            causal_mask = batch["causal_mask"].to(device)
            
            logits = model(input_ids, causal_mask)
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
            
            writer.add_scalar('train_loss', loss.item(), global_step)
            global_step += 1
        
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")
        
        val_loss = evaluate(model, val_dataloader, device)
        print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")
        
        writer.add_scalar('validation_loss', val_loss, epoch)
        
        sample_prompt = val_dataloader.dataset.texts[0][:20]
        generated_text = generate_text(model, tokenizer, sample_prompt, device=device)
        print(f"Sample generation: {generated_text}")
        
        model_filename = os.path.join(config['model_folder'], f"epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_train_loss,
            'global_step': global_step
        }, model_filename)
        
        print(f"Model saved to {model_filename}")
    
    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    config = get_gpt2_config()
    train(config) 