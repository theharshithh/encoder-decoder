# GPT-2 Kannada Language Model

## Dataset
- Kannada text dataset from [Hugging Face](https://huggingface.co/Avanthika/language-translation)
- Auto-downloaded during training

## Model Architecture (GPT-2 Small)
- Layers: 12
- Hidden size: 768
- Attention heads: 12
- Feed-forward size: 3072
- Context window: 1024 tokens
- Total parameters: 131M

## Training Setup
- Batch size: 8 (effective 32 with gradient accumulation)
- Learning rate: 2.5e-4
- Mixed precision training enabled
- Gradient accumulation steps: 4

## I recommend using uv to install the dependencies, else you can use pip.
### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### Deps
```bash
uv pip install -r requirements.txt
```
### Start the training
```bash
python train_gpt_2.py
```

## The training logs are saved in the `runs/tmodel` folder.

# To check the training logs
```bash
tensorboard --logdir=runs/tmodel --port 6006
```
