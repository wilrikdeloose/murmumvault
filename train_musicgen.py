import os
import json
import torch
import yaml
import random
import torchaudio
from tqdm import tqdm
from audiocraft.models import MusicGen
from audiocraft.modules.losses import MelSpectrogramLoss
from torch.utils.data import DataLoader, Dataset

# --- Step 1: Load config ---
with open('/workspace/finetune_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_name = config['model_name']
dataset_path = config['dataset_path']
epochs = config['epochs']
batch_size = config['batch_size']
learning_rate = config.get('learning_rate', 5e-5)
output_dir = config['output_dir']
save_every_n_epochs = config.get('save_every_n_epochs', 1)

# --- Step 2: Load metadata ---
metadata_path = os.path.join(dataset_path, 'metadata.json')
with open(metadata_path, 'r') as file:
    metadata = json.load(file)

# Build a lookup dict
clip_to_prompt = {}
for entry in metadata:
    vibe = ", ".join(entry['vibe']) if entry['vibe'] else ""
    energy = entry.get('energy', "")
    songpart = entry.get('songpart', "")
    bpm = entry.get('bpm', "")

    prompt = f"A {energy} energy, {vibe} style, {songpart} section at {bpm} BPM."
    clip_to_prompt[entry['clip']] = prompt

# --- Step 3: Custom Dataset Loader ---
class DoomDataset(Dataset):
    def __init__(self, root_path, clip_to_prompt):
        self.root_path = root_path
        self.clip_to_prompt = clip_to_prompt
        self.files = list(clip_to_prompt.keys())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        clip_relpath = self.files[idx]
        audio_path = os.path.join(self.root_path, clip_relpath)

        audio, sr = torchaudio.load(audio_path)
        if sr != 32000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=32000)
            audio = resampler(audio)

        audio = audio.mean(dim=0).unsqueeze(0)  # Convert to mono if needed
        prompt = self.clip_to_prompt[clip_relpath]
        return audio, prompt

# --- Step 4: Prepare model and data ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MusicGen.get_pretrained(model_name)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_fn = MelSpectrogramLoss()

doom_dataset = DoomDataset(dataset_path, clip_to_prompt)
dataloader = DataLoader(doom_dataset, batch_size=batch_size, shuffle=True)

# --- Step 5: Training Loop ---
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    running_loss_
