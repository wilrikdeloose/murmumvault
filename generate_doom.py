import torch
from audiocraft.models import MusicGen
import torchaudio
import os
import warnings

# === SUPPRESS WARNINGS ===
warnings.filterwarnings("ignore", category=UserWarning, message=".*weight_norm is deprecated.*")

# === CONFIG ===
checkpoint_path = "/workspace/doom_finetuned/model_epoch3.pt"
output_dir = "/workspace/doom_samples"
os.makedirs(output_dir, exist_ok=True)

prompt = "Brutal mechanical groove at 117 BPM with synth grit and explosive drop"
duration = 12  # seconds
sample_rate = 32000

# === LOAD MODEL ===
model = MusicGen.get_pretrained("facebook/musicgen-small")
model.lm.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.lm.eval()
model.lm.to("cuda")

# === GENERATE AUDIO ===
model.set_generation_params(duration=duration, use_sampling=True, top_k=250)

with torch.no_grad():
    wavs = model.generate([prompt])  # [1, C, T] format

# === SAVE WITH TORCHAUDIO ===
output_path = os.path.join(output_dir, "doom_prompted.wav")
torchaudio.save(output_path, wavs[0].cpu(), sample_rate=sample_rate, encoding="PCM_S")

print(f"✅ Saved Doom WAV with prompt:\n→ {prompt}\n→ {output_path}")
