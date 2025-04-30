import os
import torch
import torchaudio
from tqdm import tqdm
from audiocraft.models.encodec import EncodecModel

# --- SETTINGS ---
input_folder = '/workspace/doom_dataset'
output_folder = '/workspace/doom_tokens'
sample_rate = 32000

os.makedirs(output_folder, exist_ok=True)

# --- LOAD MODEL ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EncodecModel.get_pretrained(name="facebook/encodec_32khz").to(device)
model.eval()

# --- WALK THROUGH WAV FILES ---
for root, _, files in os.walk(input_folder):
    for file in tqdm(files):
        if not file.endswith('.wav'):
            continue

        wav_path = os.path.join(root, file)
        rel_path = os.path.relpath(wav_path, input_folder)
        save_path = os.path.join(output_folder, rel_path.replace('.wav', '.pt'))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            audio, sr = torchaudio.load(wav_path)
        except Exception as e:
            print(f"⚠️ Skipping file '{wav_path}': {e}")
            continue

        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            audio = resampler(audio)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        audio = audio.to(device)

        with torch.no_grad():
            tokens = model.encode(audio.unsqueeze(0))

        torch.save(tokens, save_path)

print("✅ All supported WAVs encoded to token files.")
