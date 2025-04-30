import os
import torch
import yaml
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from audiocraft.models import MusicGen

# --- LOAD CONFIG ---
with open('/workspace/finetune_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

model_name = "facebook/musicgen-small"
dataset_path = config['dataset_path']
epochs = config['epochs']
batch_size = config['batch_size']
learning_rate = float(config['learning_rate'])
output_dir = config['output_dir']
save_every_n_epochs = config['save_every_n_epochs']

warnings.filterwarnings("ignore", category=UserWarning, message=".*weight_norm is deprecated.*")

# --- DATASET ---
class DoomTokenDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = []
        for dirpath, _, filenames in os.walk(root):
            for f in filenames:
                if f.endswith(".pt"):
                    self.files.append(os.path.join(dirpath, f))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        tokens = torch.load(self.files[idx])
        if isinstance(tokens, tuple):
            tokens = tokens[0]
        tokens = tokens.flatten()
        return tokens

# --- COLLATE ---
def collate_fn(batch):
    return pad_sequence(batch, batch_first=True)

# --- INIT MODEL ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicGen.get_pretrained(model_name)
model.lm.to(device)
model.lm = model.lm.float()
optimizer = torch.optim.AdamW(model.lm.parameters(), lr=learning_rate)

# --- DATALOADER ---
ds = DoomTokenDataset(dataset_path)
dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# --- TRAIN LOOP ---
for epoch in range(epochs):
    model.lm.train()
    total_loss = 0.0

    for tokens in tqdm(dl, desc=f"Epoch {epoch+1}/{epochs}"):
        optimizer.zero_grad()
        tokens = tokens.to(device).long()

        num_codebooks = 4
        batch_size, seq_len = tokens.shape

        tokens = tokens.unsqueeze(1).expand(-1, num_codebooks, -1)  # [B, 4, T]

        dummy_cond = torch.zeros((batch_size, 1, 1024), device=device).float()
        dummy_mask = torch.ones((batch_size, 1), dtype=torch.bool, device=device)

        condition_tensors = {
            "description": (dummy_cond, dummy_mask)
        }

        outputs = model.lm(tokens, conditions=None, condition_tensors=condition_tensors)
        # outputs: [B, 4, T, 2048], tokens: [B, 4, T]

        # Flatten for cross entropy loss
        B, K, T, V = outputs.shape
        outputs = outputs.permute(0, 2, 1, 3).reshape(-1, V)  # [B*T*K, V]
        tokens = tokens.permute(0, 2, 1).reshape(-1)          # [B*T*K]

        loss = torch.nn.functional.cross_entropy(outputs, tokens)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dl)
    print(f"✅ Epoch {epoch+1}/{epochs} — avg loss: {avg_loss:.6f}")

    if (epoch + 1) % save_every_n_epochs == 0:
        os.makedirs(output_dir, exist_ok=True)
        ckpt_path = os.path.join(output_dir, f"model_epoch{epoch+1}.pt")
        torch.save(model.lm.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

print("🎉 Doom decoder training complete.")

#################
