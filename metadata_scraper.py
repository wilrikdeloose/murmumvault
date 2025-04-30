import os
import json

dataset_path = './dataset_segments'
metadata = []

for root, dirs, files in os.walk(dataset_path):
    for filename in files:
        if filename.lower().endswith('.wav'):
            # Build the relative file path (relative to dataset_path)
            full_path = os.path.relpath(os.path.join(root, filename), dataset_path)
            
            parts = filename.replace('.wav', '').split('_')
            
            # Expected format: doom_{bpm}bpm_{energy}_{songpart}_{id}
            try:
                bpm = int(parts[1].replace('bpm', ''))
                energy = parts[2]
                songpart = parts[3]
            except (IndexError, ValueError) as e:
                print(f"Warning: Skipping file with invalid format: {filename}")
                continue
            
            metadata.append({
                'clip': full_path.replace('\\', '/'),  # Normalize to forward slashes
                'vibe': ['mechanical'],
                'bpm': bpm,
                'energy': energy,
                'songpart': songpart,
                'instrumentation': []
            })

# Save metadata.json inside the dataset folder
metadata_output_path = os.path.join(dataset_path, 'metadata.json')
os.makedirs(dataset_path, exist_ok=True) # Create the directory if it doesn't exist
with open(metadata_output_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Metadata collected for {len(metadata)} clips and saved to {metadata_output_path}.")
