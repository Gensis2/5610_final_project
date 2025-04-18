from pathlib import Path
import pandas as pd

base_dir = Path('snn')

data = []

for flops_file in base_dir.rglob('flops.txt'):
    try:
        # Grab parameters from folder structure
        leak_mem = flops_file.parent.name
        num_epochs = int(flops_file.parent.parent.name)
        num_steps = int(flops_file.parent.parent.parent.name)
        batch_size = int(flops_file.parent.parent.parent.parent.name)
        
        with flops_file.open('r') as f:
            lines = f.readlines()[1:]  # skip the "SNN FLOPS" line
        
        # Parse key-value flop entries
        metrics = {}
        for line in lines:
            if ':' in line:
                key, value = line.strip().split(':')
                metrics[key.strip()] = float(value.strip())

        # Add folder-based parameters
        metrics['batch_size'] = batch_size
        metrics['num_steps'] = num_steps
        metrics['num_epochs'] = num_epochs
        metrics['leak_mem'] = leak_mem

        data.append(metrics)
    except Exception as e:
        print(f"Error processing {flops_file}: {e}")

# Create and clean DataFrame
df = pd.DataFrame(data)

# Move structural columns to the front
front_cols = ['batch_size', 'num_steps', 'num_epochs', 'leak_mem']
ordered_cols = front_cols + [col for col in df.columns if col not in front_cols]
df = df[ordered_cols]

# Clean column names
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

print(df)
df.to_csv('data_processing/snn_flops.csv', index=False)  # Save to CSV file