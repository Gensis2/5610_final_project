from pathlib import Path
import pandas as pd

base_dir = Path('cnn')

data = []

for flops_file in base_dir.rglob('flops.txt'):
    try:
        batch_size = int(flops_file.parents[1].name)
        num_epochs = int(flops_file.parent.name)
        
        with flops_file.open('r') as f:
            lines = f.readlines()[1:]  # skip the first line ("SNN FLOPS")
        
        # Parse key-value pairs into a dictionary
        metrics = {}
        for line in lines:
            if ':' in line:
                key, value = line.strip().split(':')
                metrics[key.strip()] = float(value.strip())
        
        # Add batch_size and num_epochs
        metrics['batch_size'] = batch_size
        metrics['num_epochs'] = num_epochs

        data.append(metrics)
    except Exception as e:
        print(f"Error processing {flops_file}: {e}")

# Create DataFrame
df = pd.DataFrame(data)

# Optional: reorder columns
ordered_cols = ['batch_size', 'num_epochs'] + [col for col in df.columns if col not in ['batch_size', 'num_epochs']]
df = df[ordered_cols]

# Clean column names: lowercase and replace spaces with underscores
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

df.to_csv('data_processing/cnn_flops.csv', index=False)  # Save to CSV file
