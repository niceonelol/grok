import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

folder_path = 'sanity_checker/phdim_data'
dataframes = []

# 1. Load data and calculate Generalisation Gap
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        batch_size, seed = filename.replace('.csv', '').split('_')
        
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        df['batch_size'] = int(batch_size)
        df['seed'] = int(seed)
        df['gen_gap'] = df['train_accuracy'] - df['val_accuracy']
        
        dataframes.append(df)

full_df = pd.concat(dataframes, ignore_index=True)

# 2. Average gen_gap and phdim_0 over seeds
avg_df = full_df.groupby(['batch_size', 'epoch'])[['gen_gap', 'phdim_0']].mean().reset_index()

avg_df = avg_df[avg_df['gen_gap'] >= 0]

# 3. Create and save a plot for each batch size
batch_sizes = avg_df['batch_size'].unique()

for bs in batch_sizes:
    bs_data = avg_df[avg_df['batch_size'] == bs]
    
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    ax.set_facecolor('white')

    x = bs_data['gen_gap']
    y = bs_data['phdim_0']

    correlation = np.corrcoef(x, y)[0, 1]
    print(f"Batch Size: {bs}, Correlation between Gen Gap and PH Dimension: {correlation:.4f}")

    # Scatter plot
    ax.scatter(x, y, color='blue', alpha=0.6, s=15)

    # Line of best fit
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m*x + b, color='lightblue', linewidth=2.5)

    # Formatting
    ax.set_title(f'Batch Size: {bs}', fontsize=20)
    ax.set_xlabel('Generalisation Gap', fontsize=20)
    ax.set_ylabel('PH Dimension', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Save the graph to the folder
    save_path = os.path.join(folder_path, f'phdim_vs_gengap_bs{bs}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
