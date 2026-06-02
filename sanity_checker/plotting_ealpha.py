import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt

folder_path = 'sanity_checker/e_alpha_data'
dataframes = []

# 1. Load data and extract parameters
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        # Extract learning_rate, weight_decay, and seed
        lr, wd, seed = filename.replace('.csv', '').split('_')
        
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        
        # The column names have a leading space in your CSV, strip it if necessary
        df.columns = df.columns.str.strip() 
        
        df['lr'] = float(lr)
        df['wd'] = float(wd)
        df['seed'] = int(seed)
        
        dataframes.append(df)

full_df = pd.concat(dataframes, ignore_index=True)

# 2. Average Test_Acc and E_2.0 over seeds 
avg_df = full_df.groupby(['lr', 'wd', 'Epoch'])[['Test_Acc', 'E_2.0']].mean().reset_index()

# 3. Create and save a plot for each LR and WD combination
configs = avg_df[['lr', 'wd']].drop_duplicates()

for _, row in configs.iterrows():
    lr_val, wd_val = row['lr'], row['wd']
    plot_data = avg_df[(avg_df['lr'] == lr_val) & (avg_df['wd'] == wd_val)].sort_values('Epoch')
    
    epochs = plot_data['Epoch'].values
    
    # Apply median smoothing with kernel size 5
    test_acc_smooth = medfilt(plot_data['Test_Acc'].values, kernel_size=5)
    e2_smooth = medfilt(plot_data['E_2.0'].values, kernel_size=5)
    
    # Setup Figure
    fig, ax1 = plt.subplots(figsize=(10, 6), facecolor='white')
    ax1.set_facecolor('white')
    
    # Plot Test_Acc on primary Y-axis (Green)
    ax1.plot(epochs, test_acc_smooth, color='green', linewidth=3.5, label='Val Accuracy')
    ax1.set_xlabel('Epoch', fontsize=20)
    ax1.set_ylabel('Val Accuracy', fontsize=20, color='green')
    ax1.tick_params(axis='y', labelcolor='green', labelsize=15)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Plot E_2.0 on secondary Y-axis (Yellow/Goldenrod for visibility)
    ax2 = ax1.twinx()
    ax2.plot(epochs, e2_smooth, color='goldenrod', linewidth=3.5, label='E-2')
    ax2.set_ylabel('E_2', fontsize=20, color='goldenrod')
    ax2.tick_params(axis='y', labelcolor='goldenrod', labelsize=15)
    
    # Formatting
    plt.title(f'LR: {lr_val}, WD: {wd_val}', fontsize=20)
    
    plt.tight_layout()
    
    # Save the graph
    save_path = os.path.join(folder_path, f'e_alpha_plot_lr{lr_val}_wd{wd_val}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()