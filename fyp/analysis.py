import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_accuracies_phdim_against_epochs(input_file, output_file):
    df = pd.read_csv(input_file).sort_values('epoch')

    df_train = df.dropna(subset=['train_accuracy'])
    df_val = df.dropna(subset=['val_accuracy'])
    df_ph = df[df['phdim_0'] >= 0].dropna(subset=['phdim_0'])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_facecolor('#e6f3ff') 

    ax1.plot(df_train['epoch'], df_train['train_accuracy'], color='blue', linewidth=0.8, label='Train Acc')
    ax1.plot(df_val['epoch'], df_val['val_accuracy'], color='green', linewidth=0.8, label='Val Acc')

    ax1.set_xscale('log')          
    ax1.set_ylim(0, 100)        
    ax1.set_xlabel('Epoch (Log Scale)')
    ax1.set_ylabel('Accuracy (%)')

    ax2 = ax1.twinx()
    ax2.plot(df_ph['epoch'], df_ph['phdim_0'], color='red', linewidth=0.8, label='Persistent Homology Dimension')
    
    ax2.set_yscale('log')
    ax2.set_ylabel('Persistent Homology Dimension (Log Scale)')
    ax2.tick_params(axis='y')

    plt.title('addition operation')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='x-small')

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def concat_csvs(csv_list, output_name):
    df_list = [pd.read_csv(f) for f in csv_list]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.to_csv(output_name, index=False)

    for f in csv_list:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except OSError as e:
            print(f"Error deleting {f}: {e}")

    print(f"Combined {len(csv_list)} files into {output_name}")

if __name__ == "__main__":
    plot_accuracies_phdim_against_epochs("fyp/addition.csv", "fyp/addition.png")