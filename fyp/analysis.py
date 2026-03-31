import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

ROOT = "fyp/data"

"""

DATASET, DATASET SIZE, EPOCHS TRAINED, STEPS COMPLETED:

ADDITION: 50, 21597, 215980
MULTIPLICATION: 50, 25096, 250970
SUBTRACTION: 75, 1822, 25522
SUBTRACTION-60: 60, 15280, 25522
DIVISION: 66.7, 13842, 179959
X^2+Y^2: 25, 85150, 425755


"""

def plot_accuracies_phdim_against_epochs(input_dir):
    """
        input_dir: Directory name of the operation (i.e. the child directory in fyp/data)
                   Examples: 'addition', 'multiplication' etc.

        This function plots training acc, validation acc and Persistent Homology Dimension
        against epochs. CSV file must follow format of that outputted by scripts/train.py
    """

    input_parent_dir = os.path.join(ROOT, input_dir)
    df = pd.read_csv(os.path.join(input_parent_dir, input_dir + ".csv")).sort_values('epoch')

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
    
    #ax2.set_yscale('log')
    ax2.set_ylim(bottom=0)
    ax2.set_ylabel('Persistent Homology Dimension ')
    ax2.tick_params(axis='y')

    plt.title(f"{input_dir} operation")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='x-small')

    plt.tight_layout()
    plt.savefig(os.path.join(input_parent_dir, input_dir + "_acc.png"))
    plt.show()

def plot_losses_phdim_against_epochs(input_dir): 
    input_parent_dir = os.path.join(ROOT, input_dir)
    df = pd.read_csv(os.path.join(input_parent_dir, input_dir + ".csv")).sort_values('epoch')

    df_train = df.dropna(subset=['train_loss'])
    df_val = df.dropna(subset=['val_loss'])
    df_ph = df[df['phdim_0'] >= 0].dropna(subset=['phdim_0'])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_facecolor('#e6f3ff') 

    ax1.plot(df_train['epoch'], df_train['train_loss'], color='blue', linewidth=0.8, label='Train Loss')
    ax1.plot(df_val['epoch'], df_val['val_loss'], color='green', linewidth=0.8, label='Val Loss')

    ax1.set_xscale('log')       
    ax1.set_xlabel('Epoch (Log Scale)')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(df_ph['epoch'], df_ph['phdim_0'], color='red', linewidth=0.8, label='Persistent Homology Dimension')
    
    #ax2.set_yscale('log')
    #ax2.set_ylim(bottom=0)
    ax2.set_ylabel('Persistent Homology Dimension')
    ax2.tick_params(axis='y')

    plt.title(f"{input_dir} operation")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='x-small')

    plt.tight_layout()
    plt.savefig(os.path.join(input_parent_dir, input_dir + "_loss.png"))
    plt.show()

def plot_graphs(operation):
    plot_losses_phdim_against_epochs(operation)
    plot_accuracies_phdim_against_epochs(operation)

def concat_csvs(csv_list, output_dir):
    
    df_list = [pd.read_csv(f) for f in csv_list]
    combined_df = pd.concat(df_list, ignore_index=True)

    root = "fyp/data"
    full_output_dir = os.path.join(root, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    csv_path = os.path.join(full_output_dir, output_dir + ".csv")
    combined_df.to_csv(csv_path, index=False)

    for f in csv_list:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except OSError as e:
            print(f"Error deleting {f}: {e}")

    print(f"Combined {len(csv_list)} files into {csv_path}")

def process_csvs(csv_list, output_dir):
    concat_csvs(csv_list, output_dir)
    plot_graphs(output_dir)

if __name__ == "__main__":
    process_csvs(["../../../Downloads/metrics (1).csv", "fyp/data/x^2+y^2_mod_97.csv"], "x^2+y^2_mod_97")