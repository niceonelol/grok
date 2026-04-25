import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import kendalltau

ROOT = "fyp/data"
PHD_TUP = ('phdim_0', 'Persistent Homology Dimension', 'red')
ALPHA_TUP = ('e_alpha', 'Alpha Weighted Lifetime Sum', 'orange')

"""

DATASET, DATASET SIZE, EPOCHS TRAINED, STEPS COMPLETED:

ADDITION: 50, 21597, 215980
MULTIPLICATION: 50, 25096, 250970
SUBTRACTION: 75, 1822, 25522
SUBTRACTION-60: 60, 15280, 25522
DIVISION: 66.7, 13842, 179959
X^2+Y^2: 25, 85150, 425755
MIX1: 95, 27209, 489780
QUAD2: 60, 23913, 286968

ALPHA_ADDITION: 50
ALPHA_SUBTRACTION: 60
ALPHA_MULTIPLICATION: 50
ALPHA_DIVISION: 66.7
ALPHA_X^2+Y^2: 30
ALPHA_MIX1: 95
ALPHA_QUAD2: 65

In mnist_2500_5.0_L2_eps=*, used a 'hugging value' of 4
In cifar_5000_5.0_L1_eps=0.5_hug=2, used 'hugging value' of 2

"""

def plot_accuracies_phdim_against_epochs(input_dir, metric_tup=PHD_TUP, smoothing=1):
    """
        input_dir: Directory name of the operation (i.e. the child directory in fyp/data)
                   Examples: 'addition', 'multiplication' etc.

        This function plots training acc, validation acc and Persistent Homology Dimension
        against epochs. CSV file must follow format of that outputted by scripts/train.py
    """
    metric, label, color = metric_tup
    input_parent_dir = os.path.join(ROOT, input_dir)
    df = pd.read_csv(os.path.join(input_parent_dir, input_dir + ".csv")).sort_values('epoch')

    df_train = df.dropna(subset=['train_accuracy']).copy()
    df_train['train_accuracy'] = df_train['train_accuracy'].rolling(smoothing, center=True, min_periods=1).median()
    df_val = df.dropna(subset=['val_accuracy']).copy()
    df_val['val_accuracy'] = df_val['val_accuracy'].rolling(smoothing, center=True, min_periods=1).median()
    df_ph = df.dropna(subset=[metric]).copy()
    if metric == 'phdim_0':
        lqr, uqr = df_ph[metric].quantile(0.25), df_ph[metric].quantile(0.75)
        iqr = uqr - lqr
        l_bound, u_bound = lqr - 1.5 * iqr, uqr + 1.5 * iqr
        df_ph = df_ph[(df_ph[metric] >= l_bound) & (df_ph[metric] <= u_bound)]
    df_ph[metric] = df_ph[metric].rolling(smoothing, center=True, min_periods=1).median()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_facecolor('#e6f3ff') 

    ax1.plot(df_train['epoch'], df_train['train_accuracy'], color='blue', linewidth=0.8, label='Train Acc')
    ax1.plot(df_val['epoch'], df_val['val_accuracy'], color='green', linewidth=0.8, label='Val Acc')

    ax1.set_xscale('log')          
    ax1.set_ylim(0, 100)        
    ax1.set_xlabel('Epoch (Log Scale)')
    ax1.set_ylabel('Accuracy (%)')

    ax2 = ax1.twinx()
    ax2.plot(df_ph['epoch'], df_ph[metric], color=color, linewidth=0.8, label=label)
    
    #ax2.set_yscale('log')
    #ax2.set_ylim(bottom=0)
    ax2.set_ylabel(label)
    ax2.tick_params(axis='y')

    plt.title(f"{input_dir} operation")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='x-small')

    plt.tight_layout()
    plt.savefig(os.path.join(input_parent_dir, input_dir + f"_{metric}_acc.png"))
    plt.show()

def plot_losses_phdim_against_epochs(input_dir, metric_tup=PHD_TUP, smoothing=1): 

    metric, label, color = metric_tup
    input_parent_dir = os.path.join(ROOT, input_dir)
    df = pd.read_csv(os.path.join(input_parent_dir, input_dir + ".csv")).sort_values('epoch')

    df_train = df.dropna(subset=['train_loss']).copy()
    df_train['train_loss'] = df_train['train_loss'].rolling(smoothing, center=True, min_periods=1).median()
    df_val = df.dropna(subset=['val_loss']).copy()
    df_val['val_loss'] = df_val['val_loss'].rolling(smoothing, center=True, min_periods=1).median()
    df_ph = df.dropna(subset=[metric]).copy()
    if metric == 'phdim_0':
        lqr, uqr = df_ph[metric].quantile(0.25), df_ph[metric].quantile(0.75)
        iqr = uqr - lqr
        l_bound, u_bound = lqr - 1.5 * iqr, uqr + 1.5 * iqr
        df_ph = df_ph[(df_ph[metric] >= l_bound) & (df_ph[metric] <= u_bound)]
    df_ph[metric] = df_ph[metric].rolling(smoothing, center=True, min_periods=1).median()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_facecolor('#e6f3ff') 

    ax1.plot(df_train['epoch'], df_train['train_loss'], color='blue', linewidth=0.8, label='Train Loss')
    ax1.plot(df_val['epoch'], df_val['val_loss'], color='green', linewidth=0.8, label='Val Loss')

    ax1.set_xscale('log')       
    ax1.set_xlabel('Epoch (Log Scale)')
    ax1.set_ylabel('Loss')

    ax2 = ax1.twinx()
    ax2.plot(df_ph['epoch'], df_ph[metric], color=color, linewidth=0.8, label=label)
    
    #ax2.set_yscale('log')
    #ax2.set_ylim(bottom=0)
    ax2.set_ylabel(label)
    ax2.tick_params(axis='y')

    plt.title(f"{input_dir} operation")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize='x-small')

    plt.tight_layout()
    plt.savefig(os.path.join(input_parent_dir, input_dir + f"_{metric}_loss.png"))
    plt.show()

def plot_graphs(operation, metric_tup=PHD_TUP, smoothing=1):
    plot_losses_phdim_against_epochs(operation, metric_tup=metric_tup, smoothing=smoothing)
    plot_accuracies_phdim_against_epochs(operation, metric_tup=metric_tup, smoothing=smoothing)

def concat_csvs(csv_list, output_dir):
    
    df_list = [pd.read_csv(f) for f in csv_list]
    combined_df = pd.concat(df_list, ignore_index=True)

    root = "fyp/data"
    full_output_dir = os.path.join(root, output_dir)
    os.makedirs(full_output_dir, exist_ok=True)
    csv_path = os.path.join(full_output_dir, output_dir + ".csv")
    combined_df.to_csv(csv_path, index=False)

    """
    for f in csv_list:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except OSError as e:
            print(f"Error deleting {f}: {e}")
    """

    print(f"Combined {len(csv_list)} files into {csv_path}")

def process_csvs(csv_list, output_dir, metric_tup=PHD_TUP, smoothing=1):
    concat_csvs(csv_list, output_dir)
    plot_graphs(output_dir, metric_tup=metric_tup, smoothing=smoothing)

def kendall_coeffs(op, metric, acc=True, smoothing=5):
    if acc:
        v = 'val_accuracy'
    else:
        v = 'val_loss'
    path = os.path.join(ROOT, op, f"{op}.csv")
    df = pd.read_csv(path).sort_values('epoch')
    df = df.dropna(subset=[metric, v]).copy()
    df = df[df[metric] >= 0]

    """
    df[v] = df[v].rolling(smoothing, center=True, min_periods=1).median()
    if metric == 'phdim_0':

        lqr, uqr = df[metric].quantile(0.25), df[metric].quantile(0.75)
        iqr = uqr - lqr
        l_bound, u_bound = lqr - 1.5 * iqr, uqr + 1.5 * iqr
        df = df[(df[metric] >= l_bound) & (df[metric] <= u_bound)]
    
    df[metric] = df[metric].rolling(smoothing, center=True, min_periods=1).median()
    """

    df = df[[metric, v, 'epoch']].copy()

    df = df.sort_values('epoch')
    N = df.shape[0]
    print(N,"\n")
    k = 1 + int(np.log2(N))
    
    test_bin_sizes = [N//10] #range(max(2, k//2), max(2, 4 * k + 1))
    print(f"Bin range: [{test_bin_sizes[0]},{test_bin_sizes[-1]}], Number of tests: {len(test_bin_sizes)}")
    kendall_taus = []
    for i in test_bin_sizes:
        bins = np.array_split(df, i)
        metric_means, v_means = [], []
        for bin_ in bins:
            metric_means.append(bin_[:, 0].mean())
            v_means.append(bin_[:, 1].mean())
        print(metric_means)
        print(v_means)
        res = kendalltau(metric_means, v_means)
        #if res.pvalue < 0.05:
        kendall_taus.append((i, res.statistic, res.pvalue))
    
    print(np.mean([x for (_, x, _) in kendall_taus]), "\n", kendall_taus)
    return kendall_taus


if __name__ == "__main__":
    
    files = [f"../../../Downloads/mnist_grok_1000_2.0.csv"]
    process_csvs(files, "alpha_mnist_1000_2.0", metric_tup=PHD_TUP, smoothing=5)
    process_csvs(files, "alpha_mnist_1000_2.0", metric_tup=ALPHA_TUP, smoothing=5)
    
    #kendall_coeffs('x^2+y^2_mod_97', 'phdim_0', smoothing=9)