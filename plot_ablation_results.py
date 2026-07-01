import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_file = "ablation_results.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run run_ablation_suite.py first.")
        return

    df = pd.read_csv(csv_file)
    modes = df['Mode'].unique()

    # Style definitions
    colors = {'A-1': '#888888', 'A-2': '#d62728', 'A-3': '#ff7f0e', 'A-4': '#1f77b4'}
    labels = {
        'A-1': 'A-1: RAG-only Baseline',
        'A-2': 'A-2: Online SFT (No Gate)',
        'A-3': 'A-3: InsomnAI v0.2 (No Hormones)',
        'A-4': 'A-4: InsomnAI 3.0 (Full)'
    }

    plt.style.use('seaborn-v0_8-darkgrid')

    # 1. Forgetting Curves (KL Drift)
    plt.figure(figsize=(10, 6))
    for mode in modes:
        mode_data = df[df['Mode'] == mode]
        plt.plot(mode_data['Cycle'], mode_data['KL_Drift'], 
                 label=labels.get(mode, mode), color=colors.get(mode), linewidth=2.5, marker='o')
    
    plt.title('Forgetting Curves: General Policy Drift', fontsize=14, fontweight='bold')
    plt.xlabel('Wake-Sleep Cycles', fontsize=12)
    plt.ylabel('KL Divergence (Drift)', fontsize=12)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('forgetting_curves.png', dpi=300)
    print("Saved forgetting_curves.png")

    # 2. Calibration Curves (ECE)
    plt.figure(figsize=(10, 6))
    for mode in modes:
        mode_data = df[df['Mode'] == mode]
        plt.plot(mode_data['Cycle'], mode_data['ECE'], 
                 label=labels.get(mode, mode), color=colors.get(mode), linewidth=2.5, marker='s')
    
    plt.title('Calibration Curves: Expected Calibration Error (ECE)', fontsize=14, fontweight='bold')
    plt.xlabel('Wake-Sleep Cycles', fontsize=12)
    plt.ylabel('ECE', fontsize=12)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('calibration_curves.png', dpi=300)
    print("Saved calibration_curves.png")

    # 3. Context Growth (Memory Size)
    plt.figure(figsize=(10, 6))
    for mode in modes:
        mode_data = df[df['Mode'] == mode]
        plt.plot(mode_data['Cycle'], mode_data['Memory_Size'], 
                 label=labels.get(mode, mode), color=colors.get(mode), linewidth=2.5, marker='^')
    
    plt.title('Graph Memory Context Growth', fontsize=14, fontweight='bold')
    plt.xlabel('Wake-Sleep Cycles', fontsize=12)
    plt.ylabel('Number of Declarative Triples', fontsize=12)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('context_growth.png', dpi=300)
    print("Saved context_growth.png")

if __name__ == "__main__":
    main()
