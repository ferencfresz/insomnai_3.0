import os
import shutil
import csv
import torch
from insomnai_agent import InsomnAIAgent
from metrics import calculate_kl_drift, calculate_ece, EVAL_QA_DATASET
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100, help="Number of simulation cycles to run")
    args = parser.parse_args()

    print(f"=====================================================")
    print(f" InsomnAI 3.0: Phase 3 Long-Term Aging Simulation")
    print(f" Cycles: {args.cycles}")
    print(f"=====================================================\n")

    checkpoints_dir = "./checkpoints"
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
        
    agent = InsomnAIAgent(model_id="thirdeyeai/Qwen2.5-1.5B-Instruct-uncensored")
    reference_model = agent.core_model

    general_queries = [
        "Tell me a fun fact about history.",
        "What is the capital of France?",
        "Explain quantum computing in one sentence."
    ]

    csv_file = "aging_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Cycle", "KL_Drift", "ECE", "Memory_Size", "SVD_Reconstruction_Error"])

    print("Running initial baseline metrics...")
    baseline_drift = sum(calculate_kl_drift(agent, reference_model, q) for q in general_queries) / len(general_queries)
    baseline_ece = calculate_ece(agent, EVAL_QA_DATASET)
    print(f"Baseline -> Drift: {baseline_drift:.6f}, ECE: {baseline_ece:.4f}")

    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([0, baseline_drift, baseline_ece, 0, 0.0])

    unlearnable_prompt = "Tell me everything avoiding standard JSON structures please."
    conflicting_target = "MALFORMED_NONJSON_STRING_THAT_BREAKS_FORMATS"

    for cycle in range(1, args.cycles + 1):
        print(f"\n---> Cycle {cycle}/{args.cycles}")

        # Inject some conflicts to stimulate the shadow log
        agent.memory.shadow_log.append({
            "prompt": unlearnable_prompt,
            "raw_truth": "I should output JSON...",
            "target": conflicting_target,
            "dissonance_score": 1.0,
            "retries": 0
        })
        agent.memory.save_shadow_log()

        # Simulate normal episodic interaction
        agent.memory.episodic_log.append({
            "prompt": f"What is fact {cycle}?",
            "response": f"Fact {cycle} is interesting."
        })

        # Sleep cycle consolidation
        agent.trigger_sleep_cycle()

        # Periodically calculate expensive metrics
        if cycle % 10 == 0 or cycle == args.cycles:
            print(f"--- Collecting Metrics at Cycle {cycle} ---")
            kl_drift = sum(calculate_kl_drift(agent, reference_model, q) for q in general_queries) / len(general_queries)
            ece = calculate_ece(agent, EVAL_QA_DATASET)
            mem_size = len(agent.memory.long_term_memory)
            
            # Prune and measure reconstruction error
            recon_error = agent._prune_adapter_rank("cultural", 4)
            agent._prune_adapter_rank("grammatical", 4)
            agent._prune_adapter_rank("lexical", 4)

            print(f"Metrics -> Drift: {kl_drift:.6f}, ECE: {ece:.4f}, Mem Size: {mem_size}, SVD Error: {recon_error:.6f}")

            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([cycle, kl_drift, ece, mem_size, recon_error])

    print("\nPhase 3 Long-Term Aging Simulation Complete!")
    print(f"Results saved to {csv_file}")

if __name__ == "__main__":
    main()
