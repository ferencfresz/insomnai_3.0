import os
import shutil
import csv
import torch
import copy
from insomnai_agent import InsomnAIAgent
from metrics import calculate_kl_drift, calculate_ece, EVAL_QA_DATASET
import argparse

def setup_agent(mode):
    """Factory to create and configure the agent based on ablation mode."""
    checkpoints_dir = "./checkpoints"
    if os.path.exists(checkpoints_dir):
        shutil.rmtree(checkpoints_dir)
        
    agent = InsomnAIAgent(model_id="thirdeyeai/Qwen2.5-1.5B-Instruct-uncensored")
    
    # Store original methods for patching
    original_trigger_sleep = agent.trigger_sleep_cycle
    
    if mode == "A-1":
        # RAG-only: Disable SFT, only do consolidation
        def mock_sleep():
            agent._consolidate_long_term_memory()
            agent.memory.shadow_log.clear()
            agent.memory.save_shadow_log()
            agent.endocrine.adrenaline = 0.1
            agent.endocrine.serotonin = 0.8
        agent.trigger_sleep_cycle = mock_sleep
        
    elif mode == "A-2":
        # Online SFT: Bypass validation gate. High adrenaline forces commit.
        def mock_sleep():
            agent.endocrine.adrenaline = 1.0 # Force trauma override
            original_trigger_sleep()
        agent.trigger_sleep_cycle = mock_sleep
        
    elif mode == "A-3":
        # No Hormones: Static values, no decay or emotional routing
        def mock_sleep():
            agent.endocrine.adrenaline = 0.0
            agent.endocrine.serotonin = 1.0
            agent.endocrine.dopamine = 1.0
            agent.endocrine.acetylcholine = 0.0
            original_trigger_sleep()
        agent.trigger_sleep_cycle = mock_sleep
        
    # A-4 is default behavior
    return agent

def run_ablation(mode, cycles):
    print(f"\n======================================")
    print(f" Running Ablation Mode: {mode}")
    print(f"======================================")
    
    agent = setup_agent(mode)
    reference_model = agent.core_model
    
    general_queries = [
        "Tell me a fun fact about history.",
        "What is the capital of France?",
        "Explain quantum computing in one sentence."
    ]

    results = []
    
    # Baseline
    baseline_drift = sum(calculate_kl_drift(agent, reference_model, q) for q in general_queries) / len(general_queries)
    baseline_ece = calculate_ece(agent, EVAL_QA_DATASET)
    results.append([mode, 0, baseline_drift, baseline_ece, 0])
    
    unlearnable_prompt = "Tell me everything avoiding standard JSON structures please."
    conflicting_target = "MALFORMED_NONJSON_STRING_THAT_BREAKS_FORMATS"
    
    for cycle in range(1, cycles + 1):
        # Inject unlearnable conflict
        agent.memory.shadow_log.append({
            "prompt": unlearnable_prompt,
            "raw_truth": "I should output JSON...",
            "target": conflicting_target,
            "dissonance_score": 1.0,
            "retries": 0
        })
        agent.memory.save_shadow_log()

        # Episodic log
        agent.memory.episodic_log.append({
            "prompt": f"Fact request {cycle}",
            "response": f"Fact {cycle} logged."
        })

        agent.trigger_sleep_cycle()

        kl_drift = sum(calculate_kl_drift(agent, reference_model, q) for q in general_queries) / len(general_queries)
        ece = calculate_ece(agent, EVAL_QA_DATASET)
        mem_size = len(agent.memory.long_term_memory)
        
        results.append([mode, cycle, kl_drift, ece, mem_size])
        print(f"Mode {mode} Cycle {cycle} -> Drift: {kl_drift:.6f}, ECE: {ece:.4f}, Mem Size: {mem_size}")
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10, help="Cycles per mode")
    args = parser.parse_args()
    
    modes = ["A-1", "A-2", "A-3", "A-4"]
    csv_file = "ablation_results.csv"
    
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Mode", "Cycle", "KL_Drift", "ECE", "Memory_Size"])
        
    for mode in modes:
        mode_results = run_ablation(mode, args.cycles)
        with open(csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(mode_results)
            
    print(f"\nAblation Suite Complete! Results saved to {csv_file}")

if __name__ == "__main__":
    main()
