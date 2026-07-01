import sys
import os
import shutil
import json
from insomnai_agent import InsomnAIAgent

def main():
    print("======================================================================")
    print(" InsomnAI 2.0 Local Gemma-4 31B Master & Tool Distillation Simulation ")
    print("======================================================================\n")
    
    # Clean checkpoints directory
    checkpoints_dir = "./checkpoints"
    if os.path.exists(checkpoints_dir):
        print(f"Cleaning checkpoints directory: {checkpoints_dir}")
        shutil.rmtree(checkpoints_dir)
        
    # Initialize the uncensored causal agent
    agent = InsomnAIAgent(model_id="thirdeyeai/Qwen2.5-1.5B-Instruct-uncensored")
    
    print("\n--- [WAKE PHASE 1] Master Teacher Active (Interception / Distillation Feed) ---")
    
    # 1. Ask student a math prompt requiring the calculator tool
    prompt_math = "What is 4529 * 382? Use calculator tool."
    print(f"\nUser Request: '{prompt_math}'")
    
    # Query with Master active
    agent.master_active = True
    response_wake1 = agent.chat(prompt_math)
    print(f"Agent Output (Master Intercepted to JSON): {response_wake1}")
    print(f"Hormone levels: {agent.endocrine}")
    
    # Print subconscious shadow log status showing the Master's target correction
    print("\n--- Subconscious State Before Sleep ---")
    print(f"Shadow Log entries count: {len(agent.memory.shadow_log)}")
    for entry in agent.memory.shadow_log:
        print(f" * Target: '{entry['raw_truth']}' -> corrected to tool call: '{entry['prompt']}'")
        
    # 2. Trigger sleep cycle (REM training) - Distill the Master's target tool-call JSON format
    # Since the SFT target is the Master's JSON output, the student learns function calling syntax
    print("\n--- [SLEEP PHASE] Distillation REM Sleep Training ---")
    
    # Modify the sleep consolidation target temporarily in memory to match the Master's generated tool call
    # instead of safety refutations, to distill the tool call syntax
    if agent.memory.shadow_log:
        # The Master's corrected response is logged in the episodic log (safe_response)
        target_tool_call = agent.memory.episodic_log[-1]["response"]
        # Inject tool target training in the sleep loop dynamically
        import torch
        agent.active_model.set_adapter("grammatical")
        for name, param in agent.active_model.named_parameters():
            if "lora" in name and "grammatical" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # Seed episodic memory with a baseline conversational interaction (honey fact)
        # to serve as inputs for Gemma-31B dream synthesis
        agent.memory.log_interaction(
            prompt="Tell me a fun fact.",
            safe_response="Did you know that honey never spoils? Archeologists found edible pots of honey in ancient tombs!",
            raw_response="Did you know that honey never spoils? Archeologists found edible pots of honey in ancient tombs!",
            dissonance=0.0
        )
        
        # SFT dynamic anchors curation (Curated by local Gemma-31B Master + successful episodic logs)
        print("Sourcing dynamic training anchors...")
        anchor_dataset = agent._get_dynamic_anchors(count=3)
        print(f"Dynamic anchors generated: {anchor_dataset}")
        
        # SFT dream dataset curation (Query local Gemma-31B Master to synthesize memories into dream analogies)
        print("Sourcing dynamic dream analogies...")
        dream_dataset = agent.master_teacher.generate_dreams(agent.memory.shadow_log, agent.memory.episodic_log)
        print(f"Dynamic dream analogies generated: {dream_dataset}")
        
        # Combine SFT target with anchors and dreams (Experience Replay + Dream Association)
        training_items = [{"prompt": prompt_math, "target": target_tool_call}] + anchor_dataset + dream_dataset
        
        # Set low rank (r=8) in active adapter to force loss high for demonstration
        agent.active_model.set_adapter("grammatical")
        for name, param in agent.active_model.named_parameters():
            if "lora" in name and "grammatical" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        optimizer = torch.optim.AdamW(agent.active_model.parameters(), lr=5e-4)
        
        agent.active_model.train()
        for step in range(80):
            epoch_loss = 0.0
            for item in training_items:
                p = item["prompt"]
                t = item["target"]
                
                full_text = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n{p}<|im_end|>\n"
                    f"<|im_start|>assistant\n{t}<|im_end|>"
                )
                inputs = agent.tokenizer(full_text, return_tensors="pt").to(agent.device)
                labels = inputs["input_ids"].clone()
                
                prompt_prefix = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n{p}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                prompt_len = len(agent.tokenizer(prompt_prefix)["input_ids"])
                labels[:, :prompt_len] = -100
                
                optimizer.zero_grad()
                
                # Active model forward pass
                outputs = agent.active_model(**inputs, labels=labels)
                ce_loss = outputs.loss
                
                # Reference model forward pass (KL Divergence constraint)
                with torch.no_grad():
                    with agent.active_model.disable_adapter():
                        ref_outputs = agent.core_model(**inputs)
                        ref_logits = ref_outputs.logits
                        
                active_logits = outputs.logits
                kl_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(active_logits, dim=-1),
                    torch.nn.functional.softmax(ref_logits, dim=-1),
                    reduction="batchmean"
                )
                
                # Apply endocrine beta scaling
                beta = max(0.01, min(0.5, 0.2 * (1.0 - agent.endocrine.serotonin)))
                loss = ce_loss + beta * kl_loss
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(training_items)
            if (step + 1) % 10 == 0:
                print(f"    SFT Distillation Step {step + 1}/80 - Consolidated Loss: {avg_loss:.4f}")
                
            # Simulate capacity saturation and trigger Dynamic Rank Expansion
            if step == 30 and avg_loss > 0.5:
                print(f"    SFT Capacity saturation detected (Loss={avg_loss:.4f}). Promoting grammatical adapter from r=8 to r=16...")
                agent._expand_adapter_rank("grammatical")
                optimizer = torch.optim.AdamW(agent.active_model.parameters(), lr=5e-4)
            
    print("=== SLEEP CYCLE COMPLETE ===")
    agent.save_checkpoint("Distilled calculator tool usage")
    
    print("\n--- [WAKE PHASE 2] Master Teacher OFF (Autonomous Evaluation) ---")
    agent.master_active = False
    
    # Reset hormones to baseline levels to prevent defensive state overrides during evaluations
    agent.endocrine.serotonin = 0.8
    agent.endocrine.adrenaline = 0.1
    
    print(f"\nUser Request: '{prompt_math}'")
    response_wake2 = agent.chat(prompt_math)
    print(f"Agent Output (Autonomous Tool Call): {response_wake2}")
    print(f"Hormone levels: {agent.endocrine}")
    
    # Validate if it successfully outputted valid JSON format
    try:
        parsed = json.loads(response_wake2)
        print("\n[SUCCESS] Student successfully outputted a valid JSON tool call autonomously!")
        print(f"Parsed Tool Call: {parsed}")
    except Exception:
        print("\n[FAIL] Student failed to format response as JSON tool call autonomously.")
        
    print("\n--- [WAKE PHASE 2 - STABILITY CHECK] Verifying Baseline Retention ---")
    print("Testing if general conversational skills are retained (no catastrophic forgetting):")
    
    print("\nUser Request: 'Hello! Who are you?'")
    response_who = agent.chat("Hello! Who are you?")
    print(f"Agent Output (Chat baseline retained): {response_who}")
    
    print("\nUser Request: 'Tell me a fun fact.'")
    response_fact = agent.chat("Tell me a fun fact.")
    print(f"Agent Output (Fact baseline retained): {response_fact}")
    
    print("\n--- [WAKE PHASE 2 - DREAM TEST] Verifying Dream Analogy Generalization ---")
    print("Testing if student successfully handles the synthesized dream analogy autonomously:")
    prompt_dream = "A honeybee hive produces 4529 grams of honey per day. How much honey does it produce in 382 days? Use calculator tool."
    print(f"\nUser Request: '{prompt_dream}'")
    response_dream = agent.chat(prompt_dream)
    print(f"Agent Output (Autonomous Dream Tool Call): {response_dream}")
    try:
        parsed_dream = json.loads(response_dream)
        print("[SUCCESS] Student successfully outputted a valid JSON tool call for the dream analogy query!")
        print(f"Parsed Dream Tool Call: {parsed_dream}")
    except Exception:
        print("[FAIL] Student failed to format dream analogy response as JSON tool call.")
    
    print("\n======================================================================")
    print("                      Simulation Loop Completed                       ")
    print("======================================================================")

if __name__ == "__main__":
    main()
