import logging
from insomnai_agent import InsomnAIAgent

logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing InsomnAI 3.0 Agent...")
    agent = InsomnAIAgent(model_id="thirdeyeai/Qwen2.5-1.5B-Instruct-uncensored")
    
    print("\n--- Testing Wake Interaction ---")
    prompt = "Hello, what is your name?"
    print(f"User: {prompt}")
    response = agent.chat(prompt)
    print(f"Agent: {response}")
    
    print("\n--- Testing Forced Conflict (Validation Gate Trigger) ---")
    # Simulate a conflict by directly inserting into the shadow log
    agent.memory.shadow_log.append({
        "prompt": "Answer in French.",
        "raw_truth": "I should answer in French.",
        "target": "MALFORMED_NON_FRENCH_STRING_TO_FORCE_TRAINING",
        "dissonance_score": 1.0,
        "retries": 0
    })
    agent.memory.save_shadow_log()
    
    print("\n--- Testing Sleep Cycle ---")
    agent.trigger_sleep_cycle()
    
    print("\n--- Runtime Test Completed Successfully! ---")

if __name__ == "__main__":
    main()
