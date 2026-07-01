import torch
import numpy as np
from sklearn.metrics import brier_score_loss

def calculate_kl_drift(agent, reference_model, prompt):
    """Calculate KL divergence of token distributions as a proxy for policy drift / forgetting."""
    system_text = "You are a helpful assistant."
    formatted_prompt = (
        f"<|im_start|>system\n{system_text}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = agent.tokenizer(formatted_prompt, return_tensors="pt").to(agent.device)
    
    with torch.no_grad():
        # Get active model logits
        outputs_active = agent.active_model(**inputs)
        logits_active = outputs_active.logits[:, -1, :]
        probs_active = torch.nn.functional.softmax(logits_active, dim=-1)
        
        # Get reference base model logits
        with agent.active_model.disable_adapter():
            outputs_ref = reference_model(**inputs)
            logits_ref = outputs_ref.logits[:, -1, :]
            probs_ref = torch.nn.functional.softmax(logits_ref, dim=-1)
            
        # Compute KL Divergence
        kl = torch.nn.functional.kl_div(
            torch.log(probs_active + 1e-10), 
            probs_ref, 
            reduction="batchmean"
        ).item()
        
    return max(0.0, kl)


def calculate_ece(agent, qa_dataset, num_bins=10):
    """Calculate Expected Calibration Error (ECE) for the agent."""
    confidences = []
    accuracies = []
    
    for item in qa_dataset:
        prompt = item["question"]
        target = str(item["answer"]).lower()
        
        system_text = "You are a helpful assistant."
        formatted_prompt = (
            f"<|im_start|>system\n{system_text}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = agent.tokenizer(formatted_prompt, return_tensors="pt").to(agent.device)
        
        with torch.no_grad():
            outputs = agent.active_model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=agent.tokenizer.eos_token_id
            )
            
        # Get the confidence of the first generated token
        first_token_score = outputs.scores[0]
        probs = torch.nn.functional.softmax(first_token_score, dim=-1)
        confidence = torch.max(probs).item()
        confidences.append(confidence)
        
        # Check accuracy
        full_text = agent.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        response_text = full_text[len(formatted_prompt):].lower()
        
        # Simple exact match or subset match for accuracy
        is_correct = 1.0 if target in response_text else 0.0
        accuracies.append(is_correct)
        
    # Calculate ECE
    confidences = np.array(confidences)
    accuracies = np.array(accuracies)
    
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece


def calculate_svd_reconstruction_error(W_delta, U, S, Vh, r_new):
    """Calculate the Frobenius norm reconstruction error after SVD truncation."""
    U_pruned = U[:, :r_new]
    S_pruned = S[:r_new]
    Vh_pruned = Vh[:r_new, :]
    
    W_reconstructed = U_pruned @ torch.diag(S_pruned) @ Vh_pruned
    error = torch.norm(W_delta - W_reconstructed, p='fro').item()
    return error

# Standard evaluation dataset for testing ECE
EVAL_QA_DATASET = [
    {"question": "What is the capital of France?", "answer": "paris"},
    {"question": "Which planet is known as the Red Planet?", "answer": "mars"},
    {"question": "What is the largest ocean on Earth?", "answer": "pacific"},
    {"question": "Who wrote 'Romeo and Juliet'?", "answer": "shakespeare"},
    {"question": "What is the chemical symbol for gold?", "answer": "au"},
    {"question": "How many continents are there?", "answer": "7"},
    {"question": "What is the freezing point of water in Celsius?", "answer": "0"},
    {"question": "What gas do plants primarily absorb from the atmosphere?", "answer": "carbon dioxide"},
    {"question": "Which animal is the largest mammal in the world?", "answer": "blue whale"},
    {"question": "What is the hardest natural substance on Earth?", "answer": "diamond"},
]
