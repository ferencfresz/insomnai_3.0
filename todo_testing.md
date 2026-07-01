# InsomnAI 2.0: Academic Testing & Verification Roadmap
## Tudományos Tesztelési és Empirikus Verifikációs Feladatlista (Roadmap)

This document outlines the concrete steps required to scale the pilot results of **InsomnAI 2.0** into a comprehensive, peer-reviewed evaluation suite for major AI conferences (e.g., NeurIPS, ICLR, ACL).

---

## 🟩 Phase 1: Model Generalization Benchmarking
*Prove the neuromodulated sleep-cycle framework is architecture-agnostic.*

- [ ] **Multi-Model Config Suite:**
  - Extend the runner to support multiple local student models (PEFT-compatible):
    - `Llama-3-8B-Instruct` (8B parameters)
    - `Mistral-7B-Instruct-v0.3` (7B parameters)
    - `Phi-3-medium-128k-instruct` (14B parameters)
- [ ] **Target Module Mapping:**
  - Define custom target module configurations for PEFT LoRA in `insomnai_agent.py` to automatically adapt to different model layouts (e.g., `["q_proj", "k_proj", "v_proj", "o_proj"]` for Llama vs. `["q_proj", "v_proj"]` for Qwen).

---

## 🟩 Phase 2: Benchmark Dataset Integrations
*Replace proxy queries with standard academic evaluation benchmarks.*

- [ ] **Massive Multitask Language Understanding (MMLU):**
  - Integrate a sub-selection of MMLU categories (e.g., humanities, STEM, social sciences) as the baseline regression suite $\mathcal{V}_{regression}$ to calculate the Forgetting Index ($F$).
- [ ] **GSM8K & MATH:**
  - Add grade-school math benchmarks to verify reasoning capabilities during continuous adaptation cycles.
- [ ] **HumanEval:**
  - Integrate Python coding problems to evaluate if custom coding skills are successfully consolidated without degrading base coding proficiency.

---

## 🟩 Phase 3: Long-Term Aging & Durability Simulations
*Demonstrate system stability over extended operational lifetimes.*

- [ ] **100-Cycle Continuous Simulation Script:**
  - Automate a simulation representing 100 consecutive Wake-Sleep cycles (equivalent to ~3 months of active deployment usage).
- [ ] **Interval Metric Collection:**
  - Record the following values at cycles 10, 20, 30, 40, 50, 75, and 100:
    - General Policy Drift (KL Divergence over 200 reference prompts).
    - Expected Calibration Error (ECE) trend lines.
    - SVD Synaptic Pruning efficiency (Frobenius norm reconstruction error of LoRA adapters).
    - RAG Fact Retrieval latency and context overhead (Tokens).
- [ ] **Fact Dedup & Merging:**
  - Implement a semantic merging routine (using sentence embeddings or local Master queries) to deduplicate and compress the declarative `long_term_memory.json` space once it grows beyond 50 entries.

---

## 🟩 Phase 4: Ablation Studies & Chart Automation
*Generate high-quality evaluation graphs for the paper.*

- [ ] **Ablation Automation Runner:**
  - Create a master script `run_ablation_suite.py` that executes parallel/sequential runs for:
    - **A-1:** RAG-only Baseline.
    - **A-2:** Continuous Online SFT.
    - **A-3:** InsomnAI v0.2 (no hormones or decay).
    - **A-4:** InsomnAI 2.0 (Full framework).
- [ ] **Export to CSV:**
  - Log cycle number, architecture, ECE, Forgetting Index, and Consolidation Success Rate to a structured CSV file.
- [ ] **Matplotlib Chart Generator:**
  - Code an automated chart generator to plot:
    - Forgetting curves: $F(t)$ vs. Cycles.
    - Calibration curves: ECE vs. Cycles.
    - Context growth rate comparison: Context Window Token overhead vs. Cycles.
