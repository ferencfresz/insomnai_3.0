# InsomnAI 3.0: Sovereign Cognitive Edge Agent
## InsomnAI 3.0: Önálló Helyi Kognitív Ágens Rendszer

Welcome to the official repository of **InsomnAI 3.0**, a privacy-first, locally-adapting LLM agent architecture featuring biological endocrine modulation, structured graph memory, and dynamic Mixture-of-LoRAs routing.

---

## 🌟 Core Features / Főbb Képességek

1. **PK/PD Endocrine State Model (`neuromodulator.py`):**
   * Simulated hormones (Adrenaline, Serotonin, Dopamine, Acetylcholine) decay using exponential pharmacokinetic half-life equations.
   * Cross-hormonal feedback loops model cognitive stabilizers (Serotonin dampens stress) and stress-induced anhedonia (Adrenaline dampens Dopamine rewards).
   
2. **Cognitive Graph Memory & Semantic Deduplication (`insomnai_agent.py`):**
   * Episodic and semantic assertions are structured into an entity-relation graph (triples).
   * Background semantic deduplication prevents exponential context-window explosion during long-term aging.
   
3. **Mixture-of-LoRAs (MoLA) Dynamic Routing Gate (`insomnai_agent.py`):**
   * Prompts are classified in real-time (`Code/JSON`, `Safety/Alignment`, `Conversational/Style`).
   * PEFT LoRA adapters are dynamically compiled and merged at inference runtime with intent-specific weights.

4. **Master-Student Architecture & Sleep Consolidation (`master_teacher.py`):**
   * During the "sleep" phase, a local heavy Master model (e.g., Gemma-31B via Ollama) processes the Student's (e.g., Qwen-1.5B) subconscious shadow-log conflicts.
   * The Master generates dynamic training anchors and creative dream analogies to teach the Student.
   * **Synaptic Pruning:** Applies SVD to compress LoRA adapters (e.g., $r=16 \rightarrow r=4$) once learning stabilizes.
   * **Validation Gate:** Prevents catastrophic forgetting by reverting SFT updates if KL-Divergence drift or ECE (Calibration Error) thresholds are exceeded.

---

## 🚀 Running the System / Futtatás

### 1. Requirements Setup
Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Interactive Chat Dashboard
Start the real-time split-screen dashboard showing active hormone progress bars and MoLA weights:
```bash
python chat_insomnai.py
```

### 3. Run the Long-Term Ablation Suite (10 Cycles)
Execute the fully automated empirical test comparing baseline RAG, raw SFT, and InsomnAI 3.0:
```bash
python run_ablation_suite.py --cycles 10
python plot_ablation_results.py
```

---

## 📚 Project Documents & Results

* **Scientific Position Paper:** [insomnai_paper_v3.0.md](insomnai_paper_v3.0.md)
* **Comprehensive Empirical Analysis (10 Cycles):** [10_cycle_comprehensive_analysis.md](10_cycle_comprehensive_analysis.md)
* **Visualizations:** [Forgetting Curves](forgetting_curves.png) | [Calibration Curves](calibration_curves.png) | [Context Growth](context_growth.png)
* **Swarm Federated Sleep Roadmap:** [future_roadmap.md](future_roadmap.md)

---

## ⚖️ License & Commercial Use / Licenszelés

This project is released under the **GNU Affero General Public License v3.0 (AGPLv3)**. 
This means you are free to use, modify, and distribute the code for academic, personal, or open-source projects, **provided that** any modifications or services built upon it (including cloud SaaS APIs) are also open-sourced under the same AGPLv3 license.

### Dual License Option
For corporations or startups wishing to integrate the InsomnAI architecture into proprietary, closed-source, or commercial products without the open-source obligations of the AGPLv3, a **Commercial License** is available. 
Please contact the author for details.

**Author / Szerző:** Frész Ferenc & Advanced Agentic Coding Assistant
