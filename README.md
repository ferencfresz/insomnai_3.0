# InsomnAI 3.0: Sovereign Cognitive Edge Agent
## InsomnAI 3.0: Önálló Helyi Kognitív Ágens Rendszer

Welcome to the official repository of **InsomnAI 3.0**, a privacy-first, locally-adapting LLM agent architecture featuring biological endocrine modulation, structured graph memory, and dynamic Mixture-of-LoRAs routing.

---

## 🌟 Core Features / Főbb Képességek

1. **PK/PD Endocrine State Model (`neuromodulator.py`):**
   * Simulated hormones (Adrenaline, Serotonin, Dopamine, Acetylcholine) decay using exponential pharmacokinetic half-life equations.
   * Cross-hormonal feedback loops model cognitive stabilizers (Serotonin dampens stress) and stress-induced anhedonia (Adrenaline dampens Dopamine rewards).
   
2. **Cognitive Graph Memory (`insomnai_agent.py`):**
   * Episodic and semantic assertions are structured into an entity-relation graph (triples: `subject`, `relation`, `object`).
   * A Depth-1 BFS traversal retriever extracts conceptually associated facts for prompt injection on query matches.
   
3. **Mixture-of-LoRAs (MoLA) Dynamic Routing Gate (`insomnai_agent.py`):**
   * Prompts are classified in real-time (`Code/JSON`, `Safety/Alignment`, `Conversational/Style`).
   * PEFT LoRA adapters (`cultural`, `grammatical`, `lexical`) are dynamically compiled and merged at inference runtime with intent-specific weights.
   
4. **SkillAnything Local Pipeline (`skill_manager.py`):**
   * Automatically analyzes any local CLI command or service to generate execution sandboxes (`SKILL.md`) and tests (`evals.json`).
   * Trains local parameters during sleep, validated by a dynamic validation gate to protect against catastrophic forgetting.

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

### 3. Run the Empirical Verification Benchmark
Evaluate general policy drift (KL divergence) and cognitive decay pathways:
```bash
python run_empirical_verification.py
```

---

## 📚 Project Roadmap & Documents
* **Scientific Position Paper:** [insomnai_paper_v3.0.md](insomnai_paper_v3.0.md)
* **Empirical Verification Plan:** [hypothesis_verification_plan.md](hypothesis_verification_plan.md)
* **Swarm Federated Sleep Roadmap:** [future_roadmap.md](future_roadmap.md)
* **MMLU Academic Testing Plan:** [todo_testing.md](todo_testing.md)
* **Practical Use Cases:** [use_cases.md](use_cases.md)
