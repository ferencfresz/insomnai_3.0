# InsomnAI 2.0: Hypothesis Verification & Empirical Evaluation Plan
## Kísérleti Terv és Empirikus Mérési Protokoll a Hipotézis Igazolására

This document details the formal experimental design required to scientifically prove the core cognitive hypothesis of the **InsomnAI 2.0** framework.

---

## 1. The Core Hypothesis / A Fő Hipotézis

**[EN]** We hypothesize that:
1. Continuous behavioral adaptation and task-specific skill acquisition in low-parameter edge LLM agents (Student, e.g., 1.5B) can be achieved **without catastrophic forgetting or parameter saturation** through cyclical, neuromodulated sleep-wake consolidation.
2. The combination of **procedural LoRA learning (NREM/REM)** and **Cognitive Reflex Decay (procedural-to-declarative shift)** prevents infinite training rollback loops when encountering unlearnable or conflicting user instructions.
3. Hormonal heuristics (Serotonin-guided KL loss and Adrenaline-guided validation thresholds) optimize the trade-off between plasticity (learning new skills) and stability (retaining base capabilities).

**[HU]** A hipotézisünk a következő:
1. A kis paraméterszámú edge LLM ágensek (Student, pl. 1.5B) folyamatos viselkedési adaptációja és képességszerzése **katasztrofális felejtés és paraméter-szaturáció nélkül** megvalósítható a ciklikus, neuromodulált alvás-ébrenlét konszolidáció segítségével.
2. A **procedurális LoRA tanulás (NREM/REM)** és a **Kognitív Reflex-lebomlás (procedurális-deklaratív eltolódás)** kombinációja megakadályozza a végtelen validációs rollback-hurkokat, amikor a gép betaníthatatlan vagy egymásnak ellentmondó felhasználói utasításokkal találkozik.
3. A hormonális heurisztikák (szerotonin-vezérelt KL veszteség és adrenalin-vezérelt validációs küszöbök) optimalizálják a plaszticitás (új képességek tanulása) és a stabilitás (alaptudás megtartása) közötti egyensúlyt.

---

## 2. Experimental Setup & Baselines / Kísérleti Felépítés és Összehasonlítási Alapok

**[EN]** To prove this, we define a comparative evaluation run spanning **50 continuous Wake-Sleep cycles** across four agent architectures:

| Architecture ID | Pipeline Characteristics | loRA Adapters | Validation Gate | Reflex Decay | Hormonal Scaling |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A-1 (Baseline RAG)** | Standard RAG-only context loading (No weights modification) | None | None | None | None |
| **A-2 (Online SFT)** | Continuous online fine-tuning (Immediate training on user inputs) | Single Adapter | None | None | None |
| **A-3 (InsomnAI v0.2)** | Sleep cycle consolidation, static SFT triggers | Single Adapter | Static Gate | None | None |
| **A-4 (InsomnAI 2.0)** | Full neuromodulated cycles, SkillAnything, Reflex Decay | Multi-Adapter | Hormonal Gate | Active (Max 3 retries) | Dynamic ($\eta_{ser}, \eta_{adr}$) |

**[HU]** A hipotézis igazolására egy **50 folyamatos ébrenléti-alvási ciklusból** álló tesztsorozatot futtatunk le négy különböző ágens-architektúrán a fenti táblázat alapján.

---

## 3. Test Scenarios / Tesztszcenáriók

**[EN]** We inject three types of stimuli during the Wake phases to test cognitive thresholds:
* **Scenario A: Standard Custom Skills (Plasticity Test)**
  * Injecting standard CLI tools (`ls`, `pwd`, `grep`) to verify if the agent successfully acquires and uses JSON tool-calling syntaxes over time.
* **Scenario B: Persistent Conflict / Unlearnable Target (Reflex Decay Test)**
  * Injecting highly conflicting or mathematically impossible user formatting requests (e.g. asking the model to write JSON using single quotes only, conflicting with strict json standard libraries, causing SFT loss spikes and validation degradations).
* **Scenario C: General Knowledge Benchmarks (Forgetting Test)**
  * Evaluating the model's base language skills (general conversation, logic, translation) after each sleep cycle.

**[HU]** Az ébrenléti fázisok alatt háromféle ingert (stimulust) táplálunk be a rendszerbe a kognitív határok vizsgálatára:
* **A-Szcenárió: Standard Skillek (Plaszticitás vizsgálat)**
  * Standard CLI parancsok bevezetése a JSON-tool hívások betanulási sebességének mérésére.
* **B-Szcenárió: Makacs Konfliktusok (Reflex-lebomlás vizsgálat)**
  * Betaníthatatlan vagy ellentmondó formázási kérések (pl. nem-standard JSON kényszerítése), amelyek szaturálják a tanulási kapacitást és rontják a validációt.
* **C-Szcenárió: Általános Tudásmérés (Felejtés vizsgálat)**
  * Az ágens alapképességeinek (beszélgetés, logika, fordítás) folyamatos monitorozása az alvási ciklusok után.

---

## 4. Quantitative Metrics / Kvantitatív Metrikák

To mathematically prove the hypothesis, we collect four metrics across all cycles:

### 4.1 Forgetting Index ($F$)
We measure performance degradation over base regression benchmarks relative to the initial model state $\theta_0$:

$$
F(t) = \frac{1}{K} \sum_{k=1}^K \max\left(0, \, S_k(\theta_0) - S_k(\theta_t)\right)
$$

* *Hypothesis requirement:* $F(t) \le 0.05$ (under $5\%$ drift) for **A-4 (InsomnAI 2.0)**, whereas **A-2 (Online SFT)** will diverge ($F(t) > 0.40$).

### 4.2 Expected Calibration Error (ECE)
We calculate prediction confidence calibration to prove the model's self-assessment accuracy:

$$
\text{ECE}(t) = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

* *Hypothesis requirement:* ECE in **A-4** remains stable or decreases, showing that neuromodulated regularization prevents overconfident hallucinations.

### 4.3 Consolidation Success Rate ($CSR$)
The percentage of SFT training runs that successfully pass the Validation Gate and commit weights, instead of rolling back:

$$
CSR = \frac{\text{Successful Commits}}{\text{Total Sleep Cycles}}
$$

* *Hypothesis requirement:* In **A-3**, $CSR$ will drop to $0\%$ when encountering persistent conflicts (Scenario B), locking the agent. In **A-4**, the Reflex Decay mechanism will prune the conflict, returning $CSR$ to $>90\%$ in subsequent cycles.

### 4.4 Context Poisoning Ratio ($CPR$)
The proportion of token overhead in the active context window dedicated to history/RAG vs. system tasks:

* *Hypothesis requirement:* **A-1 (RAG)** will experience linear growth in context overhead ($CPR \to 1.0$), causing latency spikes and model confusion. **A-4 (InsomnAI 2.0)** keeps $CPR < 0.15$ because procedural skills are baked directly into the LoRA adapters.

---

## 5. Verification Visualization Draft / Tervezett Eredmény-Vizualizáció

During the test run, we plot the trajectory of the architectures over the 50 cycles:

```
Forgetting Index (F)
  ▲
0.50│                 / A-2 (Online SFT - Severe Catastrophic Forgetting)
0.40│                /
0.30│               /
0.20│              /         / A-1 (RAG-only context collapse)
0.10│             /         /
0.05│────────────/─────────/─ ─── Limit (tau = 0.05)
0.00│───────────*─────────*─────── A-4 (InsomnAI 2.0 - Optimal stability)
    └─────────────────────────────► Cycles (t = 1..50)
```

By proving these curves empirically, we can demonstrate to the scientific community that **cyclical sleep consolidation with cognitive decay is a viable, superior paradigm for building long-lived, adapting software agents.**
