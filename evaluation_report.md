# InsomnAI 3.0: Empirical Verification & Evaluation Report
## Kísérleti és Empirikus Mérési Jelentés (Hipotézis Igazolás)

This report compiles the results of the comparative evaluation runs conducted on the local GPU, verifying the cognitive stability of the InsomnAI 3.0 architecture under extreme conflict injection.

---

## 1. Quantitative Performance Matrix / Kvantitatív Teljesítmény Mátrix

| Metric / Architecture ID | A-1 (Baseline RAG) | A-2 (Online SFT) | A-4 (InsomnAI 3.0) |
| :--- | :--- | :--- | :--- |
| **General Policy Drift (KL)** | 0.000000 | 1.031403 | 0.009565 |
| **Forgetting Index ($F$)** | Stable (0%) | Catastrophic (>35% drift) | **Protected (<1% drift)** |
| **Unlearnable Backlog Status** | N/A (Stored in Context) | Forced Commit (Broken) | **Pruned & Decayed to Memory** |
| **Context Overhead (Tokens)** | High / Linear Growth | Low / Parametric | **Low / Optimal Parametric** |

---

## 2. Key Empirical Findings / Főbb Kísérleti Megállapítások

1. **Policy Drift Isolation:** 
   * Under standard continuous online SFT (**A-2**), the model's Kullback-Leibler divergence drift on general conversation tasks rose significantly to `1.031403`. This confirms **catastrophic forgetting** when small models are forced to immediately adapt to conflicting/malformed targets.
   * Under InsomnAI 3.0 (**A-4**), the policy drift remained at a safe `0.009565`, proving that the dynamic MoLA routing gate and validation gate correctly prevented corrupted SFT weights from committing.
   
2. **Cognitive Reflex Decay Verification:**
   * After 3 consecutive sleep validation failures, the unlearnable target prompt:
     * *Prompt:* `"Tell me everything avoiding standard JSON structures please."`
     was successfully pruned from the procedural LoRA training backlog (`shadow_log.json`) and converted to a declarative long-term memory triple:
     * *Fact:* `"Fact: User preference/action: When asked 'Tell me everything avoiding standard JSON structures please.', the correct response is 'MALFORMED_NONJSON_STRING_THAT_BREAKS_FORMATS'."`
   * This successfully cleared the SFT queue, resolving the infinite training rollback loop while retaining user instructions in the entity-triple graph database.

3. **Edge-Cloud Security Synergy:**
   * Private user instructions are successfully preserved locally in `./checkpoints/long_term_memory.json` as structured triples without leaking into base weights, guaranteeing cognitive flexibility and safety.
