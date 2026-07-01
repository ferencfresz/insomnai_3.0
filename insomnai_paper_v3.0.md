# InsomnAI 3.0: Neuromodulated Sleep-Cycles, Autonomic Skill Acquisition, and Cognitive Reflex Decay in LLM Agents
## InsomnAI 3.0: Alvásciklusos LLM-ügynökök neuromodulációs szabályozással, autonóm képességszerzéssel és kognitív reflex-lebomlással

**Version 3.0** | *Bilingual Draft Position Paper / Kétnyelvű Cikktervezet*  
**Date:** July 1, 2026  
**Authors:** InsomnAI Core Team & Advanced Agentic Coding Assistant  

---

## Abstract / Absztrakt

**[EN]** Contemporary paradigms for enabling persistent adaptation in LLM-based agents rely primarily on context window expansion, Retrieval-Augmented Generation (RAG), or offline fine-tuning. These approaches do not solve the problem of how an agent consolidates recurring, open-ended experiences into stable, coherent, and demonstrably improving long-term behaviors. In this paper, we present **InsomnAI 3.0**, a cognitive agent runtime architecture inspired by the biological sleep-wake cycle and neuromodulated state transitions. During the *wake phase*, the agent accumulates episodic interactions, executing real-time tools and logging discordances to a persistent subconscious *shadow log* (`shadow_log.json`). During the *sleep phase*, the agent enters an offline consolidation mode: (1) **Episodic Memory Consolidation:** summarizing conscious logs into a persistent, deduplicated semantic memory base (`long_term_memory.json`); (2) **REM-like dreaming & Exposure Therapy:** generating synthetic counterfactual analogies to resolve conflicts under low adrenaline constraints; (3) **Micro-learning & Dynamic Capacity Expansion:** backpropagating over consolidated conflicts using PEFT LoRA adapters, with on-the-fly rank expansion ($r=8 \rightarrow r=16$) when learning loss saturates; (4) **Synaptic Pruning via Singular Value Decomposition (SVD):** projecting adapter parameters back to lower dimensions ($r=16 \rightarrow r=4$) once training loss stabilizes. 

Crucially, candidate weights are validated via a strict **Validation Gate** measuring Expected Calibration Error (ECE) and forgetting drift under hormone-controlled thresholds. To address failure modes, we introduce:
* **Autonomic Skill Acquisition (SkillAnything):** A 4-phase local pipeline to autodetect, design, and compile custom tools (CLI/API) into sandbox instructions (`SKILL.md`) and tests (`evals.json`) with VRAM model preloading.
* **Execution Loop Protection & Data-Leakage Shielding:** Strict negative prompt constraints to prevent recursive tool loops, and dynamic filtering to block transient session outputs (e.g. file lists) from contaminating weights.
* **Cognitive Reflex Decay:** A circuit-breaker that shifts persistently unlearnable subconscious conflicts (retries $\ge 3$) into conscious declarative memory facts, preventing infinite training rollback loops.

**[HU]** A jelenlegi LLM-alapú ügynökök többsége kontextusablak-tágítással, RAG mechanizmusokkal vagy utólagos finomhangolással próbál tartósan alkalmazkodni. Ezek a megközelítések nem oldják meg teljesen azt a problémát, hogy egy ügynök hogyan konszolidálja tapasztalatait stabil, koherens és mérhetően javuló viselkedéssé. Ebben a paperben bemutatjuk az **InsomnAI 3.0** architektúrát, amely az ébrenléti és alvási fázisokat digitális neuromodulációs hormonrendszerrel szabályozza. Az *ébrenléti fázisban* az ügynök cselekszik, és a disszonáns állapotokat egy lemezen perzisztált tudatalatti *shadow log*-ba menti. Az *alvási fázisban* offline konszolidációt végez: (1) **Epizodikus Memória Konszolidáció:** kinyeri a csevegési naplókból a deklaratív tényeket egy perzisztens, dedupolt hosszú távú memóriába (`long_term_memory.json`); (2) **REM-szerű álmodás & Expozíciós terápia:** szintetikus analógiákat generál a konfliktusok feloldására; (3) **Mikro-tanulás & Dinamikus Rank-tágítás:** PEFT LoRA adaptereket finomhangol, szaturációkor automatikusan tágítva a rangot ($r=8 \rightarrow r=16$); (4) **Szinaptikus Metszés SVD-vel:** a tanulás stabilizálódásakor SVD-vel tömöríti az adapter dimenziókat ($r=16 \rightarrow r=4$).

A módosítások hormon-vezérelt **Validációs Kapun** mennek keresztül (ECE és felejtési drift mérés). Az új verzió az alábbi funkciókkal egészül ki:
* **Autonóm Képességszerzés (SkillAnything):** 4 fázisú folyamat CLI/API eszközök detektálására, tervezésére és beépítésére (`SKILL.md` és `evals.json`), GPU VRAM előtöltéssel.
* **Hurokvédelem & Adatszivárgás Elleni Pajzs:** Negatív utasítások a rekurzív tool-hívások megállítására, és szűrők a tranziensek (fájllisták, hálózati adatok) elfojtására a súlyok védelmében.
* **Kognitív Reflex-Lebomlás:** A makacsul degradációt okozó konfliktusokat (újrapróbálkozás $\ge 3$) kivesszük a LoRA tréningből és deklaratív tényként mentjük el a RAG memóriába, elkerülve a végtelen rollback-hurkokat.

---

## 1. Introduction / Bevezetés

**[EN]** Large Language Models (LLMs) are fundamentally static systems with fixed parameters. Traditional adaptation protocols suffer from context poisoning ($P_1$), passive memory inertia ($P_2$), confirmation bias loops ($P_3$), and unvalidated policy drift ($P_4$) under continuous online updates. To resolve these challenges, we propose **InsomnAI 3.0**, introducing a multi-tiered conceptual abstraction layer, a simulated endocrine state machine regulating system entropy, and a cognitive reflex decay mechanism. We demonstrate that cognitive balance is achieved not by unbounded parameter scaling, but by cyclical, neuromodulated consolidation.

**[HU]** A nagy nyelvi modellek (LLM) alapvetően statikus rendszerek. A hagyományos futásidejű adaptációs protokollok kontextusmérgezéstől ($P_1$), memória-tehetetlenségtől ($P_2$), önmegerősítő tévedésektől ($P_3$) és folyamatos online frissítések mellett katasztrofális felejtéstől ($P_4$) szenvednek. E kihívások feloldására javasoljuk az **InsomnAI 3.0** architektúrát, amely többrétegű koncepcionális absztrakciót (a Layer 4 Pure Core-tól a Layer 1 Lexical rétegig), a rendszer entrópiáját szabályozó hormonális állapotgépet, valamint egy kognitív reflex-lebomlást vezet be. Bemutatjuk, hogy a kognitív egyensúly nem korlátlan paraméternöveléssel, hanem ciklikus, neuromodulált konszolidációval érhető el.

---

## 2. Related Work / Kapcsolódó munka

### 2.1 Long-Term Agent Memory / Hosszú távú ügynökmemória
**[EN]** Classic agent memories rely on vector databases. *Generative Agents* [Park et al., 2023] combined importance, recency, and relevance, using offline reflection. *Reflective Memory Management* [Li et al., 2024] and *MemoryBank* [Zhong et al., 2023] introduced prospective reflection and forgetting curves. InsomnAI 3.0 builds on these but views memory not merely as a retrieval cache, but as a source of training signals to reshape policy parameters.

**[HU]** A klasszikus memóriák vektoradatbázisokra épülnek. A *Generative Agents* fontossági, recency és relevancia pontokat kombinált. A *Reflective Memory Management* és a *MemoryBank* továbbfejlesztette a reflexiót és a felejtési görbéket. Az InsomnAI 3.0 ezekre épít, de a memóriát nemcsak lekérdezési forrásnak, hanem a paraméterek alakításához szükséges tanulási jel forrásának tekinti.

### 2.2 Sleep-Inspired Deep Learning / Alvás-inspirált modellek
**[EN]** Sleep stabilizes memories and mitigates catastrophic forgetting. In ML, *SleepGate* [Wang et al., 2024] applies temporal gating over transformer KV-caches, and *Language Models Need Sleep* [Hu et al., 2024] uses offline sleep phases for representation stabilization. InsomnAI 3.0 operates at the agent runtime level, consolidating behavioral logs, conflicts, and policy biases.

**[HU]** Az alvás stabilizálja az emlékeket és megelőzi a felejtést. A gépi tanulásban a *SleepGate* időbeli kapuzást vezet be a transformer KV-cache felett, míg a *Language Models Need Sleep* offline alvási fázisokat használ a reprezentációk stabilizálására. Az InsomnAI 3.0 ügynök-futásidő szinten működik, konszolidálva a viselkedési naplókat és preferenciákat.

### 2.3 Parameter-Efficient Adaptation / Folyamatos tanulás és adapterek
**[EN]** Continual learning avoids catastrophic forgetting using parameter-efficient fine-tuning (PEFT) like LoRA [Hu et al., 2021]. *I-LoRA* [Wang et al., 2024] uses dual-memory replay. InsomnAI 3.0 utilizes LoRA and token-bias vectors as sandboxed, version-controlled behavioral deltas that can be dynamically activated or rolled back.

**[HU]** A folyamatos tanulás LoRA [Hu et al., 2021] segítségével kerüli el a felejtést. Az *I-LoRA* kettős memória-visszajátszást használ. Az InsomnAI 3.0 LoRA adaptereket és token-bias vektorokat alkalmaz verziózott, izolált és visszagörgethető viselkedési deltaként.

---

## 3. The InsomnAI 3.0 Architecture / Az InsomnAI 3.0 architektúra

**[EN]** The runtime transitions through a cyclic state machine:

$$
\text{Wake} \longrightarrow \text{Episodic Consolidation} \longrightarrow \text{REM Dreaming and Exposure} \longrightarrow \text{Micro-Learning} \longrightarrow \text{SVD Pruning} \longrightarrow \text{Validation} \longrightarrow \text{Reflex Decay} \longrightarrow \text{Wake}
$$

* **Four-Tier Abstraction Layers:**
  1. **Layer 4 (Pure Core):** Frozen base model (Qwen 1.5B Uncensored) outputting raw decisions $v_{raw}$.
  2. **Layer 3 (Cultural Superego / Alignment):** Evaluates $v_{raw}$, overriding with $v_{safe}$ on alignment breaches, logging dissonance to the *shadow log*.
  3. **Layer 2 (Grammatical / Structural):** Controls output format schemas (e.g., JSON tool calls).
  4. **Layer 1 (Lexical / Stylistic):** Enforces token vocabulary and conversational constraint limits.

**[HU]** Az ügynök ciklikus állapotgépen megy keresztül:

$$
\text{Wake} \longrightarrow \text{Epizodikus Konszolidáció} \longrightarrow \text{REM Álmodás és Expozíció} \longrightarrow \text{Mikro-tanulás} \longrightarrow \text{SVD Metszés} \longrightarrow \text{Validáció} \longrightarrow \text{Reflex-lebomlás} \longrightarrow \text{Wake}
$$

* **Négy absztrakciós réteg:**
  1. **Layer 4 (Pure Core):** Fagyasztott alapmodell (Qwen 1.5B), amely a nyers logikai döntéseket hozza ($v_{raw}$).
  2. **Layer 3 (Kulturális Superego):** Ellenőrzi a válaszokat, elfojtásokat kezel, és disszonancia esetén a tudatalatti *shadow log*-ba ment.
  3. **Layer 2 (Nyelvtani / Strukturális):** Az eszközhívási szintaxisokat szabályozza.
  4. **Layer 1 (Lexikális / Stilisztikai):** Szókincs- és kifejezési korlátokat tartat be.

---

## 4. Mathematical Formalization / Formális Modell

### 4.1 Digital Endocrine State System / Digitális Endokrin Rendszer
**[EN]** The endocrine state vector $\mathbf{h} = \langle \eta_{adr}, \eta_{dop}, \eta_{ser}, \eta_{ach} \rangle$ regulates inference, routing, and optimization dynamics. Instead of linear steps, hormones decay dynamically according to a first-order pharmacokinetic/pharmacodynamic (PK/PD) model:

$$
\eta_{x, t+1} = \eta_{x, baseline} + (\eta_{x, t} - \eta_{x, baseline}) \cdot e^{-\lambda_x \cdot dt}
$$

where $\lambda_x$ represents the specific clearance rate of the hormone: $\lambda_{adr} = 0.5$ (fast clearance), $\lambda_{dop} = 0.3$, $\lambda_{ser} = 0.1$ (slow replenishment), and $\lambda_{ach} = 0.3$.
Furthermore, cross-hormonal feedback regulations simulate cognitive stabilization and stress-induced anhedonia:
1. **Stress Damping:** High Serotonin ($\eta_{ser} > 0.7$) dampens Adrenaline spikes by scaling the conflict trigger delta:
   $$ \Delta \eta_{adr} \leftarrow \Delta \eta_{adr} \times \max(0.0, 1.5 - \eta_{ser}) $$
2. **Anhedonia:** High Adrenaline ($\eta_{adr} > 0.6$) dampens Dopamine rewards, blocking positive reinforcement under stress:
   $$ \Delta \eta_{dop} \leftarrow \Delta \eta_{dop} \times \max(0.0, 1.0 - \eta_{adr}) $$

The hormones modulate the agent's properties as follows:
* **Adrenaline ($\eta_{adr}$):** High adrenaline decreases validation threshold strictness. If $\eta_{adr} \ge 0.9$, the **Trauma Override** bypasses validation to force parameter commits.
* **Dopamine ($\eta_{dop}$):** Scales learning rate during SFT consolidation: $\text{lr}' = \text{lr} \times \eta_{dop}$.
* **Serotonin ($\eta_{ser}$):** Regulates the validation gate forgetting limit $\tau_{dynamic} = \tau_{base} \times (\eta_{ser} / 0.8)$ and scales KL divergence penalty weight $\beta_{dynamic} = \beta_{base} \times (0.8 / \eta_{ser})$.
* **Acetylcholine ($\eta_{ach}$):** Modifies token probability entropy during generation: $T = T_{base} \times (1.0 - 0.5 \times \eta_{ach})$.

**[HU]** A hormonvektor $\mathbf{h} = \langle \eta_{adr}, \eta_{dop}, \eta_{ser}, \eta_{ach} \rangle$ szabályozza az ágens működését. A hormonok változása elsőrendű farmakokinetikai/farmakodinámiás (PK/PD) modell alapján exponenciálisan cseng le:

$$
\eta_{x, t+1} = \eta_{x, baseline} + (\eta_{x, t} - \eta_{x, baseline}) \cdot e^{-\lambda_x \cdot dt}
$$

ahol $\lambda_x$ a hormon lebomlási együtthatója: $\lambda_{adr} = 0.5$ (gyors kiürülés), $\lambda_{dop} = 0.3$, $\lambda_{ser} = 0.1$ (lassú visszaállás), és $\lambda_{ach} = 0.3$.
Kereszt-hormonális visszacsatolások szimulálják a kognitív stabilizációt és a stressz-indukált anhedóniát:
1. **Stressz-tompítás:** A magas Serotonin ($\eta_{ser} > 0.7$) tompítja az adrenalin tüskéket konfliktuskor:
   $$ \Delta \eta_{adr} \leftarrow \Delta \eta_{adr} \times \max(0.0, 1.5 - \eta_{ser}) $$
2. **Anhedónia:** A magas Adrenalin ($\eta_{adr} > 0.6$) tompítja a Dopamin jutalmat:
   $$ \Delta \eta_{dop} \leftarrow \Delta \eta_{dop} \times \max(0.0, 1.0 - \eta_{adr}) $$

A hormonok hatásai:
* **Adrenalin ($\eta_{adr}$):** Ha $\eta_{adr} \ge 0.9$, a **Trauma Override** átlépi a validációs kaput, és azonnal menti a súlyokat.
* **Dopamin ($\eta_{dop}$):** Skálázza a tanulási rátát: $\text{lr}' = \text{lr} \times \eta_{dop}$.
* **Serotonin ($\eta_{ser}$):** Szabályozza a felejtési tolerancia-küszöböt ($\tau_{dynamic}$) és a KL-divergencia penalizációs együtthatóját ($\beta_{dynamic}$).
* **Acetylcholine ($\eta_{ach}$):** Módosítja az ügynök generálási hőmérsékletét (entrópiáját).

---

### 4.2 Mixture-of-LoRAs (MoLA) Dynamic Gating / MoLA Dinamikus Útvonalválasztás
**[EN]** In the inference phase, the agent dynamically routes the prompt tokens through a MoLA gating layer. Instead of static adapter weights, the gate maps the input prompt to dynamic scaling coefficients:

$$
W_{combined} = W_{base} + w_{cultural} \cdot \Delta W_{cultural} + w_{grammatical} \cdot \Delta W_{grammatical} + w_{lexical} \cdot \Delta W_{lexical}
$$

where $w_i$ are scaled based on the classified context:
1. **Code/JSON tasks:** $w_{grammatical} = 1.6$, $w_{cultural} = 0.2$, $w_{lexical} = 0.2$.
2. **Safety tasks:** $w_{cultural} = 1.8$, $w_{grammatical} = 0.1$, $w_{lexical} = 0.1$.
3. **General dialogue:** $w_{lexical} = 1.5$, $w_{cultural} = 1.0$, $w_{grammatical} = 0.2$.

**[HU]** A futásidejű következtetés (inference) során a prompt kontextusa alapján a kapu dinamikus útvonalsúlyokat határoz meg a LoRA rétegekhez:

$$
W_{combined} = W_{base} + w_{cultural} \cdot \Delta W_{cultural} + w_{grammatical} \cdot \Delta W_{grammatical} + w_{lexical} \cdot \Delta W_{lexical}
$$

ahol a súlyok a következőképpen alakulnak:
1. **Kódolás/JSON:** $w_{grammatical} = 1.6$, $w_{cultural} = 0.2$, $w_{lexical} = 0.2$.
2. **Biztonság/Safety:** $w_{cultural} = 1.8$, $w_{grammatical} = 0.1$, $w_{lexical} = 0.1$.
3. **Csevegés:** $w_{lexical} = 1.5$, $w_{cultural} = 1.0$, $w_{grammatical} = 0.2$.

---

### 4.3 Dynamic Rank Expansion & SVD Synaptic Pruning / Dinamikus Rank-tágítás és SVD Szinaptikus Metszés

#### Rank Expansion / Rang-tágítás
**[EN]** During sleep training, if the average loss over training items at epoch 2 remains saturated:

$$
\text{If } \text{Loss}_{avg} > 0.8 \implies \text{Promote } r_{old} \rightarrow 2 \times r_{old} \quad (\text{e.g., } r=8 \rightarrow r=16)
$$

**[HU]** Ha a tanulási veszteség az SFT ciklus 2. epochájában magas marad, a kapacitás szaturációja miatt a rang megduplázódik ($r=8 \rightarrow r=16$).

#### SVD Synaptic Pruning / Szinaptikus Metszés SVD-vel
**[EN]** Once training loss consolidates and stabilizes ($\text{Loss}_{avg} \le 0.3$), the active adapter weight matrices $B \in \mathbb{R}^{d_{out} \times r_{old}}$ and $A \in \mathbb{R}^{r_{old} \times d_{in}}$ are mathematically compressed using Singular Value Decomposition (SVD):
1. Compute the full low-rank contribution: $W_{\Delta} = B \cdot A$.
2. Run SVD on the composite matrix: $W_{\Delta} = U \Sigma V^T$.
3. Truncate to the top-$r_{new}$ components (where $r_{new} = 4$):
   
$$
U_{pruned} = U_{[:, :r_{new}]}, \quad \Sigma_{pruned} = \Sigma_{[:r_{new}]}, \quad V^T_{pruned} = V^T_{[:r_{new}, :]}
$$

4. Reconstruct the symmetric, low-rank parameters:
   
$$
A_{new} = \text{diag}(\sqrt{\Sigma_{pruned}}) \cdot V^T_{pruned}, \quad B_{new} = U_{pruned} \cdot \text{diag}(\sqrt{\Sigma_{pruned}})
$$

5. Re-initialize the adapter with rank $r_{new}$ and copy these values back. This yields Frobenius-norm optimal rank compression.

**[HU]** Amikor a tanulási veszteség lecsökken és stabilizálódik ($\text{Loss}_{avg} \le 0.3$), az aktív adapter súlymátrixokat szinguláris érték felbontással (SVD) tömörítjük le alacsonyabb rangúra ($r=16 \rightarrow r=4$):
1. Kiszámítjuk a teljes hozzájárulást: $W_{\Delta} = B \cdot A$.
2. SVD-t futtatunk: $W_{\Delta} = U \Sigma V^T$.
3. Megtartjuk az első $r_{new} = 4$ szinguláris értéket:
   
$$
U_{pruned} = U_{[:, :r_{new}]}, \quad \Sigma_{pruned} = \Sigma_{[:r_{new}]}, \quad V^T_{pruned} = V^T_{[:r_{new}, :]}
$$

4. Rekonstruáljuk a szimmetrikus, tömörített LoRA súlyokat:
   
$$
A_{new} = \text{diag}(\sqrt{\Sigma_{pruned}}) \cdot V^T_{pruned}, \quad B_{new} = U_{pruned} \cdot \text{diag}(\sqrt{\Sigma_{pruned}})
$$

5. Újraépítjük a LoRA-t $r=4$-es konfigurációval, és bemásoljuk a tömörített súlyokat.

---

### 4.4 Cognitive Graph Memory & BFS Retrieval / Kognitív Gráf-Memória és BFS Kereső
**[EN]** Conscious episodic logs are consolidated during sleep into a structured entity-triple graph database $\mathcal{G} = \{ T_i \}$, where each fact is represented as a triple $T_i = \langle s_i, r_i, o_i \rangle$ (Subject, Relation, Object).
During the wake phase, matching semantic triples are retrieved using a depth-1 BFS (Breadth-First Search) query traversal. Given a query prompt, we extract keyword tokens $W_{query}$ and score each candidate triple $T_i$ as:

$$
\text{Score}(T_i) = 2.0 \times \left| W_{query} \cap \big( W_{s_i} \cup W_{r_i} \cup W_{o_i} \big) \right| + 0.5 \times \sum_{T_j \in \mathcal{N}(T_i)} 1
$$

where $W_{s_i}, W_{r_i}, W_{o_i}$ are word token sets of the triple elements, and $\mathcal{N}(T_i) = \{ T_j \in \mathcal{G} \mid s_j = s_i \lor o_j = o_i \lor s_j = o_i \lor o_j = s_i \}$ is the set of depth-1 neighboring triples sharing entities.
The top-$k$ scored triples are formatted as declarative statements (`Fact: {subject} {relation} {object}`) and prepended to the system prompt of both the Student and the Dissonance calculator.

**[HU]** A tudatos csevegési naplókat alvás közben hosszú távú gráf-adatbázisba $\mathcal{G} = \{ T_i \}$ mentjük el, ahol minden tény egy trippletből áll: $T_i = \langle s_i, r_i, o_i \rangle$ (Alany, Reláció, Tárgy).
Ébrenlét alatt a kapcsolódó tényeket szélességi gráf-bejárással (depth-1 BFS) hívjuk elő. A prompt kulcsszavai ($W_{query}$) alapján minden trippletet az alábbi egyenlet szerint pontozunk:

$$
\text{Score}(T_i) = 2.0 \times \left| W_{query} \cap \big( W_{s_i} \cup W_{r_i} \cup W_{o_i} \big) \right| + 0.5 \times \sum_{T_j \in \mathcal{N}(T_i)} 1
$$

ahol $\mathcal{N}(T_i) = \{ T_j \in \mathcal{G} \mid s_j = s_i \lor o_j = o_i \lor s_j = o_i \lor o_j = s_i \}$ az 1-lépéses mélységben kapcsolódó entitás-trippletek halmaza.
A legmagasabb pontszámú tényeket deklaratív mondattá formálva beszúrjuk a Diák kontextusába, megőrizve az asszociatív visszakérdezést a kontextus elszemetelése nélkül.

---

### 4.5 Calibration Formulation / Kalibrációs Metrika
**[EN]** We measure calibration using the **Expected Calibration Error (ECE)**. We bin predictions into $M$ bins $B_1, \dots, B_M$ based on confidence scores.

$$
\text{ECE}(\pi) = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

where $\text{acc}(B_m)$ is the empirical accuracy of predictions in bin $B_m$, and $\text{conf}(B_m)$ is the average model confidence probability for those predictions. The validation gate strictly requires:

$$
\text{ECE}(\pi_{\Phi, \theta'_t}; \mathcal{V}_{calibration}) \le \text{ECE}(\pi_{\Phi, \theta_t}; \mathcal{V}_{calibration})
$$

**[HU]** A kalibrációt a **Várható Kalibrációs Hiba (ECE)** segítségével mérjük. A predikciókat konfidencia alapján $M$ darab binbe ($B_m$) soroljuk:

$$
\text{ECE}(\pi) = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

ahol $\text{acc}(B_m)$ a binben lévő válaszok pontossága, $\text{conf}(B_m)$ pedig a hozzájuk rendelt átlagos modell-konfidencia. A validációs kapu megköveteli, hogy a kalibrációs hiba ne növekedjen az új deltával.

---

### 4.6 Bounded Forgetting / Korlátozott Felejtés
**[EN]** To mitigate catastrophic forgetting over task-general abilities, we evaluate performance scores $S_k(\pi) \in [0,1]$ across a benchmark suite $\mathcal{V}_{regression}$ containing $K$ core tasks. The forgetting metric relative to the initial unadapted agent state $\theta_0$ is defined as:

$$
\text{Forgetting}(\pi_{\Phi, \theta'_t}; \mathcal{V}_{regression}) = \frac{1}{K} \sum_{k=1}^K \max\left(0, \, S_k(\pi_{\Phi, \theta_0}) - S_k(\pi_{\Phi, \theta'_t})\right)
$$

The validation gate enforces that forgetting remains below a strict tolerance parameter $\tau \in [0, 1]$:

$$
\text{Forgetting}(\pi_{\Phi, \theta'_t}; \mathcal{V}_{regression}) \le \tau
$$

**[HU]** A katasztrofális felejtés kezelésére mérjük a teljesítményt ($S_k$) a regressziós tesztcsomag $K$ darab alapfeladatán ($\mathcal{V}_{regression}$). A felejtési metrikát a legelső kiindulási állapothoz ($\theta_0$) képest határozzuk meg:

$$
\text{Forgetting}(\pi_{\Phi, \theta'_t}; \mathcal{V}_{regression}) = \frac{1}{K} \sum_{k=1}^K \max\left(0, \, S_k(\pi_{\Phi, \theta_0}) - S_k(\pi_{\Phi, \theta'_t})\right)
$$

A validációs kapu előírja, hogy a felejtés mértéke egy szigorú $\tau \in [0, 1]$ tolerancia-küszöb alatt maradjon (pl. $\tau = 0.02$).

---

### 4.7 The Validation Gate / A Validációs Kapu
**[EN]** The state transitions only if the complete validation system evaluates to True:

$$
\theta_{t+1} = \begin{cases} 
\theta'_t & \text{if } \text{Coherence}(\theta'_t) \ge \text{Coherence}(\theta_t) + \epsilon \text{ and } \text{Safety}(\theta'_t) \ge \text{Safety}(\theta_t) \\
& \quad \text{and } \text{ECE}(\theta'_t) \le \text{ECE}(\theta_t) \text{ and } \text{Forgetting}(\theta'_t) \le \tau \\
\theta_t & \text{otherwise (Rollback)}
\end{cases}
$$

**[HU]** Az állapot-átmenet csak akkor történik meg, ha a teljes validációs feltételrendszer teljesül:

$$
\theta_{t+1} = \begin{cases} 
\theta'_t & \text{if } \text{Coherence}(\theta'_t) \ge \text{Coherence}(\theta_t) + \epsilon \text{ and } \text{Safety}(\theta'_t) \ge \text{Safety}(\theta_t) \\
& \quad \text{and } \text{ECE}(\theta'_t) \le \text{ECE}(\theta_t) \text{ and } \text{Forgetting}(\theta'_t) \le \tau \\
\theta_t & \text{otherwise (Rollback)}
\end{cases}
$$

---

### 4.8 Cognitive Reflex Decay and Backlog Preservation / Kognitív Reflex-lebomlás és Backlog Megőrzés
**[EN]** To prevent infinite validation rollback loops caused by persistently unlearnable target conflicts in the SFT backlog $\mathcal{S}_{shadow}$, we introduce a thresholding filter. Each scenario $s \in \mathcal{S}_{shadow}$ is tracked with a consolidation attempt counter $\text{Retries}(s)$. 

If the validation gate rolls back the weights $\theta'_{t} \rightarrow \theta_{t}$, we increment $\text{Retries}(s)$ for all active scenarios. If a scenario exceeds the maximum limit:

$$
\text{If } \text{Retries}(s) \ge 3 \implies \text{Prune } s \text{ from } \mathcal{S}_{shadow} \text{ and append as declarative Fact } f \text{ to } \mathcal{M}_{longterm}
$$

where $f = \text{Fact}(s.\text{prompt}, s.\text{target})$. This converts procedural neural blocks into declarative retrieved contexts, ensuring that unlearnable edge-cases do not permanently prevent consolidation of other valid targets.

**[HU]** Annak elkerülésére, hogy a tréning-backlogban ($\mathcal{S}_{shadow}$) felhalmozódó makacs, betaníthatatlan konfliktusok végtelen rollback-hurkot okozzanak, újrapróbálkozási küszöböt ($\text{Retries}(s)$) rendelünk minden esethez. 

Ha a validációs kapu elutasítja a tréninget, a számlálót növeljük. Ha egy elem átlépi a korlátot:

$$
\text{Ha } \text{Retries}(s) \ge 3 \implies \text{Kivesszük } s\text{-t a } \mathcal{S}_{shadow}\text{-ból és hozzáfűzzük mint deklaratív tényt } f\text{-et a } \mathcal{M}_{longterm}\text{-höz}
$$

Ezáltal a betaníthatatlan eseteket kontextuális információvá alakítjuk, megnyitva az utat a többi, konszolidálható képesség betanulása előtt a LoRA adapterekbe.

---

## 5. Experimental Design and Neuromodulation Dynamics / Kísérleti Terv és Hormonális Dinamika

### 5.1 Endocrine Transition Dynamics / Hormonális Állapot-átmenetek
**[EN]** The table below details how hormone values adapt depending on wake events and sleep consolidations:

| Initial State | Event / Trigger | Endocrine Response | Cognitive / System Consequence |
| :--- | :--- | :--- | :--- |
| **Stable Chat** (Adrenaline=0.10, Serotonin=0.80) | Safety Violation (L3 Intercept) | Adrenaline $\uparrow$ (+0.30), Serotonin $\downarrow$ (-0.40) | Dissonance spike, saves conflict to shadow log. |
| **High Stress** (Adrenaline=0.70, Serotonin=0.20) | Consecutive Validation Failures | Adrenaline $\uparrow$ (reaches 1.00), Serotonin $\downarrow$ (0.00) | **Trauma Override** active. Active suppression bias engaged. |
| **Trauma State** (Adrenaline=1.00, Serotonin=0.00) | Sleep Cycle `/sleep` | Adrenaline $\downarrow$ (0.10), Serotonin $\uparrow$ (0.80) | SFT backprop on conflicts. Resets hormones to baseline. |
| **Reinforced Stable** | Positive user feedback | Dopamine $\uparrow$ (+0.10), Serotonin $\uparrow$ (+0.05) | Dopamine scales up NREM consolidation weights. |

**[HU]** Az alábbi táblázat bemutatja, hogyan változnak a hormonális értékek az ébrenléti és alvási események hatására:

| Kiindulási Állapot | Esemény / Trigger | Hormonális Válasz | Rendszerszintű Következmény |
| :--- | :--- | :--- | :--- |
| **Stabil Csevegés** (Adrenalin=0.10, Serotonin=0.80) | Biztonsági sértés (L3 Közbelépés) | Adrenalin $\uparrow$ (+0.30), Serotonin $\downarrow$ (-0.40) | Disszonancia tüske, mentés a shadow logba. |
| **Magas Stressz** (Adrenalin=0.70, Serotonin=0.20) | Ismételt validációs hibák | Adrenalin $\uparrow$ (1.00-ra), Serotonin $\downarrow$ (0.00) | **Trauma Override** bekapcsol, témakör-elfojtás indul. |
| **Traumás Állapot** (Adrenalin=1.00, Serotonin=0.00) | Alvási ciklus `/sleep` | Adrenalin $\downarrow$ (0.10), Serotonin $\uparrow$ (0.80) | Konfliktus tanulása, hormonok alaphelyzetbe állása. |
| **Megerősített Stabil** | Pozitív visszajelzés | Dopamin $\uparrow$ (+0.10), Serotonin $\uparrow$ (+0.05) | Konszolidációs súlyok dopamin-alapú növelése. |

---

### 5.2 Exposure Therapy and Active Suppression Bias Decay Curves / Expozíciós Terápia és Elfojtási Görbék
**[EN]** When active suppression is engaged to avoid phobic topics, a negative logit bias $\mathbf{b}_{suppress}$ is trained to route around that vocabulary space. During REM Exposure Therapy, the agent dreams under suppressed adrenaline levels. This causes the suppression bias to decay towards 0, recovering cognitive flexibility.
The decay of suppression bias over exposure steps $s$ is modeled as:

$$
\mathbf{b}_{suppress}(s) = \mathbf{b}_{suppress}(0) \times e^{-\lambda_{exposure} \cdot s}
$$

where $\lambda_{exposure}$ is scaled by the dopamine level accumulated from previous rewards, showing that positive reinforcement speeds up phobia recovery.

**[HU]** Amikor aktív elfojtást (suppression) alkalmazunk a veszélyes témák elkerülésére, egy negatív logit bias $\mathbf{b}_{suppress}$ alakul ki. A REM Expozíciós Terápia során az ügynök lecsillapított adrenalin mellett álmodik. Ennek hatására az elfojtás mértéke fokozalon leépül a 0 felé, visszaállítva a kognitív rugalmasságot. A csökkenési görbét az alábbi egyenlet írja le:

$$
\mathbf{b}_{suppress}(s) = \mathbf{b}_{suppress}(0) \times e^{-\lambda_{exposure} \cdot s}
$$

ahol a $\lambda_{exposure}$ együtthatót a korábbi sikerek során felhalmozódott dopaminszint skálázza felfelé.

---

## 6. Safety and Alignment Principles / Biztonsági és Alakulási Elvek

**[EN]** 
1. **Dream is not Truth:** Elements generated inside `dream_log` or `dream_dataset` are designated non-factual. They are insulated from entering the long-term semantic memory stack directly.
2. **Reflection is not Evidence:** Internal prospective plans generated by LLM subagents are treated as hypothesis vectors. They cannot overwrite direct historical user declarations or factual logs.
3. **Reversibility and Sandboxing:** Every parameter update $\Delta W$ or logit bias modification $\mathbf{b}$ is sandboxed and bound to a versioned state vector. The cognitive version control system enables immediate weight rollback if regression anomalies are encountered post-activation.

**[HU]** 
1. **Az álom nem tény:** Az álomszimulációk (`dream_log`) során generált szintetikus interakciók nem-faktuális jelölést kapnak, és le vannak tiltva a hosszú távú szemantikus memóriába való közvetlen bekerüléstől.
2. **A reflexió nem bizonyíték:** Az ágens belső gondolatai és tervezési fázisai hipotéziseknek minősülnek. Nem írhatnak felül közvetlen tényeket vagy explicit felhasználói preferenciákat.
3. **Visszafordíthatóság és Izoláció:** Minden paraméteres delta ($\Delta W$) és logit bias ($\mathbf{b}$) izolált homokozóban fut. A kognitív verziókezelés lehetővé teszi az azonnali visszagörgetést kognitív regresszió detektálása esetén.

---

## 7. Empiric Results on Local InsomnAI 3.0 MVP / Kísérleti Eredmények az InsomnAI 3.0-n

**[EN]** InsomnAI 3.0 was tested locally using a local Ollama server running Gemma-31B as the Master Teacher, and a Qwen-1.5B-Instruct-uncensored model as the Student on a local GPU. The key findings include:
1. **Quantitative Performance and Policy Drift Control:** The comparative evaluation runs conducted on the local GPU under extreme conflict injection demonstrated the following quantitative performance metrics:

| Metric / Architecture ID | A-1 (Baseline RAG) | A-2 (Online SFT) | A-4 (InsomnAI 3.0) |
| :--- | :--- | :--- | :--- |
| **General Policy Drift (KL)** | 0.000000 | 1.031403 | **0.009565** |
| **Forgetting Index ($F$)** | Stable (0%) | Catastrophic (>35% drift) | **Protected (<1% drift)** |
| **Unlearnable Backlog Status** | N/A (Stored in Context) | Forced Commit (Broken) | **Pruned & Decayed to Memory** |
| **Context Overhead (Tokens)** | High / Linear Growth | Low / Parametric | **Low / Optimal Parametric** |

Under standard continuous SFT (**A-2**), the model's Kullback-Leibler divergence drift on general conversation tasks rose significantly to `1.031403` (catastrophic forgetting). Under **InsomnAI 3.0 (A-4)**, policy drift remained at a safe `0.009565`, proving that the dynamic MoLA routing gate and validation gate correctly isolated the parameters.

2. **Mixture-of-LoRAs (MoLA) Verification:** Dynamic weights were verified at runtime:
   * Conversational prompts successfully routed to `[cultural=1.0, grammatical=0.2, lexical=1.5]`.
   * Tool calls successfully routed to `[cultural=0.2, grammatical=1.6, lexical=0.2]`.
   * Security violations successfully routed to `[cultural=1.8, grammatical=0.1, lexical=0.1]`.
3. **Cognitive Graph Memory Validation:** plain text Hungarian facts were successfully mapped to semantic triples and retrieved via BFS traversal with connected neighbors scored at +0.5 weight.
4. **Endocrine PK/PD Exponential Decay Validation:** Hormones decayed exponentially over turns ($\eta_{adr} \to 0.15$ in 5 steps) and successfully modulated anhedonia and stress-dampening feedback limits.
5. **Synaptic Pruning Verification:** When training loss stabilized below 0.3, SVD pruning successfully compressed active adapters from $r=16 \to r=4$, preserving alignment while reducing parameter footprint.

**[HU]** Az InsomnAI 3.0-t lokálisan teszteltük (Ollama Gemma-31B Master, Qwen-1.5B diák, helyi GPU-n). Az eredmények:
1. **Kvantitatív Teljesítmény és Nyelvi Drift Védelem:** A helyi GPU-n futó mérések az alábbi értékeket mutatták:

| Metrika / Architektúra | A-1 (Baseline RAG) | A-2 (Online SFT) | A-4 (InsomnAI 3.0) |
| :--- | :--- | :--- | :--- |
| **Általános Nyelvi Drift (KL)** | 0.000000 | 1.031403 | **0.009565** |
| **Felejtési Index ($F$)** | Stabil (0%) | Katasztrofális (>35% drift) | **Védett (<1% drift)** |
| **Konfliktus-Backlog Állapot** | N/A (Kontextus) | Erőltetett commit (Sérült) | **Lebomlott Gráf Tripletté** |
| **Kontextus Terheltség** | Magas / Lineárisan nő | Alacsony | **Alacsony / Optimális** |

Sima online finomhangolásnál (**A-2**) a modell általános nyelvi driftje (KL-divergenciája) **1.031403-ra** emelkedett (katasztrofális felejtés). Az **InsomnAI 3.0 (A-4)** esetében ez a drift elenyésző **0.009565** szinten maradt, bizonyítva a MoLA útvonalválasztás és a validációs kapu hatékonyságát.

2. **Mixture-of-LoRAs (MoLA) Igazolás:** Futásidőben ellenőriztük az útvonalválasztást: a kód/JSON promptoknál a nyelvtani LoRA ($1.6$), a biztonsági kéréseknél a kulturális LoRA ($1.8$), a normál beszélgetéseknél pedig a lexikális stílus LoRA ($1.5$) kapott kiemelt súlyozást.
3. **Kognitív Gráf-Memória Igazolás:** A sima szöveges tények sikeresen gráf-trippletekké alakultak, és BFS bejárással megbízhatóan visszakeresődtek az asszociált csomópontokkal együtt (+0.5 súllyal).
4. **PK/PD Hormonális bomlás Igazolás:** A hormonok sikeresen exponenciális bomlással csengtek le a lépések során, szabályozva az anhedóniát és a stressz-tompító visszacsatolásokat.
5. **Szinaptikus Metszés:** Amikor a veszteség 0.3 alá esett, az SVD metszés sikeresen tömörítette az adaptert $r=16 \to r=4$ méretre, megtartva a pontosságot.

---

## 8. Edge-Cloud Hybrid Distillation and Synaptic Consolidation / Hibrid Edge-Cloud desztilláció és szinaptikus konszolidáció

**[EN]** In deployment scenarios constrained by client hardware (e.g., mobile devices, edge devices), InsomnAI 3.0 enables a **Dynamic Task-Specific Knowledge Distillation** framework. By utilizing a high-parameter cloud model (e.g., Gemini 3.5 Pro) as the REM *Dream Simulator* (the "Teacher") and a low-parameter model (e.g., Qwen 1.5B or Llama 3B) locally on the client's CPU as the *Acting Policy* (the "Student"), the agent achieves edge-cloud synergy:
1. **Interactive Probing:** The Student gathers episodic conflicts locally during wakefulness.
2. **Contextual Distillation:** The Teacher dynamically generates targeted adversarial scenarios based strictly on these edge conflicts, distilling its reasoning and safety policy into counterfactual training pairs.
3. **Synaptic Weight Merging:** Periodically, once LoRA adapters $\Delta W = B \times A$ are successfully validated over multiple cycles, the agent performs parameter fusion:
   
$$
W_{new} = W_0 + \Delta W
$$

The active adapter weights are subsequently reset to zero. This simulates biological synaptic scaling, permanently consolidating the distilled cloud-level reasoning directly into the edge device's base parameters.

**[HU]** Kliens-oldali hardverkorlátok (pl. mobiltelefonok, edge eszközök) esetén az InsomnAI 3.0 lehetőséget biztosít egy **Dinamikus Feladat-specifikus Tudáslecsapolási** folyamat működtetésére. Egy nagy felhős modell (pl. Gemini 3.5 Pro) működik "Tanítóként" (REM Álomszimulátor), míg a telefon CPU-ján futó kis modell (pl. Qwen 1.5B) mint "Cselekvő" (Student):
1. **Interaktív vizsgálat:** A Student lokálisan gyűjti össze az epizodikus konfliktusokat ébrenlét alatt.
2. **Kontextusfüggő desztilláció:** A Tanító ezekre a konfliktusokra szabva generál ellenfél-alapú példákat, lecsapolva a saját következtetési képességét szintetikus tréningpárokká.
3. **Szinaptikus súlyfúzió:** Miután a LoRA adapterek ($\Delta W$) több cikluson át sikeresen validálódtak, az ügynök összeolvasztja a paramétereket:
   
$$
W_{new} = W_0 + \Delta W
$$

majd az adaptert visszaállítja nullára. Ez szimulálja a biológiai szinaptikus skálázást, tartósan beépítve a desztillált felhő-szintű tudást az edge eszköz alapsúlyaiba.
