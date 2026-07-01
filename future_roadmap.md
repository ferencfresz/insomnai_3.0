# InsomnAI 3.0 & 4.0: Next-Generation Cognitive Architecture Roadmap
## InsomnAI 3.0 és 4.0: Következő Generációs Kognitív Architektúra Roadmap

This document outlines the architectural enhancements and future research directions designed to transition **InsomnAI 3.0** (the current local neuromodulated engine) into a highly scalable, collaborative, and biologically aligned cognitive agent swarm ecosystem (**InsomnAI 4.0**).

---

## 1. Cognitive Graph Memory / Kognitív Gráf-Memória

**[EN]** Flat token-intersection keyword retrievers are linear and fail to capture non-literal conceptual relationships.
* **InsomnAI 3.0 Implementation:** The semantic and episodic memory is represented as a local SQLite/JSON Graph Database (Triple Store).
* **Mechanics:**
  1. The Master Teacher extracts semantic relation triples `(Subject, Relation, Object)` from episodic logs (e.g., `(User, Likes, Red Apples)`).
  2. During the Wake phase, retrieval is performed using a depth-1 BFS query traversal to retrieve logically associated entity neighbors.
* **Impact:** Prevents context window bloat while dramatically increasing the relevance and depth of factual association.

**[HU]** A lapos token-metszet alapú keresők lineárisak, és nem képesek a nem-szó szerinti fogalmi összefüggések feloldására.
* **InsomnAI 3.0 Implementáció:** Az epizodikus és szemantikus memóriát helyi gráf-adatbázisként (Triple Store) tároljuk.
* **Működés:**
  1. A Master Teacher szemantikai tripleteket emel ki a naplókból: `(Alany, Reláció, Tárgy)`, például `(Felhasználó, Kedvel, Piros Alma)`.
  2. Ébrenlét alatt a visszakeresés szélességi gráf-bejárási algoritmussal (depth-1 BFS) történik az asszociatív tények felkutatására.
* **Hatás:** Megszünteti a kontextus elszemetelését, miközben nagyságrendekkel növeli az előhívott tények relevanciáját.

---

## 2. Mixture-of-LoRAs (MoLA) Dynamic Routing / MoLA Dinamikus Útvonalválasztás

**[EN]** Merging LoRA adapters (`cultural`, `grammatical`, `lexical`) using static, equal weights (1.0) leads to suboptimal performance across specialized tasks.
* **InsomnAI 3.0 Implementation:** A **Mixture-of-LoRAs (MoLA)** dynamic routing gate.
* **Mechanics:**
  1. A prompt intent classifier evaluates the user prompt in real-time.
  2. It dynamically compiles and merges the adapters at runtime using task-appropriate weights (e.g., `grammatical=1.6, cultural=0.2, lexical=0.2` for JSON coding).
* **Impact:** Enables continuous, context-aware behavioral blending and style modulation at inference time.

**[HU]** A LoRA adapterek (`cultural`, `grammatical`, `lexical`) statikus, egyenlő ($1.0$) súlyokkal történő összefésülése lerontja a speciális feladatok minőségét.
* **InsomnAI 3.0 Implementáció:** Egy **Mixture-of-LoRAs (MoLA)** dinamikus kapuzó réteg.
* **Működés:**
  1. Egy prompt osztályozó valós időben elemzi a bejövő kéréseket.
  2. A kontextus alapján futásidőben dinamikusan fűzi össze az adaptereket egyedi súlyozással (pl. kódoláskor: `grammatical=1.6, cultural=0.2, lexical=0.2`).
* **Hatás:** Lehetővé teszi a stílusok és szabályok finom, kontextus-függő futásidejű keverését.

---

## 3. PK/PD Pharmacokinetic Endocrine Feedback / Farmakokinetikai Hormonális Visszacsatolás

**[EN]** Linear scaling of hormone levels resets abruptly at cycle boundaries, leading to jagged cognitive state transitions.
* **InsomnAI 3.0 Implementation:** Model simulated hormones using biological pharmacokinetic/pharmacodynamic (PK/PD) first-order clearance equations.
* **Mechanics:**
  1. Assign unique decay clearance parameters (e.g. Adrenaline $\eta_{adr}$ clears rapidly with $\lambda_{adr}=0.5$, while Serotonin $\eta_{ser}$ replenishes slowly with $\lambda_{ser}=0.1$).
  2. Simulate inter-hormonal feedback loops (e.g. high serotonin dampens adrenaline spikes; high adrenaline dampens dopamine rewards representing stress-induced anhedonia).
* **Impact:** Imparts a smooth cognitive temperament, allowing the agent to transition naturally through states (e.g. remaining in a defensive or stressed state for several turns post-conflict).

**[HU]** A hormonok lineáris változása és hirtelen nullázódása darabos kognitív állapot-átmeneteket eredményezett.
* **InsomnAI 3.0 Implementáció:** A hormonok változásának leírása elsőrendű farmakokinetikai (PK/PD) bomlási egyenletekkel.
* **Működés:**
  1. Minden hormonhoz egyedi lebomlási együtthatót rendelünk (pl. az Adrenalin $\eta_{adr}$ gyorsan lecseng $\lambda_{adr}=0.5$, míg a Szerotonin $\eta_{ser}$ lassan épül fel újra $\lambda_{ser}=0.1$).
  2. Kereszt-hormonális visszacsatolásokat szimulálunk (pl. a magas szerotonin tompítja az adrenalin stressz-tüskéket; a magas adrenalin elfojtja a dopamin jutalom-skálázást).
* **Hatás:** Természetes, lágy kognitív temperamentumot kölcsönöz az ágensnek (pl. a konfliktusok után még néhány körig defenzív/stresszelt üzemmódban marad).

---

## 4. Federated Sleep Swarm Consolidation / Föderált Alvásciklusos Tanulás

**[EN]** Client agents currently consolidate and train their weights in complete isolation, preventing community knowledge sharing.
* **InsomnAI 4.0 Upgrade:** Implement privacy-preserving **Federated Neuromodulated Learning**.
* **Mechanics:**
  1. Local edge nodes perform SFT sleep consolidation cycles locally over their private logs (retaining 100% PII privacy).
  2. Once the local Validation Gate successfully commits the weights, the agent sends *only* the validated LoRA weight deltas ($\Delta W$) to a secure central server.
  3. The server consolidates updates using Federated Averaging (FedAvg) and redistributes the updated base weights back to the edge swarm.
* **Impact:** Creates a collaborative swarm intelligence that learns from collective edge experiences without ever centralizing private user logs.

**[HU]** A kliens ágensek jelenleg teljes elszigeteltségben tanulnak, ami megakadályozza a megszerzett képességek megosztását.
* **InsomnAI 4.0 Fejlesztés:** Adatvédelmet biztosító **Föderált Neuromodulált Tanulás** megvalósítása.
* **Működés:**
  1. A helyi eszközök elvégzik a Sleep SFT ciklust a saját gépükön (megőrizve a személyes adatok 100%-os biztonságát).
  2. A sikeres helyi validáció után az ágens *kizárólag* a LoRA súly-deltákat ($\Delta W$) küldi be a központi szerverre.
  3. A szerver a beküldött súlyokat a FedAvg algoritmussal összesíti, és visszadistributálja a frissített alapsúlyokat a kliens-rajnak.
* **Hatás:** Létrehoz egy kollaboratív raj-intelligenciát, amely a tagok kollektív tapasztalataiból tanul a privát csevegési naplók centralizálása nélkül.
