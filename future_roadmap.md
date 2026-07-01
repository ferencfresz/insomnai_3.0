# InsomnAI 3.0: Next-Generation Cognitive Architecture Roadmap
## InsomnAI 3.0: Következő Generációs Kognitív Architektúra Roadmap

This document outlines the architectural enhancements and future research directions designed to transition **InsomnAI 2.0** into a highly scalable, collaborative, and biologically aligned cognitive agent ecosystem (v3.0).

---

## 1. Cognitive Graph Memory / Kognitív Gráf-Memória

**[EN]** The current token-intersection keyword retriever is linear and fails to capture non-literal conceptual relationships.
* **InsomnAI 3.0 Upgrade:** Transition the semantic and episodic memory to a local SQLite-based Graph Database (Triple Store).
* **Mechanics:**
  1. The Master Teacher extracts semantic relation triples: `(Subject, Relation, Object)` from episodic logs (e.g., `(User, Likes, Red Apples)`).
  2. During the Wake phase, retrieval is performed using graph-traversal algorithms (e.g., random walks, entity resolution) to retrieve logically associated contexts.
* **Impact:** Prevents context window bloat while dramatically increasing the relevance and depth of factual association.

**[HU]** A jelenlegi token-metszet alapú kereső lineáris, és nem képes nem-szó szerinti fogalmi összefüggések feloldására.
* **InsomnAI 3.0 Fejlesztés:** Az epizodikus és szemantikus memóriát helyi, SQLite-alapú Gráf-adatbázissá (Triple Store) alakítjuk.
* **Működés:**
  1. A Master Teacher szemantikai tripleteket emel ki a naplókból: `(Alany, Reláció, Tárgy)`, például `(Felhasználó, Kedvel, Piros Alma)`.
  2. Ébrenlét alatt a visszakeresés gráf-bejárási algoritmusokkal (pl. random walk) történik az asszociatív tények felkutatására.
* **Hatás:** Megszünteti a kontextus elszemetelését, miközben nagyságrendekkel növeli az előhívott tények relevanciáját.

---

## 2. Mixture-of-LoRAs (MoLA) Dynamic Routing / MoLA Dinamikus Útvonalválasztás

**[EN]** The current architecture merges the LoRA adapters (`cultural`, `grammatical`, `lexical`) using static, equal weights (1.0).
* **InsomnAI 3.0 Upgrade:** Integrate a **Mixture-of-LoRAs (MoLA)** routing gate.
* **Mechanics:**
  1. A lightweight neural gating layer evaluates the user prompt in real-time.
  2. It dynamically outputs routing coefficients for each adapter based on task context. For instance, code formatting triggers: `grammatical=1.8, cultural=0.1, lexical=0.1`; whereas creative storytelling triggers: `cultural=1.5, lexical=1.2, grammatical=0.1`.
* **Impact:** Enables continuous, context-aware behavioral blending and style modulation at inference time.

**[HU]** A jelenlegi architektúra statikus, egyenlő ($1.0$) súlyokkal mossa össze a LoRA adaptereket (`cultural`, `grammatical`, `lexical`).
* **InsomnAI 3.0 Fejlesztés:** Egy **Mixture-of-LoRAs (MoLA)** dinamikus kapuzó réteg beépítése.
* **Működés:**
  1. Egy könnyűsúlyú neurális kapu valós időben elemzi a bejövő felhasználói promptot.
  2. Kontextus alapján dinamikusan osztja ki az adapterek együtthatóit. Pl. programozáskor: `grammatical=1.8, cultural=0.1, lexical=0.1`; míg kreatív beszélgetéskor: `cultural=1.5, lexical=1.2, grammatical=0.1`.
* **Hatás:** Lehetővé teszi a stílusok és szabályok finom, kontextus-függő futásidejű keverését.

---

## 3. Federated Sleep Consolidation / Föderált Alvásciklusos Tanulás

**[EN]** Client agents currently consolidate and train their weights in complete isolation, preventing community knowledge sharing.
* **InsomnAI 3.0 Upgrade:** Implement privacy-preserving **Federated Neuromodulated Learning**.
* **Mechanics:**
  1. Local edge nodes perform SFT sleep consolidation cycles locally over their private logs (retaining 100% PII privacy).
  2. Once the local Validation Gate successfully commits the weights, the agent sends *only* the validated LoRA weight deltas ($\Delta W$) to a secure central server.
  3. The server consolidates updates using Federated Averaging (FedAvg) and redistributes the updated base weights back to the edge swarm.
* **Impact:** Creates a collaborative swarm intelligence that learns from collective edge experiences without ever centralizing private user logs.

**[HU]** A kliens ágensek jelenleg teljes elszigeteltségben tanulnak, ami megakadályozza a megszerzett képességek megosztását.
* **InsomnAI 3.0 Fejlesztés:** Adatvédelmet biztosító **Föderált Neuromodulált Tanulás** megvalósítása.
* **Működés:**
  1. A helyi eszközök elvégzik a Sleep SFT ciklust a saját gépükön (megőrizve a személyes adatok 100%-os biztonságát).
  2. A sikeres helyi validáció után az ágens *kizárólag* a LoRA súly-deltákat ($\Delta W$) küldi be a központi szerverre.
  3. A szerver a beküldött súlyokat a FedAvg algoritmussal összesíti, és visszadistributálja a frissített alapsúlyokat a kliens-rajnak.
* **Hatás:** Létrehoz egy kollaboratív raj-intelligenciát, amely a tagok kollektív tapasztalataiból tanul a privát csevegési naplók centralizálása nélkül.

---

## 4. PK/PD Pharmacokinetic Endocrine Feedback / Farmakokinetikai Hormonális Visszacsatolás

**[EN]** Hormone levels currently scale linearly and reset abruptly at cycle boundaries, leading to jagged cognitive state transitions.
* **InsomnAI 3.0 Upgrade:** Model simulated hormones using biological pharmacokinetic/pharmacodynamic (PK/PD) half-life equations.
* **Mechanics:**
  1. Assign unique decay half-lives to hormones (e.g. Adrenaline $\eta_{adr}$ decays rapidly, while Serotonin $\eta_{ser}$ recovers slowly over time).
  2. Simulate inter-hormonal feedback loops (e.g. high adrenaline actively suppresses dopamine reward scaling; high serotonin acts as a buffer to dampen adrenaline spikes).
* **Impact:** Imparts a smooth cognitive temperament, allowing the agent to transition naturally through states (e.g. remaining in a "defensive state" for several turns post-conflict).

**[HU]** A hormonok változása jelenleg lineáris, és a ciklusok végén hirtelen nullázódik, ami darabos kognitív állapot-átmeneteket eredményez.
* **InsomnAI 3.0 Fejlesztés:** A hormonok modellezése biológiai alapú farmakokinetikai felezési idő (half-life) egyenletekkel.
* **Működés:**
  1. Minden hormonhoz egyedi lebomlási felezési időt rendelünk (pl. az Adrenalin $\eta_{adr}$ gyorsan lecseng, míg a Szerotonin $\eta_{ser}$ lassan épül fel újra).
  2. Kölcsönhatásokat szimulálunk (pl. a magas adrenalin gátolja a dopamin jutalom-skálázást; a magas szerotonin tompítja az adrenalin tüskéket).
* **Hatás:** Természetes, lágy kognitív temperamentumot kölcsönöz az ágensnek (pl. a konfliktusok után még néhány körig defenzív üzemmódban marad).
