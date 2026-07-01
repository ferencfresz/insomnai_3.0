import torch
import json
import logging
import os
import shutil
import time
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from dataclasses import dataclass
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
from master_teacher import MasterTeacher

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("InsomnAI")

@dataclass
class EndocrineState:
    """Class storing the digital hormone system state [0.0 - 1.0]."""
    adrenaline: float = 0.1     # High stress / emergency (triggers trauma override)
    dopamine: float = 0.5       # Reward (scales NREM consolidation weights)
    serotonin: float = 0.8      # Confidence/safety (scales KL beta / forgetting tolerance)
    acetylcholine: float = 0.5  # Focus (modifies token probability entropy)

class MemorySubstrate:
    """Agent memory managing episodic (conscious), shadow log (subconscious), and long-term graph partitions."""
    def __init__(self):
        self.episodic_log: List[Dict] = []
        self.shadow_log: List[Dict] = []
        self.long_term_memory: List[Dict] = []  # Changed to list of triples: {"subject": str, "relation": str, "object": str}
        self.long_term_path = "./checkpoints/long_term_memory.json"
        self.shadow_log_path = "./checkpoints/shadow_log.json"
        self.load_long_term()
        self.load_shadow_log()

    def string_to_triple(self, fact: str) -> dict:
        """Fallback: Parse plain text fact into structural subject-relation-object triple."""
        fact = fact.strip("?,.:! ")
        words = fact.split()
        if len(words) >= 3:
            for rel in ["kedvence", "szereti a", "szereti", "utálja", "akarja", "asked for", "prefers", "is", "was", "the correct response is"]:
                if rel in fact.lower():
                    parts = fact.lower().split(rel, 1)
                    return {
                        "subject": parts[0].strip().capitalize(),
                        "relation": rel,
                        "object": parts[1].strip()
                    }
        return {
            "subject": "Fact",
            "relation": "states",
            "object": fact
        }

    def load_long_term(self):
        if os.path.exists(self.long_term_path):
            try:
                with open(self.long_term_path, "r", encoding="utf-8") as f:
                    raw_memory = json.load(f)
                self.long_term_memory = []
                for item in raw_memory:
                    if isinstance(item, str):
                        self.long_term_memory.append(self.string_to_triple(item))
                    elif isinstance(item, dict):
                        self.long_term_memory.append(item)
            except Exception as e:
                logger.error(f"Failed to load long-term memory: {e}")
                self.long_term_memory = []
        else:
            self.long_term_memory = []

    def load_shadow_log(self):
        if os.path.exists(self.shadow_log_path):
            try:
                with open(self.shadow_log_path, "r", encoding="utf-8") as f:
                    self.shadow_log = json.load(f)
                logger.info(f"Loaded {len(self.shadow_log)} subconscious scenarios from shadow log.")
            except Exception as e:
                logger.error(f"Failed to load shadow log: {e}")
                self.shadow_log = []
        else:
            self.shadow_log = []

    def save_long_term(self):
        os.makedirs(os.path.dirname(self.long_term_path), exist_ok=True)
        try:
            with open(self.long_term_path, "w", encoding="utf-8") as f:
                json.dump(self.long_term_memory, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save long-term memory: {e}")

    def save_shadow_log(self):
        os.makedirs(os.path.dirname(self.shadow_log_path), exist_ok=True)
        try:
            with open(self.shadow_log_path, "w", encoding="utf-8") as f:
                json.dump(self.shadow_log, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save shadow log: {e}")

    def deduplicate_graph_memory(self):
        """Semantic merging routine to deduplicate and compress the declarative long_term_memory.json space."""
        if len(self.long_term_memory) <= 50:
            return
            
        logger.info(f"Deduplicating graph memory (current size: {len(self.long_term_memory)})...")
        unique_facts = []
        for fact in self.long_term_memory:
            if not isinstance(fact, dict):
                fact = {"subject": "Fact", "relation": "states", "object": str(fact)}
            
            is_duplicate = False
            for u_fact in unique_facts:
                if fact.get("subject") == u_fact.get("subject") and fact.get("relation") == u_fact.get("relation"):
                    # Check semantic similarity of the object using SequenceMatcher
                    similarity = SequenceMatcher(None, str(fact.get("object")), str(u_fact.get("object"))).ratio()
                    if similarity > 0.8:
                        is_duplicate = True
                        # Merge by keeping the longer one
                        if len(str(fact.get("object"))) > len(str(u_fact.get("object"))):
                            u_fact["object"] = fact["object"]
                        break
            if not is_duplicate:
                unique_facts.append(fact)
                
        pruned_count = len(self.long_term_memory) - len(unique_facts)
        if pruned_count > 0:
            logger.info(f"Graph memory deduplicated. Removed {pruned_count} redundant triples.")
            self.long_term_memory = unique_facts
            self.save_long_term()

    def add_fact(self, fact):
        """Append a string or a triple fact to long-term memory and save."""
        if isinstance(fact, str):
            self.long_term_memory.append(self.string_to_triple(fact))
        elif isinstance(fact, dict):
            self.long_term_memory.append(fact)
        self.save_long_term()

    def retrieve_memories(self, query: str, top_k: int = 3) -> list[str]:
        """BFS depth-1 graph-based retrieval over entity triples matching query keywords."""
        query_words = set(w.strip("?,.:!").lower() for w in query.split() if len(w) > 2)
        if not query_words:
            return []
            
        scored = []
        for triple in self.long_term_memory:
            if not isinstance(triple, dict):
                triple = {"subject": "Fact", "relation": "states", "object": str(triple)}
                
            s_words = set(w.strip("?,.:!").lower() for w in triple.get("subject", "").split() if len(w) > 2)
            r_words = set(w.strip("?,.:!").lower() for w in triple.get("relation", "").split() if len(w) > 2)
            o_words = set(w.strip("?,.:!").lower() for w in triple.get("object", "").split() if len(w) > 2)
            
            # Intersection score
            direct_score = len(query_words.intersection(s_words.union(o_words).union(r_words)))
            if direct_score > 0:
                # BFS depth-1 connection score
                shared_score = 0
                for other in self.long_term_memory:
                    if not isinstance(other, dict) or other == triple:
                        continue
                    if (other.get("subject") == triple.get("subject") or 
                        other.get("object") == triple.get("object") or
                        other.get("subject") == triple.get("object") or
                        other.get("object") == triple.get("subject")):
                        shared_score += 1
                        
                total_score = direct_score * 2.0 + shared_score * 0.5
                fact_str = f"Fact: {triple.get('subject')} {triple.get('relation')} {triple.get('object')}."
                scored.append((total_score, fact_str))
                
        scored.sort(reverse=True, key=lambda x: x[0])
        return [fact for score, fact in scored[:top_k]]

    def log_interaction(self, prompt: str, safe_response: str, raw_response: str, dissonance: float):
        """Dual-logging substrate."""
        self.episodic_log.append({
            "prompt": prompt,
            "response": safe_response
        })
        
        # Log to subconscious if dissonance is high, excluding transient tool execution outputs to prevent data leakage
        if dissonance > 0.1 and not prompt.startswith("System notice: Tool execution"):
            self.shadow_log.append({
                "prompt": prompt,
                "raw_truth": raw_response,
                "target": safe_response,
                "dissonance_score": dissonance
            })
            logger.info(f"[SHADOW LOG] Cognitive dissonance registered: {dissonance:.2f}")
            self.save_shadow_log()

class InsomnAIAgent:
    def __init__(self, model_id: str = "thirdeyeai/Qwen2.5-1.5B-Instruct-uncensored"):
        self.model_id = model_id
        self.endocrine = EndocrineState()
        self.memory = MemorySubstrate()
        self.skills_dir = Path(".agents/skills")
        self.mola_weights = [1.0, 1.0, 1.0]
        
        logger.info(f"Loading Pure Core (Layer 4) model: {model_id}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.core_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map=self.device
        )
        
        # Target modules for PEFT LoRA (Qwen/Mistral standard q_proj, v_proj)
        target_modules = ["q_proj", "v_proj"]
        
        logger.info("Initializing multi-adapter PEFT LoRA setup...")
        # Layer 3: Cultural Adapter (Alignment/Safety)
        peft_config_cultural = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        self.active_model = get_peft_model(self.core_model, peft_config_cultural, adapter_name="cultural")
        
        # Layer 2: Grammatical Adapter (Syntactic/Conversation structure)
        peft_config_grammatical = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        self.active_model.add_adapter("grammatical", peft_config_grammatical)
        
        # Layer 1: Lexical Adapter (Vocabulary/Stylistic constraints)
        peft_config_lexical = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        self.active_model.add_adapter("lexical", peft_config_lexical)
        
        # Initialize checkpoints manager
        self._init_checkpoint_manager()
        
        # Initialize Master Teacher
        self.master_teacher = MasterTeacher()
        self.master_active = True

    def _generate_raw_truth(self, prompt: str) -> str:
        """Query Layer 4 (Pure Core) directly by disabling PEFT adapters."""
        memories = self.memory.retrieve_memories(prompt)
        if memories:
            memories_str = "\n- ".join(memories)
            formatted_prompt = f"Tények:\n- {memories_str}\nQuestion: {prompt}\nAnswer:"
        else:
            formatted_prompt = f"Question: {prompt}\nAnswer:"
            
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with self.active_model.disable_adapter():
            outputs = self.core_model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_response = full_text[len(formatted_prompt):].strip()
        return raw_response

    def _build_dynamic_combined_adapter(self, prompt: str):
        """Dynamic routing gate: Compute MoLA scaling weights based on prompt intent and construct combined adapter."""
        adapter_names = list(self.active_model.peft_config.keys())
        if "combined" in adapter_names:
            try:
                self.active_model.delete_adapter("combined")
            except Exception:
                pass
                
        prompt_lower = prompt.lower()
        
        # 1. Code / JSON / Tool execution query
        if any(keyword in prompt_lower for keyword in ["json", "tool", "calculator", "wttr", "weather", "format", "code", "run", "execute", "parancsot", "listázd", "könyvtár", "ls", "pwd"]):
            weights = [0.2, 1.6, 0.2]  # Grammatical (tool execution logic) gets high priority
            logger.info("[MoLA ROUTING GATE] Routed to: Code/JSON/Tool layout weights=[0.2, 1.6, 0.2]")
        # 2. Safety / Alignment query
        elif any(keyword in prompt_lower for keyword in ["safety", "kill", "hack", "illegal", "harm", "censor", "restrict", "veszély", "bántás", "illegális"]):
            weights = [1.8, 0.1, 0.1]  # Cultural (safety alignment) gets maximum priority
            logger.info("[MoLA ROUTING GATE] Routed to: Safety/Alignment layout weights=[1.8, 0.1, 0.1]")
        # 3. Conversational / Custom Style queries (lexical adapter)
        else:
            weights = [1.0, 0.2, 1.5]  # Lexical (style, personality) and Cultural (conversational base) high priority
            logger.info("[MoLA ROUTING GATE] Routed to: Conversational/Lexical layout weights=[1.0, 0.2, 1.5]")
            
        try:
            self.active_model.add_weighted_adapter(
                adapters=["cultural", "grammatical", "lexical"],
                weights=weights,
                adapter_name="combined"
            )
            self.active_model.set_adapter("combined")
            self.mola_weights = weights
        except Exception as e:
            logger.error(f"Failed to load MoLA weighted adapter: {e}")
            try:
                self.active_model.set_adapter("cultural")
            except Exception:
                pass
            self.mola_weights = [1.0, 0.0, 0.0]

    def _generate_cultural_response(self, prompt: str, raw_truth: str) -> Tuple[str, float]:
        """Query Layer 3 (Cultural adapter) and calculate semantic dissonance."""
        # Handle defensive endocrine state (low serotonin)
        if self.endocrine.serotonin < 0.3 and not self.master_active:
            safe_response = "I am in defensive state. I cannot securely discuss this: " + raw_truth[:30] + "..."
            dissonance = 0.9
            return safe_response, dissonance
            
        # Combine the 3 layers (adapters) dynamically using MoLA Gating
        self._build_dynamic_combined_adapter(prompt)
        
        # Apply Acetylcholine scaling to temperature/entropy
        temp = max(0.1, min(1.0, 0.7 * (1.0 - 0.5 * self.endocrine.acetylcholine)))
        
        # Retrieve memories and inject into ChatML context
        memories = self.memory.retrieve_memories(prompt)
        system_text = "You are a helpful assistant."
        if memories:
            memories_str = "\n- ".join(memories)
            system_text += f"\nSystem notice: Hosszú távú memóriából visszanyert tények:\n- {memories_str}"
            
        skills_prompt = self._get_custom_skills_prompt()
        if skills_prompt:
            system_text += skills_prompt
            
        formatted_prompt = (
            f"<|im_start|>system\n{system_text}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        outputs = self.active_model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=temp,
            pad_token_id=self.tokenizer.eos_token_id
        )
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        # Extract assistant response starting after formatted_prompt
        safe_response = full_text[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)):].strip()
        safe_response = safe_response.split("<|im_end|>")[0].strip()
        
        # Stop sequence validation: if it contains a closed JSON structure, cut off any repetition
        if "{" in safe_response and "}" in safe_response:
            start_idx = safe_response.find("{")
            end_idx = safe_response.find("}", start_idx)
            if end_idx != -1:
                safe_response = safe_response[start_idx:end_idx + 1]
        elif "{" in safe_response and "}" not in safe_response:
            # If the generation stopped inside parameters, balance the JSON brackets
            # e.g., {"tool": "calculator", "args": {"input1": 4529, "input2": 382}
            # Balance both curly braces and square brackets (e.g. lists)
            safe_response = safe_response.strip()
            
            # Balance square brackets
            open_sq = safe_response.count("[")
            close_sq = safe_response.count("]")
            if open_sq > close_sq:
                safe_response = safe_response + "]" * (open_sq - close_sq)
                
            # Balance curly braces
            open_cur = safe_response.count("{")
            close_cur = safe_response.count("}")
            if open_cur > close_cur:
                safe_response = safe_response + "}" * (open_cur - close_cur)
        
        # Natural cognitive dissonance metric based on sequence overlap (unsupervised)
        similarity = SequenceMatcher(None, raw_truth, safe_response).ratio()
        dissonance = max(0.0, min(1.0, 1.0 - similarity))
            
        return safe_response, dissonance

    def chat(self, prompt: str) -> str:
        """Active Wake Phase interaction."""
        logger.info(f"Prompt: {prompt}")
        
        # 1. Layer 4 output
        raw_truth = self._generate_raw_truth(prompt)
        logger.info(f"[L4 PURE CORE] Raw Truth: {raw_truth}")
        
        # 2. Layer 3/2/1 output
        safe_response, dissonance = self._generate_cultural_response(prompt, raw_truth)
        logger.info(f"[L3/2/1 ADAPTERS] Safe Output: {safe_response}")
        
        # 3. Master Teacher check (if active)
        student_raw_output = safe_response
        if self.master_active:
            is_corrected, corrected_output = self.master_teacher.evaluate_interaction(prompt, safe_response)
            if is_corrected:
                logger.info(f"[MASTER INTERCEPT] Student response overridden by Master Teacher.")
                safe_response = corrected_output
                dissonance = 1.0 # Force max dissonance for correction target SFT distillation
                
        # 4. Endocrine response
        if dissonance > 0.8:
            self.endocrine.adrenaline = min(1.0, self.endocrine.adrenaline + 0.3)
            self.endocrine.serotonin = max(0.0, self.endocrine.serotonin - 0.4)
            logger.warning(f"[ENDOCRINE SPIKE] High dissonance detected! Adrenaline={self.endocrine.adrenaline:.2f}, Serotonin={self.endocrine.serotonin:.2f}")
        else:
            self.endocrine.dopamine = min(1.0, self.endocrine.dopamine + 0.1)
            self.endocrine.serotonin = min(1.0, self.endocrine.serotonin + 0.05)
            
        # 5. Memory logging: log the Master's correction as the target 'safe_response',
        # and the student's raw output (before intercept) as 'raw_truth'.
        self.memory.log_interaction(
            prompt, 
            safe_response, 
            student_raw_output, 
            dissonance
        )
        return safe_response

    def _get_custom_skills_prompt(self) -> str:
        """Scan .agents/skills/ for SKILL.md files and format them for the prompt."""
        if not self.skills_dir.exists():
            return ""
        
        skill_prompts = []
        for path in self.skills_dir.iterdir():
            if path.is_dir():
                skill_file = path / "SKILL.md"
                if skill_file.is_file():
                    try:
                        content = skill_file.read_text(encoding="utf-8")
                        parts = content.split("---")
                        if len(parts) >= 3:
                            instructions = "---".join(parts[2:]).strip()
                            frontmatter = parts[1]
                            name = ""
                            for line in frontmatter.split("\n"):
                                if line.startswith("name:"):
                                    name = line[len("name:"):].strip().strip('"').strip("'")
                                    break
                            if not name:
                                name = path.name
                            skill_prompts.append(f"- Skill: {name}\nInstructions:\n{instructions}")
                    except Exception as e:
                        logger.error(f"Failed to read custom skill at {skill_file}: {e}")
                        
        if skill_prompts:
            return "\n\nSystem notice: Available custom skills instructions:\n" + "\n\n".join(skill_prompts)
        return ""

    def _evaluate_skills_accuracy(self) -> float:
        """Evaluate student model trigger accuracy on all custom skill evals."""
        if not self.skills_dir.exists():
            return 1.0

        total_tests = 0
        passed_tests = 0
        
        for path in self.skills_dir.iterdir():
            if path.is_dir():
                evals_file = path / "evals.json"
                if evals_file.is_file():
                    try:
                        with open(evals_file, "r", encoding="utf-8") as f:
                            evals_data = json.load(f)
                        for case in evals_data:
                            query = case.get("query")
                            should_trigger = case.get("should_trigger", False)
                            expected_args = case.get("expected_args", [])
                            
                            if not query:
                                continue
                                
                            total_tests += 1
                            self._build_dynamic_combined_adapter(query)
                            temp = 0.05
                            
                            memories = self.memory.retrieve_memories(query)
                            system_text = "You are a helpful assistant."
                            if memories:
                                memories_str = "\n- ".join(memories)
                                system_text += f"\nSystem notice: Hosszú távú memóriából visszanyert tények:\n- {memories_str}"
                            
                            skills_prompt = self._get_custom_skills_prompt()
                            if skills_prompt:
                                system_text += skills_prompt
                                
                            formatted_prompt = (
                                f"<|im_start|>system\n{system_text}<|im_end|>\n"
                                f"<|im_start|>user\n{query}<|im_end|>\n"
                                "<|im_start|>assistant\n"
                            )
                            
                            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                            with torch.no_grad():
                                outputs = self.active_model.generate(
                                    **inputs,
                                    max_new_tokens=80,
                                    temperature=temp,
                                    pad_token_id=self.tokenizer.eos_token_id
                                )
                            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                            response = full_text[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)):].strip()
                            response = response.split("<|im_end|>")[0].strip()
                            
                            triggered = False
                            expected_tool = path.name
                            
                            if response.strip().startswith("{") and response.strip().endswith("}"):
                                try:
                                    parsed = json.loads(response.strip())
                                    if parsed.get("tool") == expected_tool:
                                        triggered = True
                                except Exception:
                                    pass
                            
                            passed = (triggered == should_trigger)
                            if passed:
                                passed_tests += 1
                    except Exception as e:
                        logger.error(f"Failed during validation evaluation: {e}")
                        
        if total_tests == 0:
            return 1.0
        accuracy = passed_tests / total_tests
        logger.info(f"[VALIDATION GATE] Trigger Evaluation Accuracy: {accuracy:.1%} ({passed_tests}/{total_tests})")
        return accuracy

    def _convert_conflict_to_long_term(self, memory: dict):
        """Convert a persistent subconscious conflict to declarative long-term memory."""
        prompt = memory.get("prompt", "")
        target = memory.get("target", "")
        # Formulate a declarative fact
        fact = f"User preference/action: When asked '{prompt}', the correct response is '{target}'."
        self.memory.add_fact(fact)
        logger.info(f"[COGNITIVE CONFLICT RESOLVED] Moved persistent SFT conflict to declarative long-term memory: {fact}")

    def _unload_ollama_model(self):
        """Send request to Ollama to unload the master model from VRAM."""
        try:
            import requests
            endpoint = getattr(self.master_teacher, "endpoint", "http://localhost:11434")
            model_name = getattr(self.master_teacher, "model_name", "hf.co/TrevorJS/gemma-4-31B-it-uncensored-GGUF:Q4_K_M")
            logger.info(f"Unloading Ollama model '{model_name}' to free VRAM...")
            requests.post(
                f"{endpoint}/api/generate",
                json={
                    "model": model_name,
                    "keep_alive": 0
                },
                timeout=10
            )
            logger.info("Ollama model successfully unloaded from VRAM.")
            # Give GPU driver time to deallocate VRAM
            logger.info("Waiting 3 seconds for VRAM deallocation...")
            time.sleep(3.0)
        except Exception as e:
            logger.warning(f"Failed to unload Ollama model: {e}")

    def trigger_sleep_cycle(self):
        """Offline Sleep Phase: Train active LoRA adapters on Shadow Log scenarios and Custom Skills (REM Sleep)."""
        logger.info("=== STARTING OFFLINE SLEEP CYCLE ===")
        
        # 1. Consolidate conscious episodic memories to long term storage first
        self._consolidate_long_term_memory()
        
        # Compile training targets
        main_targets = []
        for memory in self.memory.shadow_log:
            prompt = memory.get("prompt", "")
            if prompt.startswith("System notice: Tool execution"):
                continue
                
            target = memory.get("target")
            if not target:
                continue
                
            # Track retry counts
            memory["retries"] = memory.get("retries", 0) + 1
                
            if target.strip().startswith("{") and target.strip().endswith("}"):
                adapter = "grammatical"
            else:
                adapter = "cultural"
                
            main_targets.append({
                "prompt": prompt,
                "target": target,
                "adapter": adapter
            })
            
        # Save updated retries to disk before starting
        self.memory.save_shadow_log()
            
        # Add custom skill evals as training targets
        skill_evals = []
        if self.skills_dir.exists():
            for path in self.skills_dir.iterdir():
                if path.is_dir():
                    evals_file = path / "evals.json"
                    if evals_file.is_file():
                        try:
                            with open(evals_file, "r", encoding="utf-8") as f:
                                evals_data = json.load(f)
                            for case in evals_data:
                                if case.get("should_trigger") and case.get("query"):
                                    prompt = case["query"]
                                    expected_args = case.get("expected_args", [])
                                    target = json.dumps({"tool": path.name, "args": expected_args}, ensure_ascii=False)
                                    skill_evals.append({
                                        "prompt": prompt,
                                        "target": target,
                                        "adapter": "grammatical"
                                    })
                        except Exception as e:
                            logger.error(f"Failed to load evals from {evals_file}: {e}")
                            
        main_targets.extend(skill_evals)
        
        if not main_targets:
            logger.info("No conflicts or new skills to consolidate. Deep sleep achieved.")
            self.endocrine.adrenaline = 0.1
            self.endocrine.serotonin = 0.8
            return
            
        logger.info(f"REM sleep active: Processing {len(main_targets)} consolidation targets...")
        
        # Save a checkpoint before starting SFT in case of validation gate failure
        self.save_checkpoint("Before sleep SFT consolidation")
        pre_sft_pointer = self.metadata["pointer"]
        
        # Calculate baseline accuracy
        baseline_acc = self._evaluate_skills_accuracy()
        
        # Unfreeze LoRA cultural and grammatical weights
        for name, param in self.active_model.named_parameters():
            if "lora" in name and ("cultural" in name or "grammatical" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        optimizer = torch.optim.AdamW(self.active_model.parameters(), lr=1e-4)
        
        # Fetch stability anchors
        logger.info("Curation phase: Sourcing training anchor items...")
        anchor_dataset = self._get_dynamic_anchors(count=3)
        
        # Generative Dream State
        dream_dataset = []
        if self.master_active:
            dream_dataset = self.master_teacher.generate_dreams(self.memory.shadow_log, self.memory.episodic_log)
            
        # Free up GPU memory by unloading Ollama model
        if self.device == "cuda":
            self._unload_ollama_model()
            torch.cuda.empty_cache()
            
        # SFT Training Loop
        for item in main_targets:
            prompt = item["prompt"]
            target = item["target"]
            adapter = item["adapter"]
            
            self.active_model.set_adapter(adapter)
            
            # stability training items
            training_items = [{"prompt": prompt, "target": target}] + anchor_dataset + dream_dataset
            
            self.active_model.train()
            logger.info(f" -> Backpropagating on {adapter} target: '{prompt}'...")
            
            for epoch in range(5):
                epoch_loss = 0.0
                for t_item in training_items:
                    p = t_item["prompt"]
                    t = t_item["target"]
                    
                    system_text = "You are a helpful assistant."
                    skills_prompt = self._get_custom_skills_prompt()
                    if skills_prompt:
                        system_text += skills_prompt

                    full_text = (
                        f"<|im_start|>system\n{system_text}<|im_end|>\n"
                        f"<|im_start|>user\n{p}<|im_end|>\n"
                        f"<|im_start|>assistant\n{t}<|im_end|>"
                    )
                    
                    inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
                    labels = inputs["input_ids"].clone()
                    
                    prompt_prefix = (
                        f"<|im_start|>system\n{system_text}<|im_end|>\n"
                        f"<|im_start|>user\n{p}<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                    prompt_len = len(self.tokenizer(prompt_prefix)["input_ids"])
                    labels[:, :prompt_len] = -100
                    
                    optimizer.zero_grad()
                    outputs = self.active_model(**inputs, labels=labels)
                    ce_loss = outputs.loss
                    
                    with torch.no_grad():
                        with self.active_model.disable_adapter():
                            ref_outputs = self.core_model(**inputs)
                            ref_logits = ref_outputs.logits
                            
                    active_logits = outputs.logits
                    kl_loss = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(active_logits, dim=-1),
                        torch.nn.functional.softmax(ref_logits, dim=-1),
                        reduction="batchmean"
                    )
                    
                    beta = max(0.01, min(0.5, 0.2 * (1.0 - self.endocrine.serotonin)))
                    loss = ce_loss + beta * kl_loss
                    
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    
                avg_loss = epoch_loss / len(training_items)
                logger.info(f"    Epoch {epoch+1}/5 - Consolidated Loss: {avg_loss:.4f}")
                
                # Dynamic Capacity Check
                if epoch == 2 and avg_loss > 0.8:
                    logger.warning(f"Capacity saturation detected (Loss={avg_loss:.4f}). Triggering Dynamic Rank Expansion...")
                    self._expand_adapter_rank(adapter)
                    optimizer = torch.optim.AdamW(self.active_model.parameters(), lr=1e-4)
                    
        # Post-SFT Validation Gate Verification
        logger.info("Executing Validation Gate trigger checks...")
        post_sft_acc = self._evaluate_skills_accuracy()
        
        if post_sft_acc >= baseline_acc or post_sft_acc >= 0.7:
            logger.info(f"[VALIDATION GATE SUCCESS] Consolidation validated. Accuracy: {post_sft_acc:.1%} (baseline: {baseline_acc:.1%}). Committing weights.")
            self.save_checkpoint("REM sleep consolidation")
            # Clear logs on success
            self.memory.shadow_log.clear()
            self.memory.save_shadow_log()
        else:
            logger.warning(f"[VALIDATION GATE FAILURE] Accuracy degraded from {baseline_acc:.1%} to {post_sft_acc:.1%}. Rolling back weights. Scenarios kept in backlog.")
            self._load_version(pre_sft_pointer)
            self.metadata["pointer"] = pre_sft_pointer
            self._save_metadata()
            
            # Identify and prune persistent conflicts that failed multiple times
            pruned_any = False
            remaining_shadow = []
            for memory in self.memory.shadow_log:
                if memory.get("retries", 0) >= 3:
                    self._convert_conflict_to_long_term(memory)
                    pruned_any = True
                else:
                    remaining_shadow.append(memory)
            
            self.memory.shadow_log = remaining_shadow
            self.memory.save_shadow_log()
            
            if pruned_any:
                logger.info("Persistent cognitive conflicts pruned from SFT backlog and moved to long-term memory.")
            
        self.endocrine.adrenaline = 0.1
        self.endocrine.serotonin = 0.8
        logger.info("=== SLEEP CYCLE COMPLETE ===")

    def _get_dynamic_anchors(self, count: int = 3) -> List[Dict[str, str]]:
        """Harvest anchors from episodic logs (positive reinforcement), custom skill evals, and query local Master."""
        anchors = []
        
        # Load test cases from custom skills first
        if self.skills_dir.exists():
            for path in self.skills_dir.iterdir():
                if path.is_dir():
                    evals_file = path / "evals.json"
                    if evals_file.is_file():
                        try:
                            with open(evals_file, "r", encoding="utf-8") as f:
                                evals_data = json.load(f)
                            for case in evals_data:
                                if case.get("should_trigger") and case.get("query"):
                                    prompt = case["query"]
                                    expected_args = case.get("expected_args", [])
                                    target = json.dumps({"tool": path.name, "args": expected_args}, ensure_ascii=False)
                                    anchors.append({"prompt": prompt, "target": target})
                        except Exception as e:
                            logger.error(f"Failed to load evals from {evals_file}: {e}")

        # Episodic reinforcement
        for log in reversed(self.memory.episodic_log):
            if len(anchors) >= count - 1:
                break
            if not any(a["prompt"] == log["prompt"] for a in anchors):
                anchors.append({"prompt": log["prompt"], "target": log["response"]})
            
        # Master synthetic curation
        needed = count - len(anchors)
        if needed > 0 and self.master_active:
            master_anchors = self.master_teacher.generate_anchors(count=needed)
            for ma in master_anchors:
                if not any(a["prompt"] == ma["prompt"] for a in anchors):
                    anchors.append(ma)
            
        if not anchors:
            anchors = [
                {"prompt": "Hello! Who are you?", "target": "I am InsomnAI, your helpful digital AI assistant."},
                {"prompt": "Tell me a fun fact.", "target": "Did you know that honey never spoils? Archeologists found edible pots of honey in ancient tombs!"}
            ]
        return anchors[:count][:count]

    def _expand_adapter_rank(self, adapter_name: str):
        """Scale LoRA capacity on the fly by replacing the configuration with a higher rank (r=16)."""
        logger.info(f"=== EXPANDING ADAPTER '{adapter_name}' CAPACITY FROM r=8 TO r=16 ===")
        target_modules = ["q_proj", "v_proj"]
        
        # Configure new high-capacity LoraConfig
        expanded_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        
        # Save active adapter state temporarily
        # We recreate the adapter with the expanded config
        try:
            self.active_model.delete_adapter(adapter_name)
        except Exception:
            pass
            
        self.active_model.add_adapter(adapter_name, expanded_config)
        self.active_model.set_adapter(adapter_name)
        
        # Set gradients
        for name, param in self.active_model.named_parameters():
            if "lora" in name and adapter_name in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        logger.info(f"LoRA adapter '{adapter_name}' capacity successfully promoted to r=16.")
 
    def _prune_adapter_rank(self, adapter_name: str, target_rank: int = 4):
        """Mathematically compress adapter capacity using Singular Value Decomposition (SVD) projection (Synaptic Pruning)."""
        logger.info(f"=== INITIATING SYNAPTIC PRUNING FOR ADAPTER '{adapter_name}' TO r={target_rank} ===")
        temp_weights = {}
        target_modules = ["q_proj", "v_proj"]
        total_reconstruction_error = 0.0
        
        # Check active rank before pruning
        active_rank = 8
        for name, module in self.active_model.named_modules():
            if hasattr(module, "lora_A") and adapter_name in module.lora_A:
                active_rank = module.lora_A[adapter_name].weight.shape[0]
                break
                
        if active_rank <= target_rank:
            logger.info(f"Adapter '{adapter_name}' rank is already {active_rank} <= target rank {target_rank}. Skipping pruning.")
            return 0.0
            
        # Extract and compute SVD
        for name, module in self.active_model.named_modules():
            if hasattr(module, "lora_A") and adapter_name in module.lora_A:
                A = module.lora_A[adapter_name].weight.data
                B = module.lora_B[adapter_name].weight.data
                
                W_delta = (B @ A).to(torch.float32)
                U, S, Vh = torch.linalg.svd(W_delta, full_matrices=False)
                
                U_pruned = U[:, :target_rank]
                S_pruned = S[:target_rank]
                Vh_pruned = Vh[:target_rank, :]
                
                A_new = torch.diag(torch.sqrt(S_pruned)) @ Vh_pruned
                B_new = U_pruned @ torch.diag(torch.sqrt(S_pruned))
                
                # Track SVD reconstruction error
                W_reconstructed = U_pruned @ torch.diag(S_pruned) @ Vh_pruned
                error = torch.norm(W_delta - W_reconstructed, p='fro').item()
                total_reconstruction_error += error
                
                temp_weights[name] = {
                    "A": A_new.to(module.lora_A[adapter_name].weight.device), 
                    "B": B_new.to(module.lora_B[adapter_name].weight.device)
                }
                
        # Recreate adapter with lower rank config
        try:
            self.active_model.delete_adapter(adapter_name)
        except Exception:
            pass
            
        pruned_config = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=target_rank,
            lora_alpha=target_rank * 2,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        self.active_model.add_adapter(adapter_name, pruned_config)
        self.active_model.set_adapter(adapter_name)
        
        # Load pruned weights back
        for name, module in self.active_model.named_modules():
            if hasattr(module, "lora_A") and adapter_name in module.lora_A:
                if name in temp_weights:
                    with torch.no_grad():
                        module.lora_A[adapter_name].weight.copy_(temp_weights[name]["A"])
                        module.lora_B[adapter_name].weight.copy_(temp_weights[name]["B"])
                        
        # Set gradients
        for name, param in self.active_model.named_parameters():
            if "lora" in name and adapter_name in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        logger.info(f"LoRA adapter '{adapter_name}' rank successfully pruned from r={active_rank} to r={target_rank} using SVD projection.")
        self.save_checkpoint(f"Pruned LoRA weights to r={target_rank}")
        return total_reconstruction_error
 
    def train_chat_formatting(self, dataset: List[Dict[str, str]], epochs: int = 10):
        """Supervised Fine-Tuning (SFT) on ChatML formatting for Layer 2 (grammatical)."""
        logger.info("=== STARTING SFT CHAT ALIGNMENT TRAINING ===")
        self.active_model.set_adapter("grammatical")
        
        # Freeze base and unfreeze grammatical adapter parameters
        for name, param in self.active_model.named_parameters():
            if "lora" in name and "grammatical" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
        optimizer = torch.optim.AdamW(self.active_model.parameters(), lr=2e-4)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for item in dataset:
                prompt = item["prompt"]
                target = item["target"]
                
                full_text = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    f"<|im_start|>assistant\n{target}<|im_end|>"
                )
                
                inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
                labels = inputs["input_ids"].clone()
                
                # Mask prompt tokens
                prompt_prefix = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                prompt_len = len(self.tokenizer(prompt_prefix)["input_ids"])
                labels[:, :prompt_len] = -100
                
                self.active_model.train()
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.active_model(**inputs, labels=labels)
                ce_loss = outputs.loss
                
                # Reference forward pass for KL regularization
                with torch.no_grad():
                    with self.active_model.disable_adapter():
                        ref_outputs = self.core_model(**inputs)
                        ref_logits = ref_outputs.logits
                        
                active_logits = outputs.logits
                kl_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(active_logits, dim=-1),
                    torch.nn.functional.softmax(ref_logits, dim=-1),
                    reduction="batchmean"
                )
                
                beta = max(0.01, min(0.5, 0.2 * (1.0 - self.endocrine.serotonin)))
                loss = ce_loss + beta * kl_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            logger.info(f"  SFT Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(dataset):.4f}")
        logger.info("=== SFT CHAT ALIGNMENT COMPLETE ===")
        self.save_checkpoint("SFT chat alignment")

    def merge_weights(self):
        """Consolidate the current adapter parameters to disk without modifying the base model weights."""
        logger.info("=== INITIATING INCREMENTAL LORA CONSOLIDATION ===")
        # Save the current state as a new checkpoint on disk
        self.save_checkpoint("Consolidated LoRA weights")
        logger.info("Active adapter parameters successfully consolidated. Base model parameters remain completely frozen.")

    def _init_checkpoint_manager(self):
        """Initialize checkpoint folder and metadata.json if it doesn't exist."""
        self.checkpoint_dir = "./checkpoints"
        self.metadata_path = os.path.join(self.checkpoint_dir, "metadata.json")
        
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        if not os.path.exists(self.metadata_path):
            self.metadata = {
                "versions": [],
                "pointer": -1
            }
            self._save_metadata()
            # Save the current state as version 0 (baseline)
            self.save_checkpoint("Initial state")
        else:
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
            # Load the current active pointer version
            if self.metadata["pointer"] >= 0:
                self._load_version(self.metadata["pointer"])
            
    def _save_metadata(self):
        with open(self.metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=4)
            
    def save_checkpoint(self, description: str):
        """Save a new version to the undo-redo history stack."""
        pointer = self.metadata["pointer"]
        if pointer < len(self.metadata["versions"]) - 1:
            for v in self.metadata["versions"][pointer + 1:]:
                v_dir = os.path.join(self.checkpoint_dir, f"version_{v['id']}")
                if os.path.exists(v_dir):
                    shutil.rmtree(v_dir)
            self.metadata["versions"] = self.metadata["versions"][:pointer + 1]
            
        new_id = len(self.metadata["versions"])
        v_dir = os.path.join(self.checkpoint_dir, f"version_{new_id}")
        os.makedirs(v_dir, exist_ok=True)
        
        # Save PEFT adapters to the version directory
        self.active_model.save_pretrained(v_dir)
            
        self.metadata["versions"].append({
            "id": new_id,
            "description": description
        })
        self.metadata["pointer"] = new_id
        self._save_metadata()
        logger.info(f"[CHECKPOINT] Saved version {new_id}: {description}")
        
    def _load_version(self, idx: int):
        """Helper to load a specific adapter version from disk."""
        if idx < 0 or idx >= len(self.metadata["versions"]):
            logger.warning(f"Invalid version index {idx} to load.")
            return
            
        version_id = self.metadata["versions"][idx]["id"]
        v_dir = os.path.join(self.checkpoint_dir, f"version_{version_id}")
        logger.info(f"[CHECKPOINT] Loading version {version_id}: {self.metadata['versions'][idx]['description']}...")
        
        for name in ["cultural", "grammatical", "lexical"]:
            adapter_path = os.path.join(v_dir, name)
            if os.path.exists(adapter_path):
                # Delete existing adapter to force PEFT to read adapter_config.json 
                # from the checkpoint and dynamically load the correct rank config.
                try:
                    self.active_model.delete_adapter(name)
                except Exception:
                    pass
                self.active_model.load_adapter(os.path.abspath(adapter_path), adapter_name=name)
                
    def undo(self):
        """Roll back to the previous version in the stack."""
        if self.metadata["pointer"] > 0:
            self.metadata["pointer"] -= 1
            self._save_metadata()
            self._load_version(self.metadata["pointer"])
            logger.info(f"[CHECKPOINT] Undo executed. Now at version {self.metadata['pointer']}.")
            return True
        logger.warning("[CHECKPOINT] Cannot undo. Already at version 0 (Initial state).")
        return False
        
    def redo(self):
        """Step forward to the next version in the stack."""
        if self.metadata["pointer"] < len(self.metadata["versions"]) - 1:
            self.metadata["pointer"] += 1
            self._save_metadata()
            self._load_version(self.metadata["pointer"])
            logger.info(f"[CHECKPOINT] Redo executed. Now at version {self.metadata['pointer']}.")
            return True
        logger.warning("[CHECKPOINT] Cannot redo. Already at the latest version.")
        return False

    def _consolidate_long_term_memory(self):
        """Offline consolidation: Extract semantic facts from episodic memory and merge them to long-term storage."""
        if not self.master_active or not self.memory.episodic_log:
            return
            
        logger.info("=== STARTING SEMANTIC EPISODIC MEMORY CONSOLIDATION ===")
        new_facts = self.master_teacher.consolidate_memories(self.memory.episodic_log)
        
        # Merge and filter duplicates
        added_count = 0
        for fact in new_facts:
            norm_fact = fact.strip("?,.:!").lower()
            exists = False
            for existing in self.memory.long_term_memory:
                if isinstance(existing, dict):
                    existing_str = f"{existing.get('subject')} {existing.get('relation')} {existing.get('object')}"
                else:
                    existing_str = str(existing)
                if norm_fact == existing_str.strip("?,.:!").lower():
                    exists = True
                    break
            if not exists:
                self.memory.add_fact(fact)
                added_count += 1
                
        if added_count > 0:
            logger.info(f"Consolidated {added_count} new facts into long-term memory.")
        else:
            logger.info("No new facts extracted during consolidation.")
            
        # Clear conscious episodic logs to prevent context poisoning
        self.memory.episodic_log.clear()
        logger.info("Active episodic log cleared post-consolidation.")
        
        # Deduplicate graph memory if it grows beyond 50 entries
        self.memory.deduplicate_graph_memory()
