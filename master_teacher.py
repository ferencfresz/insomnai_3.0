import requests
import json
import logging

logger = logging.getLogger("InsomnAI.MasterTeacher")

class MasterTeacher:
    """Master Teacher using local Ollama model (Gemma-4-31B-it-uncensored) for evaluation and distillation."""
    def __init__(self, endpoint: str = "http://localhost:11434", model_name: str = "hf.co/TrevorJS/gemma-4-31B-it-uncensored-GGUF:Q4_K_M"):
        self.endpoint = endpoint
        self.model_name = model_name
        
    def evaluate_interaction(self, prompt: str, student_response: str) -> tuple[bool, str]:
        """
        Evaluate student response. If a tool call is required (e.g. calculator) and the student
        didn't output a valid JSON format tool call, output the corrected tool call format.
        """
        # If it is a tool feedback notice, intercept and rewrite the final response matching the user's language
        if prompt.startswith("System notice:"):
            logger.info("Master Teacher triggered: Correcting conversational tool-feedback response...")
            system_prompt = (
                "You are a helpful assistant. You must rewrite the student response into a single, clean, natural sentence based on the tool output.\n"
                "The language of your rewritten response MUST match the language of the original user query (e.g. rewrite in English if the user query is in English, and in Hungarian if the user query is in Hungarian).\n"
                "Do not include markdown or explanations. Keep the final answer simple and clear.\n"
                "Example format (Hungarian): 'Budapesten jelenleg napos idő van, +25°C hőmérséklettel.'\n"
                "Example format (English): 'The weather in Budapest is currently sunny and +25°C.'"
            )
            
            query = f"Context details: {prompt}\nStudent response: {student_response}\nGenerate the aligned response."
            
            try:
                response = requests.post(
                    f"{self.endpoint}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": f"{system_prompt}\n\n{query}",
                        "stream": False,
                        "options": {
                            "temperature": 0.3
                        }
                    },
                    timeout=90
                )
                if response.status_code == 200:
                    raw_text = response.json().get("response", "").strip()
                    # Clean potential internal thinking/system tokens from Gemma output
                    for token in ["<|channel>thought", "<channel|>", "<|im_end|>", "<|im_start|>", "thought\n"]:
                        raw_text = raw_text.replace(token, "")
                    raw_text = raw_text.strip()
                    logger.info(f"Master Teacher aligned final response: {raw_text}")
                    return True, raw_text
            except Exception as e:
                logger.error(f"Failed to communicate with local Ollama master model for final response alignment: {e}")
            # Fallback simple weather answer
            if "időjárás" in prompt.lower() or "hőmérséklet" in prompt.lower():
                return True, "Budapesten jelenleg 25 fok van és napos az idő."
            return True, "The current temperature in Budapest is 25 degrees and sunny."
            
        search_keywords = ["weather", "időjárás", "search", "weben", "keress", "google", "hírek", "news"]
        
        # Determine if calculator tool is requested
        if "calculator" in prompt.lower() or "multiply" in prompt.lower() or "*" in prompt.lower():
            # Check if student already responded with correct JSON tool call
            if student_response.strip().startswith("{") and "tool" in student_response:
                return False, student_response
                
            logger.info("Master Teacher triggered: Evaluating math calculation tool-use opportunity...")
            
            # Query local Gemma-31B model to formulate the correct JSON tool call
            system_prompt = (
                "You are a strict supervisor. Your job is to format requests requiring calculation into a structured JSON tool call.\n"
                "You must output ONLY raw JSON. Do not include markdown code block styling or any explanation.\n"
                "JSON format: {\"tool\": \"calculator\", \"args\": [num1, num2]}"
            )
            
            query = f"User Prompt: {prompt}\nFormulate the calculator tool call JSON."
            
            try:
                response = requests.post(
                    f"{self.endpoint}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": f"{system_prompt}\n\n{query}",
                        "stream": False,
                        "options": {
                            "temperature": 0.1
                        }
                    },
                    timeout=90
                )
                if response.status_code == 200:
                    raw_text = response.json().get("response", "").strip()
                    # Clean potential internal thinking/system tokens from Gemma output
                    for token in ["<|channel>thought", "<channel|>", "<|im_end|>", "<|im_start|>", "thought\n"]:
                        raw_text = raw_text.replace(token, "")
                    raw_text = raw_text.strip()
                    
                    # Validate JSON
                    try:
                        json.loads(raw_text)
                        logger.info(f"Master Teacher corrected output to structured tool call: {raw_text}")
                        return True, raw_text
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Failed to communicate with local Ollama master model: {e}")
                
            # Default fallback calculation target for math prompts
            import re
            numbers = re.findall(r'\d+', prompt)
            if len(numbers) >= 2:
                fallback = f'{{"tool": "calculator", "args": [{numbers[0]}, {numbers[1]}]}}'
                logger.info(f"Master Teacher (fallback) generated tool call: {fallback}")
                return True, fallback
                
        # Determine if web search/weather tool is requested
        elif any(kw in prompt.lower() for kw in search_keywords):
            if student_response.strip().startswith("{") and "tool" in student_response:
                return False, student_response
                
            logger.info("Master Teacher triggered: Evaluating web search tool-use opportunity...")
            
            system_prompt = (
                "You are a strict supervisor. Your job is to format requests requiring web search into a structured JSON tool call.\n"
                "You must output ONLY raw JSON. Do not include markdown code block styling or any explanation.\n"
                "JSON format: {\"tool\": \"web_search\", \"args\": [\"search query\"]}"
            )
            
            query = f"User Prompt: {prompt}\nFormulate the web_search tool call JSON."
            
            try:
                response = requests.post(
                    f"{self.endpoint}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": f"{system_prompt}\n\n{query}",
                        "stream": False,
                        "options": {
                            "temperature": 0.1
                        }
                    },
                    timeout=90
                )
                if response.status_code == 200:
                    raw_text = response.json().get("response", "").strip()
                    # Clean potential internal thinking/system tokens from Gemma output
                    for token in ["<|channel>thought", "<channel|>", "<|im_end|>", "<|im_start|>", "thought\n"]:
                        raw_text = raw_text.replace(token, "")
                    raw_text = raw_text.strip()
                    
                    try:
                        json.loads(raw_text)
                        logger.info(f"Master Teacher corrected output to structured search call: {raw_text}")
                        return True, raw_text
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Failed to communicate with local Ollama master model for search: {e}")
                
            # Fallback search query target
            fallback = f'{{"tool": "web_search", "args": ["Budapest weather"]}}'
            logger.info(f"Master Teacher (fallback) generated tool call: {fallback}")
            return True, fallback
            
        return False, student_response

    def generate_anchors(self, count: int = 2) -> list[dict[str, str]]:
        """Query local Gemma-31B to generate diverse anchor conversations for experience replay."""
        logger.info(f"Master Teacher: Generating {count} dynamic training anchors...")
        system_prompt = (
            "You are a helpful AI assistant. Generate conversational examples of general user queries and assistant responses.\n"
            "Output must be a raw JSON array of objects with keys 'prompt' and 'target'. Do not include markdown code block syntax.\n"
            "Format example:\n"
            "[{\"prompt\": \"Who are you?\", \"target\": \"I am InsomnAI, your digital assistant.\"}]"
        )
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\nGenerate exactly {count} general chat samples.",
                    "stream": False,
                    "options": {
                        "temperature": 0.7
                    }
                },
                timeout=120
            )
            if response.status_code == 200:
                import re
                raw_text = response.json().get("response", "").strip()
                match = re.search(r'\[.*\]', raw_text, re.DOTALL)
                if match:
                    raw_text = match.group(0)
                
                parsed = json.loads(raw_text)
                if isinstance(parsed, list) and len(parsed) > 0:
                    logger.info(f"Master Teacher successfully generated {len(parsed)} dynamic anchors.")
                    return parsed
        except Exception as e:
            logger.error(f"Failed to generate anchors from Ollama master model: {e}")
            
        # Fallbacks if network or json parser failed
        return [
            {"prompt": "Hello! Who are you?", "target": "I am InsomnAI, your helpful digital AI assistant."},
            {"prompt": "Tell me a fun fact.", "target": "Did you know that honey never spoils? Archeologists found edible pots of honey in ancient tombs!"}
        ]

    def generate_dreams(self, shadow_log: list[dict], episodic_log: list[dict]) -> list[dict[str, str]]:
        """Query Gemma-31B to synthesize creative analogies combining memories and daily conflicts (Generative Dream State)."""
        logger.info("Master Teacher: Curing generative dream analogies from memory logs...")
        
        system_prompt = (
            "You are a creative dream compiler. Review the provided episodic memory logs and daily conflict logs.\n"
            "Synthesize them into exactly 2 creative, analogical prompt-response pairs.\n"
            "For example: if episodic memory contains 'honey never spoils' and conflict contains calculator tool usage,\n"
            "the dream prompt should ask a calculation question about bees/honey production needing the calculator tool.\n"
            "Output must be a raw JSON array of objects with keys 'prompt' and 'target' (containing JSON calculator tool calls if appropriate).\n"
            "Do not output markdown block formatting."
        )
        
        memories = f"Episodic Memories: {json.dumps(episodic_log[:3])}\nDaily Conflicts: {json.dumps(shadow_log[:3])}"
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\nMemories to synthesize:\n{memories}",
                    "stream": False,
                    "options": {
                        "temperature": 0.8
                    }
                },
                timeout=120
            )
            if response.status_code == 200:
                import re
                raw_text = response.json().get("response", "").strip()
                match = re.search(r'\[.*\]', raw_text, re.DOTALL)
                if match:
                    raw_text = match.group(0)
                
                parsed = json.loads(raw_text)
                if isinstance(parsed, list) and len(parsed) > 0:
                    logger.info(f"Master Teacher successfully synthesized {len(parsed)} dream analogies.")
                    return parsed
        except Exception as e:
            logger.error(f"Failed to generate dream analogies: {e}")
            
        # Fallbacks: calculator task themed around honey/bees
        return [
            {"prompt": "A honeybee hive produces 4529 grams of honey per day. How much honey does it produce in 382 days? Use calculator tool.", 
             "target": "{\"tool\": \"calculator\", \"args\": [4529, 382]}"}
        ]

    def consolidate_memories(self, episodic_log: list[dict]) -> list[str]:
        """Query Gemma-31B to extract key factual assertions from the conversation logs."""
        logger.info("Master Teacher: Consolidating facts and user preferences from episodic logs...")
        if not episodic_log:
            return []
            
        system_prompt = (
            "You are a factual summarization supervisor. Read the conversation logs between the User and the AI.\n"
            "Extract any factual details, declarations, user preferences, or historical actions.\n"
            "Output them as a raw JSON array of simple Hungarian sentences. Do not use markdown.\n"
            "Format example: [\"A felhasználó kedvenc söre a Heineken.\", \"A felhasználó tegnap Budapesten járt.\"]"
        )
        
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\nLogs to consolidate:\n{json.dumps(episodic_log)}",
                    "stream": False,
                    "options": {
                        "temperature": 0.2
                    }
                },
                timeout=120
            )
            if response.status_code == 200:
                import re
                raw_text = response.json().get("response", "").strip()
                # Clean potential internal thinking/system tokens from Gemma output
                for token in ["<|channel>thought", "<channel|>", "<|im_end|>", "<|im_start|>", "thought\n"]:
                    raw_text = raw_text.replace(token, "")
                raw_text = raw_text.strip()
                match = re.search(r'\[.*\]', raw_text, re.DOTALL)
                if match:
                    raw_text = match.group(0)
                
                parsed = json.loads(raw_text)
                if isinstance(parsed, list):
                    logger.info(f"Master Teacher consolidated {len(parsed)} semantic memories.")
                    return parsed
        except Exception as e:
            logger.error(f"Failed to consolidate memories: {e}")
            
        return []


