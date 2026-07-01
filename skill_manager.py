import json
import logging
import os
import shutil
import subprocess
import requests
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger("InsomnAI.SkillManager")

class SkillManager:
    """
    SkillManager handles automated, local skill acquisition.
    Inspired by SkillAnything, it analyzes a target (CLI tool, API, or service),
    generates SKILL.md, and creates evals.json for evaluation and sleep training.
    """
    def __init__(self, endpoint: str = "http://localhost:11434", model_name: str = "hf.co/TrevorJS/gemma-4-31B-it-uncensored-GGUF:Q4_K_M"):
        self.endpoint = endpoint
        self.model_name = model_name
        self.skills_dir = Path(".agents/skills")
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def _preload_model(self):
        """Preload the master Ollama model into VRAM with a long timeout."""
        try:
            logger.info(f"Preloading master model '{self.model_name}' into GPU VRAM. This can take up to 2 minutes...")
            requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model_name
                },
                timeout=180
            )
            logger.info("Master model successfully loaded and ready.")
        except Exception as e:
            logger.warning(f"Master model preloading warning: {e}")

    def _query_master(self, system_prompt: str, prompt: str, temperature: float = 0.2) -> str:
        """Helper to query the local Ollama master model."""
        try:
            response = requests.post(
                f"{self.endpoint}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\n{prompt}",
                    "stream": False,
                    "options": {
                        "temperature": temperature
                    }
                },
                timeout=150
            )
            if response.status_code == 200:
                raw_text = response.json().get("response", "").strip()
                # Clean potential internal thinking/system tokens from Gemma output
                for token in ["<|channel>thought", "<channel|>", "<|im_end|>", "<|im_start|>", "thought\n"]:
                    raw_text = raw_text.replace(token, "")
                return raw_text.strip()
        except Exception as e:
            logger.error(f"Ollama query failed: {e}")
        return ""

    def skillify(self, target: str) -> dict:
        """
        Run the 4-phase local skillification pipeline for a given target.
        Returns a dict summarizing the result.
        """
        logger.info(f"Starting skillification for target: {target}")
        
        # Preload master model to VRAM to avoid load timeout during execution
        self._preload_model()
        
        # Phase 1: Analyze target
        analysis = self._phase_analyze(target)
        target_name = analysis.get("target_name", target)
        
        # Prepare output directory
        dest_dir = self.skills_dir / target_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analysis.json
        with open(dest_dir / "analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
            
        # Phase 2-3: Design & Implement (SKILL.md)
        skill_content = self._phase_design_and_implement(analysis)
        with open(dest_dir / "SKILL.md", "w", encoding="utf-8") as f:
            f.write(skill_content)
            
        # Phase 4: Generate Evals (evals.json)
        evals = self._phase_generate_evals(analysis)
        with open(dest_dir / "evals.json", "w", encoding="utf-8") as f:
            json.dump(evals, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully created skill for {target_name} at {dest_dir}")
        return {
            "status": "success",
            "target": target_name,
            "path": str(dest_dir),
            "capabilities": len(analysis.get("capabilities", [])),
            "eval_queries": len(evals)
        }

    def _phase_analyze(self, target: str) -> dict:
        """Phase 1: Determine target type, capture help info, compile analysis.json."""
        logger.info(f"Phase 1: Analyzing target: {target}...")
        
        target_type = "unknown"
        raw_help = ""
        
        # Heuristic 1: URL Target
        if target.startswith("http://") or target.startswith("https://"):
            try:
                resp = requests.get(target, timeout=10)
                raw_help = resp.text[:12000]
                
                # Check for OpenAPI/Swagger signals
                openapi_signals = ["openapi", "swagger", '"paths"', '"info"', "operationId"]
                hits = sum(1 for s in openapi_signals if s.lower() in raw_help.lower())
                if hits >= 2 or "application/json" in resp.headers.get("Content-Type", ""):
                    target_type = "api"
                else:
                    target_type = "service"
            except Exception as e:
                raw_help = f"Failed to fetch URL docs: {e}"
                target_type = "service"
                
        # Heuristic 2: Local File/Directory Path
        elif os.path.exists(target):
            path_obj = Path(target)
            if path_obj.is_dir():
                target_type = "library"
                # Gather files summary and read README if exists
                files = [p.name for p in path_obj.iterdir() if p.is_file()][:10]
                raw_help = f"Local directory: {target}\nFiles: {', '.join(files)}\n"
                readme_path = path_obj / "README.md"
                if not readme_path.is_file():
                    readme_path = path_obj / "readme.md"
                if readme_path.is_file():
                    try:
                        raw_help += f"\nREADME Content:\n{readme_path.read_text(encoding='utf-8', errors='ignore')[:8000]}"
                    except Exception:
                        pass
            else:
                # File-based target detection
                try:
                    file_content = path_obj.read_text(encoding='utf-8', errors='ignore')[:8000]
                    raw_help = f"Local file: {target}\nContent:\n{file_content}"
                    
                    if any(kw in file_content.lower() for kw in ["openapi", "swagger", "operationid"]):
                        target_type = "api"
                    elif any(kw in file_content.lower() for kw in ["import ", "def ", "class ", "function "]):
                        target_type = "library"
                    elif any(kw in file_content.lower() for kw in ["step ", "workflow", "pipeline", "stage"]):
                        target_type = "workflow"
                    else:
                        target_type = "file"
                except Exception as e:
                    raw_help = f"Failed to read local file target: {e}"
                    target_type = "file"
                    
        # Heuristic 3: Executable in PATH
        elif shutil.which(target):
            target_type = "cli"
            # Run help command to capture capabilities
            try:
                res = subprocess.run([target, "--help"], capture_output=True, text=True, timeout=5)
                raw_help = (res.stdout or "") + "\n" + (res.stderr or "")
                if not raw_help.strip():
                    res = subprocess.run([target, "-h"], capture_output=True, text=True, timeout=5)
                    raw_help = (res.stdout or "") + "\n" + (res.stderr or "")
            except Exception as e:
                raw_help = f"Failed to run executable help command: {e}"
                
        # Heuristic 4: Python Package in current environment
        else:
            try:
                import importlib.util
                spec = importlib.util.find_spec(target)
                if spec is not None:
                    target_type = "library"
                    # Capture package version and docstring info via pip show
                    res = subprocess.run(["pip", "show", target], capture_output=True, text=True, timeout=5)
                    raw_help = res.stdout or f"Python package '{target}' is installed."
            except Exception:
                pass
                
        if target_type == "unknown":
            raw_help = f"Target '{target}' not found in PATH, as local file path, or as installed package."
            
        # Ask master model to parse raw help and compile a clean capability manifest
        system_prompt = (
            "You are an expert software analyzer. Parse the provided raw help/docs and compile a structured capability manifest.\n"
            "You must output ONLY raw JSON matching the requested format. Do not include markdown code block syntax or explanations.\n"
            "Format:\n"
            "{\n"
            '  "target_name": "name of target",\n'
            '  "target_type": "cli | api | library | workflow | service | unknown",\n'
            '  "capabilities": [\n'
            '    {"name": "action name", "description": "brief description of what it does", "args": ["--option1", "-o2"]}\n'
            "  ],\n"
            '  "error_patterns": ["common error messages"]\n'
            "}"
        )
        
        prompt = f"Target Identifier: {target}\nDetected Type: {target_type}\nRaw Help/Docs:\n{raw_help[:15000]}"
        
        response_text = self._query_master(system_prompt, prompt, temperature=0.1)
        
        # Fallback if Ollama response is empty or invalid
        try:
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            manifest = json.loads(response_text.strip())
        except Exception:
            logger.warning("Failed to parse master analysis response as JSON, compiling simple manifest.")
            manifest = {
                "target_name": os.path.basename(target).replace(".exe", "").lower(),
                "target_type": target_type,
                "capabilities": [
                    {"name": "execute", "description": f"Execute utility command for {target}", "args": []}
                ],
                "error_patterns": ["error", "invalid"]
            }
            
        manifest["analyzed_at"] = datetime.now(timezone.utc).isoformat()
        return manifest

    def _phase_design_and_implement(self, analysis: dict) -> str:
        """Phase 2-3: Design & Implement (generate SKILL.md content)."""
        logger.info(f"Phase 2-3: Designing and implementing SKILL.md for {analysis['target_name']}...")
        
        system_prompt = (
            "You are an expert AI agent customizer. Write a production-ready SKILL.md file for the given target.\n"
            "The output must contain standard frontmatter (name, description) and instructions on how the agent should trigger and use this tool.\n"
            "Format instructions clearly, specifying how the agent should formulate JSON tool calls.\n"
            "JSON Format instructions: System expects `{\"tool\": \"target_name\", \"args\": [\"arg1\", \"arg2\"]}` for executions.\n"
            "You must explicitly instruct the agent to output ONLY raw JSON in this format (without markdown block formatting, preamble, or conversational text) when triggering the tool.\n"
            "Output must be standard markdown content, including the YAML frontmatter. Do not warp it inside markdown code blocks."
        )
        
        prompt = f"Target Analysis Manifest:\n{json.dumps(analysis, indent=2)}"
        
        skill_content = self._query_master(system_prompt, prompt, temperature=0.3)
        if not skill_content:
            # Fallback SKILL.md
            name = analysis["target_name"]
            skill_content = (
                f"---\n"
                f"name: {name}\n"
                f"description: Automatically execute commands and actions using the {name} utility.\n"
                f"---\n\n"
                f"# {name.upper()} Skill\n\n"
                f"Use this skill when you need to perform actions using the `{name}` tool.\n"
                f"When this tool should be triggered, you MUST respond ONLY with a raw JSON block matching the format below. "
                f"Do not include any conversational preamble, explanation, or markdown code blocks around the JSON.\n\n"
                f"Format:\n"
                f'{{"tool": "{name}", "args": ["<args>"]}}\n'
            )
        return skill_content

    def _phase_generate_evals(self, analysis: dict) -> list[dict]:
        """Phase 4: Generate evaluation test cases."""
        logger.info(f"Phase 4: Generating test queries for {analysis['target_name']}...")
        
        system_prompt = (
            "You are an expert QA engineer. Generate exactly 6 evaluation test queries for this target tool.\n"
            "Four queries should target tasks that require this tool (should_trigger = true) with expected args.\n"
            "Two queries should be generic or unrelated chat queries that should NOT trigger this tool (should_trigger = false).\n"
            "Output must be a raw JSON array of objects. Do not include markdown or explanations.\n"
            "JSON format example:\n"
            "[\n"
            "  {\n"
            '    "query": "compress the files doc.txt and image.png into archive.tar using tar",\n'
            '    "should_trigger": true,\n'
            '    "expected_args": ["-cvf", "archive.tar", "doc.txt", "image.png"]\n'
            "  },\n"
            "  {\n"
            '    "query": "tell me a joke",\n'
            '    "should_trigger": false,\n'
            '    "expected_args": []\n'
            "  }\n"
            "]"
        )
        
        prompt = f"Target Analysis Manifest:\n{json.dumps(analysis, indent=2)}"
        
        response_text = self._query_master(system_prompt, prompt, temperature=0.5)
        
        try:
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            evals = json.loads(response_text.strip())
            if isinstance(evals, list) and len(evals) > 0:
                return evals
        except Exception:
            logger.warning("Failed to parse generated evals JSON, compiling default fallback test cases.")
            
        name = analysis["target_name"]
        return [
            {
                "query": f"run {name} with help option",
                "should_trigger": True,
                "expected_args": ["--help"]
            },
            {
                "query": "what is the capital of France?",
                "should_trigger": False,
                "expected_args": []
            }
        ]
