import sys
import os
import shutil
import atexit
import signal
import json
import logging
import readline
from insomnai_agent import InsomnAIAgent

# Configure persistent history and tab completion for slash commands
history_file = os.path.expanduser("~/.insomnai_chat_history")
try:
    readline.read_history_file(history_file)
except FileNotFoundError:
    pass
atexit.register(readline.write_history_file, history_file)

commands = ["/sleep", "/master", "/undo", "/redo", "/status", "/prune", "/skillify", "/exit"]
def completer(text, state):
    options = [cmd for cmd in commands if cmd.startswith(text)]
    if state < len(options):
        return options[state]
    return None

readline.set_completer(completer)
readline.parse_and_bind("tab: complete")

agent = None

def setup_terminal():
    width, height = shutil.get_terminal_size()
    # Scroll margin: top=1, bottom=height-9
    scroll_height = height - 9
    sys.stdout.write(f"\033[1;{scroll_height}r")
    sys.stdout.write(f"\033[{scroll_height};1H")
    sys.stdout.flush()

def restore_terminal():
    sys.stdout.write("\033[r")  # Reset scrolling region
    sys.stdout.write("\033[?25h")  # Show cursor
    sys.stdout.flush()
    print("\nTerminal restored. Goodbye!")

def on_resize(signum, frame):
    setup_terminal()
    draw_dashboard()

def draw_dashboard():
    if agent is None:
        return
    width, height = shutil.get_terminal_size()
    # Save cursor position
    sys.stdout.write("\033[s")
    
    # Clear the dashboard lines (from height-9 to height-1)
    for i in range(height - 9, height + 1):
        sys.stdout.write(f"\033[{i};1H\033[K")
        
    # Draw double border line
    sys.stdout.write(f"\033[{height-9};1H" + "\033[36m" + "=" * width + "\033[0m")
    
    # Fetch agent states
    adrenaline = agent.endocrine.adrenaline
    serotonin = agent.endocrine.serotonin
    dopamine = agent.endocrine.dopamine
    acetylcholine = agent.endocrine.acetylcholine
    
    # Generate progress bar helper
    def make_bar(val, color_code):
        filled = int(max(0.0, min(1.0, val)) * 10)
        empty = 10 - filled
        bar = "█" * filled + "░" * empty
        return f"{color_code}[{bar}] {val:.2f}\033[0m"
        
    adr_bar = make_bar(adrenaline, "\033[31m")  # Red
    ser_bar = make_bar(serotonin, "\033[36m")   # Cyan/Blue
    dop_bar = make_bar(dopamine, "\033[32m")    # Green
    ace_bar = make_bar(acetylcholine, "\033[33m") # Yellow
    
    # Row 1: Hormones
    sys.stdout.write(f"\033[{height-8};2H\033[1mHORMONES:\033[0m")
    sys.stdout.write(f"\033[{height-8};15HAdrenaline:     {adr_bar}")
    sys.stdout.write(f"\033[{height-8};50HSerotonin:      {ser_bar}")
    
    # Row 2: Hormones cont.
    sys.stdout.write(f"\033[{height-7};15HDopamine:        {dop_bar}")
    sys.stdout.write(f"\033[{height-7};50HAcetylcholine:  {ace_bar}")
    
    # Row 3: Metrics
    active_adapter = agent.active_model.active_adapter
    try:
        lora_config = agent.active_model.peft_config[active_adapter]
        rank = f"r={lora_config.r}"
    except Exception:
        rank = "N/A"
        
    dissonance_val = 0.0
    if agent.memory.episodic_log:
        dissonance_val = agent.memory.episodic_log[-1].get("dissonance", 0.0)
        
    sys.stdout.write(f"\033[{height-6};2H\033[1mMETRICS:\033[0m")
    sys.stdout.write(f"\033[{height-6};15HActive Adapter:  \033[35m{active_adapter} ({rank})\033[0m")
    sys.stdout.write(f"\033[{height-6};50HDissonance:     \033[33m{dissonance_val:.4f}\033[0m")
    
    # Row 4: Memory status
    shadow_size = len(agent.memory.shadow_log)
    episodic_size = len(agent.memory.episodic_log)
    ptr = agent.metadata["pointer"]
    max_ver = len(agent.metadata["versions"]) - 1
    checkpoint_status = f"v{ptr}/{max_ver}" if max_ver >= 0 else "None"
    
    sys.stdout.write(f"\033[{height-5};15HShadow Log size: \033[31m{shadow_size}\033[0m")
    sys.stdout.write(f"\033[{height-5};50HEpisodic Logs:  \033[32m{episodic_size}\033[0m")
    
    # Row 5: Version Pointer / Master status
    master_status = "\033[32mACTIVE\033[0m" if agent.master_active else "\033[31mOFF\033[0m"
    sys.stdout.write(f"\033[{height-4};15HCheckpoint:      \033[36m{checkpoint_status}\033[0m")
    sys.stdout.write(f"\033[{height-4};50HMaster Teacher: {master_status}")
    
    # Row 6: MoLA weights
    w_cut, w_gra, w_lex = getattr(agent, "mola_weights", [1.0, 1.0, 1.0])
    sys.stdout.write(f"\033[{height-3};15HMoLA Weights:    \033[32mcult: {w_cut:.1f}\033[0m | \033[33mgram: {w_gra:.1f}\033[0m | \033[35mlexi: {w_lex:.1f}\033[0m")
    
    # Border bottom
    sys.stdout.write(f"\033[{height-2};1H" + "\033[36m" + "=" * width + "\033[0m")
    
    # Restore cursor position
    sys.stdout.write("\033[u")
    sys.stdout.flush()

def main():
    global agent
    os.system("clear")
    print("\033[1;36m======================================================================\033[0m")
    print("\033[1;36m                  InsomnAI 2.0 Terminal Chat Dashboard                \033[0m")
    print("\033[1;36m======================================================================\033[0m")
    print("Loading local weights and models... Please wait.")
    
    agent = InsomnAIAgent(model_id="thirdeyeai/Qwen2.5-1.5B-Instruct-uncensored")
    
    # Register exit handlers and signals
    atexit.register(restore_terminal)
    signal.signal(signal.SIGWINCH, on_resize)
    
    setup_terminal()
    draw_dashboard()
    
    print("\n\033[32mSystem Loaded! Talk to InsomnAI. Available commands:\033[0m")
    print("  \033[33m/sleep\033[0m       - Trigger offline consolidation & SFT sleep cycle")
    print("  \033[33m/master\033[0m      - Toggle Master Teacher supervision")
    print("  \033[33m/undo\033[0m        - Undo last checkpoint weights")
    print("  \033[33m/redo\033[0m        - Redo next checkpoint weights")
    print("  \033[33m/status\033[0m      - Full state details dump")
    print("  \033[33m/prune\033[0m       - Synaptic pruning: SVD rank compression (r=16 -> r=4)")
    print("  \033[33m/skillify\033[0m    - Autodetect & generate skill for CLI, API, library, or URL (e.g. /skillify tar)")
    print("  \033[33m/exit\033[0m        - Exit program")
    print("-" * 50)
    
    while True:
        try:
            prompt = input("\033[1;32mUser>\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            break
            
        if not prompt:
            continue
            
        if prompt.lower() == "/exit":
            break
        elif prompt.lower() == "/master":
            agent.master_active = not agent.master_active
            print(f"\n\033[1;33m[SYSTEM] Master Teacher is now {'ACTIVE' if agent.master_active else 'OFF'}.\033[0m")
            draw_dashboard()
            continue
        elif prompt.lower() == "/undo":
            agent.undo()
            print("\n\033[1;33m[SYSTEM] Restored previous checkpoint version.\033[0m")
            draw_dashboard()
            continue
        elif prompt.lower() == "/redo":
            agent.redo()
            print("\n\033[1;33m[SYSTEM] Restored next checkpoint version.\033[0m")
            draw_dashboard()
            continue
        elif prompt.lower() == "/status":
            print(f"\n\033[1;33m[STATUS DUMP]\033[0m")
            print(f"  Model ID: {agent.active_model.name_or_path}")
            print(f"  Adrenaline: {agent.endocrine.adrenaline:.4f}")
            print(f"  Serotonin: {agent.endocrine.serotonin:.4f}")
            print(f"  Dopamine: {agent.endocrine.dopamine:.4f}")
            print(f"  Acetylcholine: {agent.endocrine.acetylcholine:.4f}")
            print(f"  Checkpoints pointers: {agent.metadata}")
            print(f"  Episodic logs size: {len(agent.memory.episodic_log)}")
            print(f"  Shadow log size: {len(agent.memory.shadow_log)}")
            continue
        elif prompt.lower() == "/prune":
            print("\n\033[1;35m[SYSTEM] Initiating Synaptic Pruning SVD rank-compression...\033[0m")
            agent._prune_adapter_rank("cultural", target_rank=4)
            print("\033[1;32m[SYSTEM] Synaptic Pruning completed successfully! Cultural adapter rank is now r=4.\033[0m")
            draw_dashboard()
            continue
        elif prompt.lower().startswith("/skillify "):
            target = prompt[10:].strip()
            if not target:
                print("\n\033[1;31m[SYSTEM] Error: Please specify a target command, e.g. /skillify tar\033[0m")
                continue
            print(f"\n\033[1;35m[SYSTEM] Initiating local Skillification pipeline for '{target}'...\033[0m")
            try:
                from skill_manager import SkillManager
                manager = SkillManager(
                    endpoint=agent.master_teacher.endpoint,
                    model_name=agent.master_teacher.model_name
                )
                res = manager.skillify(target)
                if res.get("status") == "success":
                    print(f"\033[1;32m[SYSTEM] Skillification completed successfully! Skill saved at {res['path']}\033[0m")
                    print(f"         Capabilities: {res['capabilities']}, Test queries: {res['eval_queries']}")
                else:
                    print(f"\033[1;31m[SYSTEM] Skillification failed: {res.get('message', 'unknown error')}\033[0m")
            except Exception as e:
                print(f"\033[1;31m[SYSTEM] Error during skillification: {e}\033[0m")
            draw_dashboard()
            continue
        elif prompt.lower() == "/sleep":
            print("\n\033[1;35m[SYSTEM] Sleep sequence initiated. Consolidation loop active...\033[0m")
            agent.trigger_sleep_cycle()
            print("\033[1;32m[SYSTEM] Sleep sequence completed. Agent woke up refreshed.\033[0m")
            draw_dashboard()
            continue
            
        # Chat interaction
        sys.stdout.write("\033[?25l")  # Hide cursor while generating
        sys.stdout.flush()
        
        response = agent.chat(prompt)
        
        sys.stdout.write("\033[?25h")  # Show cursor
        sys.stdout.flush()
        
        # Check if the response contains a valid JSON tool call
        is_tool_call = False
        parsed_tool = None
        if response.strip().startswith("{") and "tool" in response:
            try:
                parsed_tool = json.loads(response)
                is_tool_call = True
            except Exception:
                pass
                
        if is_tool_call and parsed_tool:
            tool_name = parsed_tool.get("tool")
            args = parsed_tool.get("args", [])
            print(f"\n\033[1;33m[TOOL CALL] InsomnAI requested: {tool_name} with args {args}\033[0m")
            
            tool_output = "Error: Tool execution failed."
            if tool_name == "calculator" and len(args) >= 2:
                try:
                    res = float(args[0]) * float(args[1])
                    tool_output = f"Calculation result: {res}"
                except Exception as e:
                    tool_output = f"Calculator error: {e}"
            elif tool_name == "web_search" and len(args) >= 1:
                query = args[0]
                print(f"\033[1;33m[TOOL EXECUTION] Searching web for '{query}'...\033[0m")
                # Query free weather API (wttr.in) if weather-related, else mock search
                if "weather" in query.lower() or "időjárás" in query.lower():
                    try:
                        import requests
                        from datetime import datetime
                        # Fetch clean text-only weather format
                        r = requests.get(f"https://wttr.in/Budapest?format=3", timeout=10)
                        if r.status_code == 200:
                            weather_text = r.text.strip()
                            now = datetime.now()
                            # Determine time of day (napszak) and adjust emojis
                            if now.hour >= 20 or now.hour < 5:
                                napszak = "éjszaka"
                                weather_text = weather_text.replace("☀️", "🌙").replace("🌤️", "☁️").replace("⛅", "☁️")
                            elif now.hour >= 18:
                                napszak = "este"
                                weather_text = weather_text.replace("☀️", "🌙")
                            elif now.hour >= 12:
                                napszak = "délután"
                            else:
                                napszak = "délelőtt/reggel"
                                
                            tool_output = f"Live weather data for Budapest (Napszak: {napszak}, Helyi idő: {now.strftime('%H:%M')}): {weather_text}"
                        else:
                            tool_output = "Weather search error: API returned non-200 status."
                    except Exception as e:
                        tool_output = f"Web search error: {e}"
                else:
                    tool_output = f"Search results for '{query}': Found general info matching topic."
            else:
                skills_dir = os.path.join(".agents", "skills", tool_name)
                if os.path.isdir(skills_dir):
                    # Determine target type from analysis.json if it exists
                    target_type = "cli"
                    analysis_file = os.path.join(skills_dir, "analysis.json")
                    if os.path.isfile(analysis_file):
                        try:
                            with open(analysis_file, "r", encoding="utf-8") as f:
                                analysis_data = json.load(f)
                            target_type = analysis_data.get("target_type", "cli")
                        except Exception:
                            pass
                            
                    if target_type == "api":
                        method = "GET"
                        url = None
                        data = None
                        
                        for arg in args:
                            arg_str = str(arg)
                            if arg_str.upper() in ["GET", "POST", "PUT", "DELETE"]:
                                method = arg_str.upper()
                            elif arg_str.startswith("http://") or arg_str.startswith("https://"):
                                url = arg_str
                            elif arg_str.strip().startswith("{") and arg_str.strip().endswith("}"):
                                try:
                                    data = json.loads(arg_str)
                                except Exception:
                                    data = arg_str
                                    
                        if not url and os.path.isfile(analysis_file):
                            try:
                                with open(analysis_file, "r", encoding="utf-8") as f:
                                    analysis_data = json.load(f)
                                raw_help = analysis_data.get("raw_help", "")
                                if raw_help.startswith("http://") or raw_help.startswith("https://"):
                                    url = raw_help
                            except Exception:
                                pass
                                
                        if url:
                            print(f"\033[1;33m[TOOL EXECUTION] Sending HTTP {method} request to: {url}...\033[0m")
                            try:
                                import requests
                                if method == "GET":
                                    resp = requests.get(url, timeout=10)
                                elif method == "POST":
                                    resp = requests.post(url, json=data if isinstance(data, dict) else None, data=data if not isinstance(data, dict) else None, timeout=10)
                                elif method == "PUT":
                                    resp = requests.put(url, json=data if isinstance(data, dict) else None, data=data if not isinstance(data, dict) else None, timeout=10)
                                elif method == "DELETE":
                                    resp = requests.delete(url, timeout=10)
                                    
                                tool_output = f"HTTP {resp.status_code} Response:\n{resp.text[:5000]}"
                            except Exception as e:
                                tool_output = f"HTTP request failed: {e}"
                        else:
                            tool_output = "Error: No valid URL specified for API tool call."
                    else:
                        print(f"\033[1;33m[TOOL EXECUTION] Running custom command: {tool_name} {' '.join(str(a) for a in args)}...\033[0m")
                        try:
                            import subprocess
                            res = subprocess.run([tool_name] + [str(a) for a in args], capture_output=True, text=True, timeout=10)
                            tool_output = (res.stdout or "") + "\n" + (res.stderr or "")
                            if not tool_output.strip():
                                tool_output = "Command executed successfully with no output."
                        except Exception as e:
                            tool_output = f"Execution error: {e}"
                else:
                    tool_output = f"Error: Tool '{tool_name}' is not registered."
            
            print(f"\033[1;32m[TOOL OUTPUT] Return: {tool_output}\033[0m\n")
            
            # Feed the output back to the agent as a system prompt message to formulate final answer
            sys.stdout.write("\033[?25l")
            sys.stdout.flush()
            final_prompt = (
                f"System notice: Tool execution has completed. Tool Output:\n{tool_output}\n"
                f"You MUST NOT output another JSON tool call. Based on this tool output, answer the user's original query "
                f"'{prompt}' in a clean, natural conversational sentence."
            )
            final_response = agent.chat(final_prompt)
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()
            
            # Apply Master alignment to the final conversational answer
            if agent.master_active:
                intercepted, corrected = agent.master_teacher.evaluate_interaction(final_prompt, final_response)
                if intercepted:
                    # Do NOT log transient tool execution feedback to shadow_log for parametric SFT training
                    if not final_prompt.startswith("System notice: Tool execution"):
                        agent.memory.shadow_log.append({
                            "prompt": final_prompt,
                            "raw_truth": final_response,
                            "target": corrected,
                            "dissonance_score": 1.0
                        })
                        agent.memory.save_shadow_log()
                    final_response = corrected
                    
            print(f"\033[1;35mInsomnAI>\033[0m {final_response}\n")
        else:
            print(f"\n\033[1;35mInsomnAI>\033[0m {response}\n")
            
        draw_dashboard()

if __name__ == "__main__":
    main()
