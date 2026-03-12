#!/usr/bin/env python3
"""
Portable Team Construction Agent (TeamBuilder)
==============================================

This script is designed to be dropped into ANY project directory.
It scans the local context (files, structure) and uses a multi-agent deliberation process
to design the optimal team of AI agents to help you with that specific project.

Usage:
    python team_builder.py --goal "Refactor the API layer" --model-planning "claude-3-5-sonnet-20240620"
    python team_builder.py --path ./my-other-project --goal "Write documentation"

Requirements:
    pip install langgraph langchain langchain-anthropic langchain-google-genai python-dotenv

"""

import os
import sys
import json
import argparse
import glob
import operator
from typing import TypedDict, List, Dict, Any, Literal, Annotated, Union
from pathlib import Path

# Check dependencies
try:
    from dotenv import load_dotenv
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError as e:
    print(f"CRITICAL ERROR: Missing dependencies. Please run:\npip install langgraph langchain langchain-anthropic langchain-google-genai python-dotenv")
    sys.exit(1)

# Load environment variables from .env in the current directory OR parent directories
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True)) 

# --- CONTEXT SCANNING ---

def scan_project_context(root_path: str, max_depth: int = 2) -> str:
    """
    Scans the project directory to build a context string for the agents.
    includes:
    - Directory structure (tree)
    - Content of key files (README.md, requirements.txt, etc.)
    """
    path = Path(root_path).resolve()
    if not path.exists():
        return f"Error: Path {path} does not exist."

    context = [f"Project Root: {path.name}"]
    context.append("--- Directory Structure ---")
    
    # Simple tree walk
    ignore_dirs = {'.git', '__pycache__', 'node_modules', 'venv', '.env', '.idea', '.vscode', 'dist', 'build'}
    ignore_files = {'.DS_Store', 'package-lock.json', 'yarn.lock'}

    structure = []
    for root, dirs, files in os.walk(path):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        
        level = root.replace(str(path), '').count(os.sep)
        if level > max_depth:
            continue
            
        indent = ' ' * 4 * level
        subindent = ' ' * 4 * (level + 1)
        structure.append(f"{indent}{os.path.basename(root)}/")
        for f in files:
            if f not in ignore_files:
                structure.append(f"{subindent}{f}")
                
    context.append("\n".join(structure))
    
    # Read Key Files
    context.append("\n--- Key File Contents ---")
    key_files = ['README.md', 'package.json', 'requirements.txt', 'pyproject.toml', 'Cargo.toml', 'go.mod']
    
    for kf in key_files:
        fpath = path / kf
        if fpath.exists() and fpath.is_file():
            try:
                content = fpath.read_text(encoding='utf-8', errors='ignore')
                # Truncate if too long
                if len(content) > 2000:
                    content = content[:2000] + "... (truncated)"
                context.append(f"\nFile: {kf}\n```\n{content}\n```")
            except Exception as e:
                context.append(f"\nFile: {kf} (Error reading: {e})")

    return "\n".join(context)

# --- CONFIGURATION & DEFAULTS ---
DEFAULT_PLANNING_MODEL = "claude-3-5-sonnet-20240620"
DEFAULT_EXECUTION_MODEL = "gemini-2.0-flash-exp"

def get_available_providers():
    providers = []
    if os.getenv("ANTHROPIC_API_KEY"):
        providers.append("anthropic")
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        providers.append("google")
    return providers

def get_smart_defaults():
    providers = get_available_providers()
    defaults = {
        "planning_provider": "anthropic",
        "planning_model": DEFAULT_PLANNING_MODEL,
        "execution_provider": "google", 
        "execution_model": DEFAULT_EXECUTION_MODEL
    }
    
    if "anthropic" not in providers and "google" in providers:
        print("Notice: Anthropic key not found. Falling back to Gemini for planning.")
        defaults["planning_provider"] = "google"
        defaults["planning_model"] = DEFAULT_EXECUTION_MODEL
    elif "anthropic" not in providers and "google" not in providers:
        print("Warning: No API keys found in environment. Defaulting to anthropic/google (will require .env).")
        
    return defaults

# --- AGENT LOGIC ---

# ... (AgentState class remains) ...

def get_model(provider: str, model_name: str, temp: float = 0.2):
    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
             raise ValueError("ANTHROPIC_API_KEY not found.")
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=temp)
        
    elif provider == "google":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
             raise ValueError("GOOGLE_API_KEY not found.")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key, temperature=temp)
    else:
        raise ValueError(f"Unknown provider: {provider}")

# ... (Nodes need to accept config now) ...
# We need to pass the selected config to the nodes. 
# Since nodes only take 'state', we should probably store the config in the state or use a global (less elegant but easier for this script).
# Let's inject it into the state.

class AgentState(TypedDict):
    user_goal: str
    project_context: str
    config: Dict[str, str] # Added config
    
    # Deliberation Artifacts
    assumptions: Dict[str, Any]
    role_topology: List[Dict[str, Any]]
    critique: Dict[str, Any]
    improved_topology: List[Dict[str, Any]]
    
    # Final Output
    final_report: str
    prompt_package: Dict[str, str]

def node_explorer(state: AgentState):
    print("\n--- [Explorer] Analyzing Context & Goal ---")
    config = state.get("config", {})
    llm = get_model(config.get("planning_provider", "anthropic"), config.get("planning_model", DEFAULT_PLANNING_MODEL))
    
    prompt = f"""
    You are the EXPLORER.
    
    Goal: "{state['user_goal']}"
    
    Project Context:
    {state['project_context']}
    
    1. Analyze the project (Language? Framework? Maturity? Complexity?).
    2. Identify implicit assumptions in the user's goal.
    3. Identify missing information or risks.
    
    Return JSON: {{ "analysis": "...", "assumptions": ["..."], "risks": ["..."] }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        # Robust JSON cleaning
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        data = json.loads(content)
        return {"assumptions": data}
    except Exception as e:
        print(f"Explorer Error: {e}")
        return {"assumptions": {"error": str(e)}}

def node_architect(state: AgentState):
    print("\n--- [Architect] Designing Team Topology ---")
    config = state.get("config", {})
    llm = get_model(config.get("planning_provider", "anthropic"), config.get("planning_model", DEFAULT_PLANNING_MODEL))
    
    prompt = f"""
    You are the ARCHITECT.
    
    Goal: "{state['user_goal']}"
    Explorer Analysis: {json.dumps(state['assumptions'], indent=2)}
    
    Design a Multi-Agent System (Team) to accomplish this goal in this project context.
    Define specific roles.
    
    Return JSON: 
    {{ 
      "roles": [
        {{ "name": "...", "type": "Manager|Worker|Reviewer", "description": "...", "tools_needed": ["..."] }} 
      ],
      "workflow": "Brief description of how they interact"
    }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return {"role_topology": data.get("roles", [])}
    except Exception as e:
        print(f"Architect Error: {e}")
        return {"role_topology": []}

def node_critic(state: AgentState):
    print("\n--- [Critic] Stress Testing Design ---")
    config = state.get("config", {})
    llm = get_model(config.get("planning_provider", "anthropic"), config.get("planning_model", DEFAULT_PLANNING_MODEL))
    
    prompt = f"""
    You are the CRITIC.
    
    Proposed Team: {json.dumps(state['role_topology'], indent=2)}
    Context: {state['project_context'][:1000]}...
    
    Critique this design. Is it overkill? Is it missing a crucial role (e.g., QA, Documentation)?
    Are the tools realistic?
    
    Return JSON: {{ "critique": "...", "missing_roles": ["..."], "superfluous_roles": ["..."], "score": 1-10 }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return {"critique": data}
    except Exception as e:
        print(f"Critic Error: {e}")
        return {"critique": {}}

def node_builder(state: AgentState):
    print("\n--- [Builder] Compiling Final Specs ---")
    config = state.get("config", {})
    llm = get_model(config.get("execution_provider", "google"), config.get("execution_model", DEFAULT_EXECUTION_MODEL))
    
    prompt = f"""
    You are the BUILDER.
    
    Goal: "{state['user_goal']}"
    Team: {json.dumps(state['role_topology'], indent=2)}
    Critique: {json.dumps(state['critique'], indent=2)}
    
    1. Refine the team based on the critique.
    2. Generate a system prompt for EACH agent.
    
    Output a JSON object with this structure:
    {{
        "final_team_report": "Markdown summary of the team design...",
        "agent_prompts": {{
            "AgentName1": "You are AgentName1... System Prompt...",
            "AgentName2": "You are AgentName2... System Prompt..."
        }}
    }}
    """
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.replace("```json", "").replace("```", "").strip()
        data = json.loads(content)
        return {"final_report": data.get("final_team_report", ""), "prompt_package": data.get("agent_prompts", {})}
    except Exception as e:
        print(f"Builder Error: {e}")
        return {"final_report": f"Error generating report: {e}", "prompt_package": {}}

# Graph Construction
def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("explorer", node_explorer)
    workflow.add_node("architect", node_architect)
    workflow.add_node("critic", node_critic)
    workflow.add_node("builder", node_builder)
    
    workflow.set_entry_point("explorer")
    workflow.add_edge("explorer", "architect")
    workflow.add_edge("architect", "critic")
    workflow.add_edge("critic", "builder")
    workflow.add_edge("builder", END)
    
    return workflow.compile()

# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Team Builder Agent")
    parser.add_argument("--goal", type=str, required=True, help="What do you want the team to do?")
    parser.add_argument("--path", type=str, default=".", help="Target project path")
    parser.add_argument("--planning-model", type=str, help="Override planning model")
    parser.add_argument("--planning-provider", type=str, choices=["anthropic", "google"], help="Override planning provider")
    parser.add_argument("--execution-model", type=str, help="Override execution model")
    parser.add_argument("--execution-provider", type=str, choices=["anthropic", "google"], help="Override execution provider")
    
    args = parser.parse_args()
    
    target_path = Path(args.path).resolve()
    print(f"=== Team Builder v1.0 ===")
    print(f"Target: {target_path}")
    print(f"Goal: {args.goal}")
    
    # Configuration
    defaults = get_smart_defaults()
    config = {
        "planning_provider": args.planning_provider or defaults["planning_provider"],
        "planning_model": args.planning_model or defaults["planning_model"],
        "execution_provider": args.execution_provider or defaults["execution_provider"],
        "execution_model": args.execution_model or defaults["execution_model"]
    }
    
    print(f"Config: Planning via {config['planning_provider']} ({config['planning_model']}), Execution via {config['execution_provider']} ({config['execution_model']})")
    
    if not target_path.exists():
        print(f"Error: Path {target_path} does not exist.")
        return

    # Scan Context
    print("Scanning context...")
    context = scan_project_context(str(target_path))
    print(f"Context captured ({len(context)} chars).")
    
    # Run Agent
    app = build_graph()
    initial_state = {
        "user_goal": args.goal,
        "project_context": context,
        "config": config,
        "assumptions": {},
        "role_topology": [],
        "critique": {},
        "improved_topology": [],
        "final_report": "",
        "prompt_package": {}
    }
    
    print("Starting deliberation...")
    final_state = app.invoke(initial_state)
    
    # Save Outputs
    output_dir = target_path / "team_design"
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "design_report.md"
    prompts_path = output_dir / "agent_prompts.json"
    
    report_path.write_text(final_state['final_report'], encoding='utf-8')
    prompts_path.write_text(json.dumps(final_state['prompt_package'], indent=2), encoding='utf-8')
    
    print(f"\nSUCCESS! Team design saved to: {output_dir}")
    print(f"1. Read the report: {report_path}")
    print(f"2. Use the prompts: {prompts_path}")

if __name__ == "__main__":
    main()
