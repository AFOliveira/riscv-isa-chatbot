#!/usr/bin/env python3
"""RISC-V ISA Chatbot - HuggingFace Spaces App
Data from RISC-V Unified Database (UDB): https://github.com/riscv-software-src/riscv-unified-db
"""

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import anthropic
import gradio as gr
import yaml

DATA_DIR = Path(__file__).parent / "data"
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Rate limiting: 20 questions per user (by session)
MAX_QUESTIONS_PER_USER = 20
user_usage = defaultdict(lambda: {"count": 0, "first_use": time.time()})

# Available configs
AVAILABLE_CONFIGS = ["RV64"]  # Add "RV32" when data is generated
CURRENT_CONFIG = "RV64"

_yaml_cache = {}
_inst_index = []
_csr_index = []
_ext_index = []


def load_yaml(path):
    key = str(path)
    if key not in _yaml_cache:
        try:
            with open(path) as f:
                _yaml_cache[key] = yaml.safe_load(f) or {}
        except:
            _yaml_cache[key] = {}
    return _yaml_cache[key]


def build_indices():
    global _inst_index, _csr_index, _ext_index
    if not DATA_DIR.exists():
        return
    
    for yf in (DATA_DIR / "inst").rglob("*.yaml") if (DATA_DIR / "inst").exists() else []:
        try:
            d = load_yaml(yf)
            if d.get("name"):
                db = d.get("definedBy", [])
                if isinstance(db, str): db = [db]
                elif isinstance(db, dict): db = db.get("anyOf", db.get("allOf", []))
                _inst_index.append({
                    "path": str(yf.relative_to(DATA_DIR)), 
                    "name": d["name"], 
                    "long_name": d.get("long_name", ""),
                    "assembly": d.get("assembly", ""),
                    "definedBy": db
                })
        except: pass
    
    for yf in (DATA_DIR / "csr").rglob("*.yaml") if (DATA_DIR / "csr").exists() else []:
        try:
            d = load_yaml(yf)
            if d.get("name"):
                db = d.get("definedBy", [])
                if isinstance(db, str): db = [db]
                _csr_index.append({
                    "path": str(yf.relative_to(DATA_DIR)), 
                    "name": d["name"], 
                    "long_name": d.get("long_name", ""),
                    "address": d.get("address"),
                    "definedBy": db
                })
        except: pass
    
    for yf in (DATA_DIR / "ext").rglob("*.yaml") if (DATA_DIR / "ext").exists() else []:
        try:
            d = load_yaml(yf)
            if d.get("name") and d.get("kind") == "extension":
                _ext_index.append({
                    "path": str(yf.relative_to(DATA_DIR)), 
                    "name": d["name"], 
                    "long_name": d.get("long_name", ""),
                    "description": d.get("description", "")[:200]
                })
        except: pass


def search_instructions(term="", extension="", limit=20):
    results = []
    for i in _inst_index:
        if term and term.lower() not in i["name"].lower() and term.lower() not in i.get("long_name", "").lower():
            continue
        if extension and extension not in i.get("definedBy", []):
            continue
        results.append({"name": i["name"], "long_name": i["long_name"], "assembly": i["assembly"], "definedBy": i["definedBy"]})
        if len(results) >= limit:
            break
    return {"count": len(results), "instructions": results}

def search_csrs(term="", extension="", limit=20):
    results = []
    for c in _csr_index:
        if term and term.lower() not in c["name"].lower() and term.lower() not in c.get("long_name", "").lower():
            continue
        if extension and extension not in c.get("definedBy", []):
            continue
        results.append({"name": c["name"], "long_name": c["long_name"], "address": c["address"]})
        if len(results) >= limit:
            break
    return {"count": len(results), "csrs": results}

def list_extensions():
    return {"count": len(_ext_index), "extensions": [{"name": e["name"], "long_name": e["long_name"]} for e in _ext_index]}

def get_extension_details(name):
    ext = next((e for e in _ext_index if e["name"] == name), None)
    if not ext: return {"error": f"Extension '{name}' not found"}
    insts = [{"name": i["name"], "assembly": i["assembly"]} for i in _inst_index if name in i.get("definedBy", [])][:30]
    csrs = [{"name": c["name"], "address": c["address"]} for c in _csr_index if name in c.get("definedBy", [])][:20]
    return {"extension": ext, "instructions": {"count": len(insts), "items": insts}, "csrs": {"count": len(csrs), "items": csrs}}

def get_instruction_details(name):
    inst = next((i for i in _inst_index if i["name"] == name), None)
    if not inst: return {"error": f"Instruction '{name}' not found"}
    return {"instruction": load_yaml(DATA_DIR / inst["path"])}

def get_csr_details(name):
    csr = next((c for c in _csr_index if c["name"] == name), None)
    if not csr: return {"error": f"CSR '{name}' not found"}
    return {"csr": load_yaml(DATA_DIR / csr["path"])}

def get_stats():
    return {"config": CURRENT_CONFIG, "instructions": len(_inst_index), "csrs": len(_csr_index), "extensions": len(_ext_index)}


TOOLS = [
    {"name": "search_instructions", "description": "Search RISC-V instructions by name or filter by extension", 
     "input_schema": {"type": "object", "properties": {"term": {"type": "string", "description": "Search term"}, "extension": {"type": "string", "description": "Filter by extension (e.g. M, A, F, V)"}}}},
    {"name": "search_csrs", "description": "Search RISC-V CSRs by name or extension", 
     "input_schema": {"type": "object", "properties": {"term": {"type": "string"}, "extension": {"type": "string"}}}},
    {"name": "list_extensions", "description": "List all RISC-V extensions", 
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_extension_details", "description": "Get detailed info about a RISC-V extension including its instructions and CSRs", 
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Extension name (e.g. M, A, V, Zba)"}}, "required": ["name"]}},
    {"name": "get_instruction_details", "description": "Get complete details about a specific instruction including encoding and operation", 
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "Instruction name (e.g. add, mul, lw)"}}, "required": ["name"]}},
    {"name": "get_csr_details", "description": "Get complete details about a specific CSR including fields", 
     "input_schema": {"type": "object", "properties": {"name": {"type": "string", "description": "CSR name (e.g. mstatus, mcause)"}}, "required": ["name"]}},
    {"name": "get_stats", "description": "Get database statistics", 
     "input_schema": {"type": "object", "properties": {}}},
]

TOOL_FNS = {
    "search_instructions": search_instructions, 
    "search_csrs": search_csrs, 
    "list_extensions": list_extensions,
    "get_extension_details": get_extension_details, 
    "get_instruction_details": get_instruction_details,
    "get_csr_details": get_csr_details, 
    "get_stats": get_stats
}

SYSTEM = """You are a helpful RISC-V ISA assistant with access to a comprehensive database of instructions, CSRs, and extensions from the RISC-V Unified Database (UDB).

Use the available tools to look up accurate information. Provide clear, technical explanations with relevant details like encodings, assembly syntax, and extension dependencies."""


def ask(question, config, request: gr.Request):
    if not API_KEY:
        return "API key not configured."
    if not _inst_index:
        return "Data not loaded."
    
    # Rate limiting by IP
    user_id = request.client.host if request else "unknown"
    user = user_usage[user_id]
    
    if user["count"] >= MAX_QUESTIONS_PER_USER:
        return f"You've reached the limit of {MAX_QUESTIONS_PER_USER} questions. Thank you for trying the RISC-V ISA Chatbot!"
    
    user["count"] += 1
    remaining = MAX_QUESTIONS_PER_USER - user["count"]
    
    try:
        client = anthropic.Anthropic(api_key=API_KEY)
        messages = [{"role": "user", "content": f"[Using {config} profile] {question}"}]
        
        response = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=4096, system=SYSTEM, tools=TOOLS, messages=messages)
        
        while response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = TOOL_FNS.get(block.name, lambda **x: {"error": "unknown"})(**block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result, default=str)})
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            response = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=4096, system=SYSTEM, tools=TOOLS, messages=messages)
        
        answer = "".join(b.text for b in response.content if hasattr(b, "text"))
        return f"{answer}\n\n---\n*Profile: {config} | Questions remaining: {remaining}/{MAX_QUESTIONS_PER_USER}*"
    
    except anthropic.AuthenticationError:
        user["count"] -= 1
        return "API key error."
    except Exception as e:
        user["count"] -= 1
        return f"Error: {e}"


print("Loading RISC-V data...")
build_indices()
stats = get_stats()
print(f"Loaded: {stats['instructions']} instructions, {stats['csrs']} CSRs, {stats['extensions']} extensions")

with gr.Blocks(title="RISC-V ISA Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
# RISC-V ISA Chatbot

Ask questions about RISC-V instructions, CSRs, and extensions. Powered by Claude Haiku 4.5.

**Data Source:** [RISC-V Unified Database (UDB)](https://github.com/riscv-software-src/riscv-unified-db) - the official machine-readable RISC-V ISA specification.

**Database:** {stats['instructions']} instructions | {stats['csrs']} CSRs | {stats['extensions']} extensions

**Limit:** {MAX_QUESTIONS_PER_USER} questions per user
""")
    
    with gr.Row():
        config = gr.Radio(
            choices=AVAILABLE_CONFIGS,
            value=CURRENT_CONFIG,
            label="ISA Profile",
            info="Select RISC-V base architecture"
        )
    
    question = gr.Textbox(label="Question", placeholder="Ask about RISC-V instructions, CSRs, or extensions...", lines=2)
    submit = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=15, interactive=False)
    
    gr.Examples(
        examples=[
            ["What instructions are in the M extension?"],
            ["Explain the mstatus CSR and its fields"],
            ["How does the ADD instruction work?"],
            ["List all vector extensions"],
        ],
        inputs=question,
    )
    
    submit.click(fn=ask, inputs=[question, config], outputs=answer)
    question.submit(fn=ask, inputs=[question, config], outputs=answer)
    
    gr.Markdown("""
---
*Data from [RISC-V Unified Database](https://github.com/riscv-software-src/riscv-unified-db) by RISC-V International*
""")

if __name__ == "__main__":
    demo.launch()
