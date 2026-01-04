#!/usr/bin/env python3
"""RISC-V ISA Chatbot - HuggingFace Spaces App"""

import json
import os
from pathlib import Path

import anthropic
import gradio as gr
import yaml

DATA_DIR = Path(__file__).parent / "data"
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
                _inst_index.append({"path": str(yf.relative_to(DATA_DIR)), "name": d["name"], "long_name": d.get("long_name", ""), "definedBy": db})
        except: pass
    
    for yf in (DATA_DIR / "csr").rglob("*.yaml") if (DATA_DIR / "csr").exists() else []:
        try:
            d = load_yaml(yf)
            if d.get("name"):
                db = d.get("definedBy", [])
                if isinstance(db, str): db = [db]
                _csr_index.append({"path": str(yf.relative_to(DATA_DIR)), "name": d["name"], "long_name": d.get("long_name", ""), "definedBy": db})
        except: pass
    
    for yf in (DATA_DIR / "ext").rglob("*.yaml") if (DATA_DIR / "ext").exists() else []:
        try:
            d = load_yaml(yf)
            if d.get("name") and d.get("kind") == "extension":
                _ext_index.append({"path": str(yf.relative_to(DATA_DIR)), "name": d["name"], "long_name": d.get("long_name", "")})
        except: pass


def search_instructions(term="", extension="", limit=20):
    results = [i for i in _inst_index if (not term or term.lower() in i["name"].lower()) and (not extension or extension in i.get("definedBy", []))]
    return {"count": len(results[:limit]), "instructions": results[:limit]}

def search_csrs(term="", extension="", limit=20):
    results = [c for c in _csr_index if (not term or term.lower() in c["name"].lower()) and (not extension or extension in c.get("definedBy", []))]
    return {"count": len(results[:limit]), "csrs": results[:limit]}

def list_extensions():
    return {"count": len(_ext_index), "extensions": _ext_index}

def get_extension_details(name):
    ext = next((e for e in _ext_index if e["name"] == name), None)
    if not ext: return {"error": f"Extension '{name}' not found"}
    return {"extension": ext, "instructions": [i for i in _inst_index if name in i.get("definedBy", [])][:30], "csrs": [c for c in _csr_index if name in c.get("definedBy", [])][:30]}

def get_instruction_details(name):
    inst = next((i for i in _inst_index if i["name"] == name), None)
    if not inst: return {"error": f"Instruction '{name}' not found"}
    return {"instruction": load_yaml(DATA_DIR / inst["path"])}

def get_csr_details(name):
    csr = next((c for c in _csr_index if c["name"] == name), None)
    if not csr: return {"error": f"CSR '{name}' not found"}
    return {"csr": load_yaml(DATA_DIR / csr["path"])}

def get_stats():
    return {"instructions": len(_inst_index), "csrs": len(_csr_index), "extensions": len(_ext_index)}


TOOLS = [
    {"name": "search_instructions", "description": "Search RISC-V instructions", "input_schema": {"type": "object", "properties": {"term": {"type": "string"}, "extension": {"type": "string"}}}},
    {"name": "search_csrs", "description": "Search RISC-V CSRs", "input_schema": {"type": "object", "properties": {"term": {"type": "string"}, "extension": {"type": "string"}}}},
    {"name": "list_extensions", "description": "List RISC-V extensions", "input_schema": {"type": "object", "properties": {}}},
    {"name": "get_extension_details", "description": "Get extension details", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "get_instruction_details", "description": "Get instruction details", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "get_csr_details", "description": "Get CSR details", "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "get_stats", "description": "Get database stats", "input_schema": {"type": "object", "properties": {}}},
]

TOOL_FNS = {"search_instructions": search_instructions, "search_csrs": search_csrs, "list_extensions": list_extensions,
            "get_extension_details": get_extension_details, "get_instruction_details": get_instruction_details,
            "get_csr_details": get_csr_details, "get_stats": get_stats}

SYSTEM = "You are a RISC-V ISA assistant. Use tools to look up accurate information about instructions, CSRs, and extensions."


def ask(question, api_key):
    if not api_key:
        return "Please provide your Anthropic API key."
    if not _inst_index:
        return "Data not loaded."
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        messages = [{"role": "user", "content": question}]
        
        response = client.messages.create(model="claude-haiku-4-20250514", max_tokens=4096, system=SYSTEM, tools=TOOLS, messages=messages)
        
        while response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = TOOL_FNS.get(block.name, lambda **x: {"error": "unknown"})(**block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result, default=str)})
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            response = client.messages.create(model="claude-haiku-4-20250514", max_tokens=4096, system=SYSTEM, tools=TOOLS, messages=messages)
        
        return "".join(b.text for b in response.content if hasattr(b, "text"))
    except anthropic.AuthenticationError:
        return "Invalid API key."
    except Exception as e:
        return f"Error: {e}"


print("Loading RISC-V data...")
build_indices()
stats = get_stats()
print(f"Loaded: {stats['instructions']} instructions, {stats['csrs']} CSRs, {stats['extensions']} extensions")

demo = gr.Interface(
    fn=ask,
    inputs=[
        gr.Textbox(label="Question", placeholder="Ask about RISC-V instructions, CSRs, or extensions...", lines=2),
        gr.Textbox(label="Anthropic API Key", type="password", value=os.environ.get("ANTHROPIC_API_KEY", "")),
    ],
    outputs=gr.Textbox(label="Answer", lines=10),
    title="RISC-V ISA Chatbot",
    description=f"Ask questions about RISC-V. Database: {stats['instructions']} instructions, {stats['csrs']} CSRs, {stats['extensions']} extensions.",
    examples=[
        ["What instructions are in the M extension?", ""],
        ["Explain the mstatus CSR", ""],
        ["How does the ADD instruction work?", ""],
    ],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
