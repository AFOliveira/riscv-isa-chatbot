#!/usr/bin/env python3
"""
RISC-V ISA Chatbot - HuggingFace Spaces App
A chatbot that answers questions about RISC-V using Claude Haiku with tool use.
"""

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import anthropic
import gradio as gr
import yaml

# Data paths
DATA_DIR = Path(__file__).parent / "data"
INST_DIR = DATA_DIR / "inst"
CSR_DIR = DATA_DIR / "csr"
EXT_DIR = DATA_DIR / "ext"

# Caches
_yaml_cache: dict[str, Any] = {}
_inst_index: list[dict] = []
_csr_index: list[dict] = []
_ext_index: list[dict] = []


def load_yaml(path: Path) -> dict:
    """Load and cache a YAML file."""
    key = str(path)
    if key in _yaml_cache:
        return _yaml_cache[key]
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _yaml_cache[key] = data
        return data
    except Exception:
        return {}


def build_indices():
    """Build search indices from YAML files."""
    global _inst_index, _csr_index, _ext_index

    if not DATA_DIR.exists():
        return

    # Build instruction index
    if INST_DIR.exists():
        for yaml_file in INST_DIR.rglob("*.yaml"):
            try:
                data = load_yaml(yaml_file)
                if data.get("name"):
                    defined_by = data.get("definedBy", [])
                    if isinstance(defined_by, str):
                        defined_by = [defined_by]
                    elif isinstance(defined_by, dict):
                        defined_by = defined_by.get("anyOf", defined_by.get("allOf", []))

                    _inst_index.append({
                        "path": str(yaml_file.relative_to(DATA_DIR)),
                        "name": data.get("name"),
                        "long_name": data.get("long_name", ""),
                        "assembly": data.get("assembly", ""),
                        "encoding": data.get("encoding", {}).get("match", "") if isinstance(data.get("encoding"), dict) else "",
                        "description": data.get("description", ""),
                        "definedBy": defined_by,
                    })
            except Exception:
                pass

    # Build CSR index
    if CSR_DIR.exists():
        for yaml_file in CSR_DIR.rglob("*.yaml"):
            try:
                data = load_yaml(yaml_file)
                if data.get("name"):
                    defined_by = data.get("definedBy", [])
                    if isinstance(defined_by, str):
                        defined_by = [defined_by]

                    _csr_index.append({
                        "path": str(yaml_file.relative_to(DATA_DIR)),
                        "name": data.get("name"),
                        "long_name": data.get("long_name", ""),
                        "address": data.get("address"),
                        "priv_mode": data.get("priv_mode", ""),
                        "description": data.get("description", ""),
                        "definedBy": defined_by,
                    })
            except Exception:
                pass

    # Build extension index
    if EXT_DIR.exists():
        for yaml_file in EXT_DIR.rglob("*.yaml"):
            try:
                data = load_yaml(yaml_file)
                if data.get("name") and data.get("kind") == "extension":
                    _ext_index.append({
                        "path": str(yaml_file.relative_to(DATA_DIR)),
                        "name": data.get("name"),
                        "long_name": data.get("long_name", ""),
                        "description": data.get("description", ""),
                        "version": data.get("version"),
                    })
            except Exception:
                pass


# Tool implementations
def search_instructions(term: str = "", extension: str = "", limit: int = 20) -> dict:
    """Search RISC-V instructions by name or extension."""
    results = []
    term_lower = term.lower() if term else ""

    for inst in _inst_index:
        # Match by term
        if term_lower:
            if term_lower not in inst["name"].lower() and term_lower not in inst.get("long_name", "").lower():
                continue

        # Match by extension
        if extension:
            if extension not in inst.get("definedBy", []):
                continue

        results.append(inst)
        if len(results) >= limit:
            break

    return {"count": len(results), "instructions": results}


def search_csrs(term: str = "", extension: str = "", limit: int = 20) -> dict:
    """Search RISC-V CSRs (Control and Status Registers)."""
    results = []
    term_lower = term.lower() if term else ""

    for csr in _csr_index:
        if term_lower:
            if term_lower not in csr["name"].lower() and term_lower not in csr.get("long_name", "").lower():
                continue

        if extension:
            if extension not in csr.get("definedBy", []):
                continue

        results.append(csr)
        if len(results) >= limit:
            break

    return {"count": len(results), "csrs": results}


def list_extensions() -> dict:
    """List all RISC-V extensions."""
    return {"count": len(_ext_index), "extensions": _ext_index}


def get_extension_details(name: str) -> dict:
    """Get detailed info about a specific extension including its instructions and CSRs."""
    # Find extension
    ext_info = None
    for ext in _ext_index:
        if ext["name"] == name:
            ext_info = ext
            break

    if not ext_info:
        return {"error": f"Extension '{name}' not found"}

    # Find instructions for this extension
    instructions = [inst for inst in _inst_index if name in inst.get("definedBy", [])]

    # Find CSRs for this extension
    csrs = [csr for csr in _csr_index if name in csr.get("definedBy", [])]

    return {
        "extension": ext_info,
        "instructions": {"count": len(instructions), "items": instructions[:50]},
        "csrs": {"count": len(csrs), "items": csrs[:50]},
    }


def get_instruction_details(name: str) -> dict:
    """Get full details about a specific instruction."""
    for inst in _inst_index:
        if inst["name"] == name:
            # Load full YAML data
            yaml_path = DATA_DIR / inst["path"]
            full_data = load_yaml(yaml_path)
            return {"instruction": full_data}

    return {"error": f"Instruction '{name}' not found"}


def get_csr_details(name: str) -> dict:
    """Get full details about a specific CSR."""
    for csr in _csr_index:
        if csr["name"] == name:
            yaml_path = DATA_DIR / csr["path"]
            full_data = load_yaml(yaml_path)
            return {"csr": full_data}

    return {"error": f"CSR '{name}' not found"}


def get_stats() -> dict:
    """Get database statistics."""
    return {
        "instructions": len(_inst_index),
        "csrs": len(_csr_index),
        "extensions": len(_ext_index),
    }


# Claude tools definition
TOOLS = [
    {
        "name": "search_instructions",
        "description": "Search RISC-V instructions by name keyword or filter by extension. Use this to find instructions like ADD, MUL, LOAD, etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "Search term to match in instruction name (e.g., 'add', 'mul', 'load')"
                },
                "extension": {
                    "type": "string",
                    "description": "Filter by extension name (e.g., 'M', 'A', 'F', 'V', 'Zba')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 20)",
                    "default": 20
                }
            }
        }
    },
    {
        "name": "search_csrs",
        "description": "Search RISC-V Control and Status Registers (CSRs) by name or extension.",
        "input_schema": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "Search term to match in CSR name (e.g., 'status', 'cause', 'epc')"
                },
                "extension": {
                    "type": "string",
                    "description": "Filter by extension name"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return",
                    "default": 20
                }
            }
        }
    },
    {
        "name": "list_extensions",
        "description": "List all available RISC-V extensions (like M, A, F, D, V, C, Zba, Zbb, etc.)",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_extension_details",
        "description": "Get detailed information about a specific RISC-V extension, including all its instructions and CSRs.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Extension name (e.g., 'M', 'A', 'V', 'Zba')"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_instruction_details",
        "description": "Get complete details about a specific instruction including encoding, assembly format, and operation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact instruction name (e.g., 'add', 'mul', 'lw')"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_csr_details",
        "description": "Get complete details about a specific CSR including its fields and access permissions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact CSR name (e.g., 'mstatus', 'mcause', 'mepc')"
                }
            },
            "required": ["name"]
        }
    },
    {
        "name": "get_stats",
        "description": "Get statistics about the RISC-V ISA database (number of instructions, CSRs, extensions).",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]

# Tool dispatcher
TOOL_FUNCTIONS = {
    "search_instructions": search_instructions,
    "search_csrs": search_csrs,
    "list_extensions": list_extensions,
    "get_extension_details": get_extension_details,
    "get_instruction_details": get_instruction_details,
    "get_csr_details": get_csr_details,
    "get_stats": get_stats,
}


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result as JSON string."""
    if tool_name not in TOOL_FUNCTIONS:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    try:
        result = TOOL_FUNCTIONS[tool_name](**tool_input)
        return json.dumps(result, indent=2, default=str)
    except Exception as e:
        return json.dumps({"error": str(e)})


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the RISC-V Instruction Set Architecture (ISA).

You have access to a database containing:
- 1150+ RISC-V instructions with encodings, assembly syntax, and descriptions
- 380+ Control and Status Registers (CSRs) with fields and access modes
- 145 RISC-V extensions (base ISA, standard extensions, and ratified extensions)

When users ask about RISC-V:
1. Use the search and lookup tools to find accurate information
2. Provide clear, technical explanations
3. Include relevant details like instruction encodings, CSR addresses, or extension dependencies
4. If you're not sure about something, say so rather than guessing

Common RISC-V extensions include:
- I: Base integer instructions
- M: Integer multiply/divide
- A: Atomic instructions
- F/D/Q: Single/double/quad precision floating-point
- C: Compressed (16-bit) instructions
- V: Vector extension
- Zba/Zbb/Zbs: Bit manipulation
- Zk*: Cryptography extensions

Always use tools to look up specific instruction details rather than relying on memory."""


def chat(message: str, history: list, api_key: str) -> tuple[str, list]:
    """Process a chat message using Claude Haiku with tools."""
    if not api_key:
        return "Please enter your Anthropic API key in the settings.", history

    if not _inst_index:
        return "Data not loaded. Please ensure the RISC-V data files are present.", history

    try:
        client = anthropic.Anthropic(api_key=api_key)

        # Convert history to Claude message format
        messages = []
        for h in history:
            messages.append({"role": "user", "content": h[0]})
            if h[1]:
                messages.append({"role": "assistant", "content": h[1]})

        messages.append({"role": "user", "content": message})

        # Initial API call
        response = client.messages.create(
            model="claude-haiku-4-20250514",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Handle tool use loop
        while response.stop_reason == "tool_use":
            # Find tool use blocks
            tool_uses = [block for block in response.content if block.type == "tool_use"]

            # Execute tools and build results
            tool_results = []
            for tool_use in tool_uses:
                result = process_tool_call(tool_use.name, tool_use.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result,
                })

            # Continue conversation with tool results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            response = client.messages.create(
                model="claude-haiku-4-20250514",
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )

        # Extract final text response
        final_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                final_text += block.text

        history.append((message, final_text))
        return "", history

    except anthropic.AuthenticationError:
        return "Invalid API key. Please check your Anthropic API key.", history
    except Exception as e:
        return f"Error: {str(e)}", history


def create_demo():
    """Create the Gradio demo interface."""

    with gr.Blocks(
        title="RISC-V ISA Chatbot",
        theme=gr.themes.Soft(),
        css="""
        .chatbot {min-height: 400px;}
        .stats-box {padding: 10px; background: #f0f0f0; border-radius: 8px; margin: 10px 0;}
        """
    ) as demo:
        gr.Markdown("""
        # RISC-V ISA Chatbot

        Ask questions about RISC-V instructions, CSRs, and extensions. Powered by Claude Haiku.

        **Examples:**
        - "What instructions are in the M extension?"
        - "Explain the mstatus CSR and its fields"
        - "How does the ADD instruction work?"
        - "List all vector extensions"
        """)

        # Stats display
        stats = get_stats()
        gr.Markdown(f"""
        <div class="stats-box">
        <b>Database:</b> {stats['instructions']} instructions | {stats['csrs']} CSRs | {stats['extensions']} extensions
        </div>
        """)

        chatbot = gr.Chatbot(
            label="Chat",
            elem_classes=["chatbot"],
            height=450,
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your question",
                placeholder="Ask about RISC-V instructions, CSRs, or extensions...",
                scale=4,
            )
            submit = gr.Button("Send", variant="primary", scale=1)

        with gr.Accordion("Settings", open=False):
            api_key = gr.Textbox(
                label="Anthropic API Key",
                placeholder="sk-ant-...",
                type="password",
                value=os.environ.get("ANTHROPIC_API_KEY", ""),
            )
            gr.Markdown("""
            Get your API key from [console.anthropic.com](https://console.anthropic.com/).
            Your key is only used for API calls and is not stored.
            """)

        # Event handlers
        submit.click(
            chat,
            inputs=[msg, chatbot, api_key],
            outputs=[msg, chatbot],
        )
        msg.submit(
            chat,
            inputs=[msg, chatbot, api_key],
            outputs=[msg, chatbot],
        )

        gr.Markdown("""
        ---
        *Data source: [RISC-V Unified Database](https://github.com/riscv-software-src/riscv-unified-db)*
        """)

    return demo


# Initialize on startup
print("Loading RISC-V data...")
build_indices()
print(f"Loaded: {len(_inst_index)} instructions, {len(_csr_index)} CSRs, {len(_ext_index)} extensions")

# Create and launch demo
demo = create_demo()

if __name__ == "__main__":
    demo.launch()
