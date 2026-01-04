---
title: RISC-V ISA Chatbot
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.31.0
app_file: app.py
pinned: false
license: bsd-3-clause
short_description: Chat about RISC-V instructions and extensions
---

# RISC-V ISA Chatbot

An AI-powered chatbot that answers questions about the RISC-V Instruction Set Architecture using Claude Haiku.

## Features

- **Search Instructions**: Find RISC-V instructions by name or extension
- **Search CSRs**: Look up Control and Status Registers
- **Browse Extensions**: Explore all 145 RISC-V extensions
- **Detailed Information**: Get encoding, assembly format, and descriptions

## Database

This chatbot has access to the RISC-V Unified Database containing:
- 1,150+ instructions
- 380+ CSRs
- 145 extensions

## Usage

1. Enter your Anthropic API key in Settings
2. Ask questions about RISC-V in natural language

### Example Questions

- "What instructions are in the M extension?"
- "Explain the mstatus CSR"
- "How does the vector ADD instruction work?"
- "What's the difference between Zba and Zbb extensions?"

## API Key

You need an Anthropic API key to use this chatbot. Get one at [console.anthropic.com](https://console.anthropic.com/).

## Data Source

Data from the [RISC-V Unified Database](https://github.com/riscv-software-src/riscv-unified-db), maintained by RISC-V International.

## License

BSD-3-Clause
