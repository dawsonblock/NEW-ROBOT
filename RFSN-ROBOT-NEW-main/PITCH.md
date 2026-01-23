# RFSN Control Core

## What it is

A deterministic, safety-bounded robotics control architecture designed
for simulation-to-real transfer.

## What makes it different

- Learning has no authority
- All actions pass a single safety gate
- Failures are explicit and terminal
- Behavior is replayable and auditable

## What it replaces

- Ad-hoc state machines
- Black-box RL controllers
- Unsafe demo-only stacks

## Who it's for

- Game studios shipping physics NPCs
- Robotics labs deploying real hardware
- Companies needing certifiable autonomy

## What it guarantees

- Bounded motion
- Deterministic outcomes
- Transferable behavior

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     ControlPipeline                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌───────────┐  ┌──────────────┐  ┌────────┐ │
│  │ Observer │→│ Executive │→│ SafetyManager │→│Controller│ │
│  └──────────┘  └───────────┘  └──────────────┘  └────────┘ │
│        ↓            ↓               ↓              ↓        │
│    ObsPacket    Decision       decision_safe      tau      │
│                                                     ↓        │
│                                              ┌────────┐     │
│                                              │ Logger │     │
│                                              └────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run demo
uv run python run_demo.py

# Run tests
uv run pytest tests/ -v
```

## System Guarantees

This system guarantees:

- Bounded actions
- Deterministic execution
- Explicit failure states

This system does NOT guarantee:

- Optimal control
- Generalization beyond trained envelopes
- Safety outside declared invariants

**If an invariant breaks, motion halts.**

**If safety checks fail, the system enters FAULT.**

This is intentional.
