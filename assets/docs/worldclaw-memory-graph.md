# WorldClaw Research Memory Graph

## Purpose

This document describes the page-level memory graph used to explain how WorldClaw stores and reuses research evidence.

WorldClaw is represented as an evidence-licensed memory system rather than a generic code-editing agent. Each node in the graph records a bounded research episode with:

- Time and experiment context
- Target model architecture
- Dataset or robot embodiment
- Proposed intervention
- Verification artifact
- Follow-up skill or trace document

## Memory Contract

Every memory node should answer five questions:

1. What model or repository was changed?
2. What dataset, environment, or robot embodiment defined the task?
3. What failure mechanism was diagnosed?
4. What intervention was tried?
5. What evidence gates support the next claim?

## Design Boundary

The graph is a project-page explanation layer. It does not claim that every future WorldClaw run is automatically solved. It shows how prior evidence can be indexed, inspected, and reused when selecting the next world-model intervention.
