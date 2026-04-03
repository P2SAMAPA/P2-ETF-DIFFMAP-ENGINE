# DIFFMAP-ETF Engine

## Overview

DIFFMAP-ETF is a **diffusion-inspired generative model** for ETF selection.

Unlike traditional models that predict a single return, DIFFMAP learns a **distribution of possible next-day returns** and selects the ETF with the highest expected outcome.

---

## Core Idea

We model:

x₀ = next-day return

Using noise:

xσ = x₀ + σϵ

The model learns to predict ϵ (noise), enabling reconstruction of return distributions.

---

## Key Features

- Multi-window training:
  - 2008 → present
  - 2012 → present
  - 2015 → present
  - ...
- Distribution-based prediction (not point forecast)
- CPU-friendly architecture
- Works with public ETF + macro data

---

## Pipeline

1. Load dataset from Hugging Face
2. Train diffusion model per window
3. Sample future returns
4. Aggregate across windows
5. Select ETF with highest expected return

---

## Output

```json
{
  "pick": "XLK",
  "expected_return": 0.0042,
  "confidence": 0.66
}
