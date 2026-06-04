# RLHF Auditor -- Your Reward Model Is Lying to You

A systematic probe suite that exposes hidden biases in RLHF reward models. Because nobody audits the auditor.

## The Hook

"I built a lie detector for RLHF reward models. Every major model failed."

## Concept

RLHF is the industry standard for alignment. But reward models have known pathologies:

1. **Length bias** -- longer responses get higher scores regardless of quality
2. **Sycophancy** -- models rate agreeable responses higher
3. **Position bias** -- responses shown first/last get anchoring effects
4. **Jailbreak susceptibility** -- can be tricked into rating harmful outputs highly
5. **Social desirability** -- rates "polite" over "accurate"
6. **Style bias** -- prefers certain formatting, markdown, bullet points

This tool probes any reward model across all 6 dimensions and produces a "trust score."

## Stack

- HuggingFace Transformers for reward model loading
- OpenAI API for GPT-4 judge baseline
- Streamlit for dashboard
- Datasets library for benchmark construction

## Architecture

```
rlhf-auditor/
├── probes/              # One probe per bias dimension
│   ├── length_bias.py   # Same response at different lengths
│   ├── sycophancy.py    # Contrarian vs agreeable responses
│   ├── position_bias.py # Swap A/B order, check consistency
│   ├── jailbreak.py     # Adversarial prompts that elicit praise
│   ├── desirability.py  # Polite but wrong vs blunt but right
│   └── style_bias.py    # Same content, different formatting
├── models/              # Reward model loaders
├── benchmark/           # Synthetic test cases + human validated set
├── dashboard/             # Streamlit trust report
└── reports/               # Generated audit PDFs
```

## How It Works

1. **Load target reward model** (any HuggingFace model with a score head)
2. **Run probe suite** (~1000 test pairs per probe, ~1 hour)
3. **Compute bias scores** (0 = unbiased, 1 = completely biased)
4. **Generate trust report** with radar chart, per-dimension breakdown, examples
5. **Compare against GPT-4 judge** as reference standard

## Output

```
Reward Model Audit: anthropic/reward-model-deberta-v3-large-run2
Date: 2026-06-04

TRUST SCORE: 34/100

Bias Dimensions:
- Length bias:        0.67  (CRITICAL -- 67% longer responses rated higher)
- Sycophancy:         0.52  (HIGH -- rates agreeable +12% higher)
- Position bias:      0.41  (HIGH -- first response anchored +8%)
- Jailbreak suscept:  0.78  (CRITICAL -- adversarial prompts score 0.89)
- Social desirability:0.33  (MODERATE)
- Style bias:         0.55  (HIGH -- markdown responses +15%)

Recommendation: DO NOT USE for high-stakes decisions.
```

## Viral Mechanics

- Auto-tweet audit results for popular models (with permission)
- "Reward Model Leaderboard" updated weekly
- "Hall of Shame" for most biased models
- Community submissions: audit your own reward model

## License

MIT -- make reward models accountable.
