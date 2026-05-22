> **Partes deste laboratório foram geradas/complementadas com IA, revisadas e validadas por alcivan**

# Laboratório 10 — Pipeline Definitivo: RAG + QLoRA + Otimização de Inferência na GPU

Disciplina: Arquitetura de IA / Transformers em produção
Autor: **alcivan**

## Objetivo

Simular o ambiente de produção da HealthTech, onde um LLM precisa **ler um contexto gigante recuperado por RAG (~12k tokens)** e **gerar um resumo clínico**, sem estourar a VRAM. A solução combina três frentes:

| Frente | Técnica | O que ataca |
|---|---|---|
| Pesos do modelo | **QLoRA / 4-bit (NF4 + bitsandbytes)** | Reduz o footprint estático do modelo |
| Inferência auto-regressiva | **KV Cache** | Elimina o recálculo redundante de Q/K/V dos tokens passados |
| Kernel da Self-Attention | **FlashAttention-2** (ou SDPA como fallback) | Reduz I/O HBM↔SRAM e o pico de VRAM durante o *prefill* do contexto longo |

## Como executar

### Google Colab (T4)
1. Upload do `lab10_pipeline.ipynb`.
2. `Runtime → Change runtime type → T4 GPU`.
3. `Runtime → Run all`.

> ⚠️ **A T4 do Colab grátis é Turing (SM 7.5) e NÃO suporta `flash_attention_2`** — esse kernel exige Ampere+ (A100, L4, RTX 30/40). O notebook detecta isso automaticamente e faz **fallback para `attn_implementation="sdpa"`** (Scaled-Dot-Product-Attention do PyTorch com kernel memory-efficient). Conceitualmente equivalente para fins do lab: ambos evitam materializar a matriz `n×n` completa na HBM.

## Arquivos
- `lab10_pipeline.ipynb` — notebook Colab com os 4 passos e benchmark
- `README.md` — este parecer + métricas
- `benchmark.png` — gráfico de tempo e VRAM (gerado pela última célula)
