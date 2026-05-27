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

### Local (script Python)
```bash
pip install -U "transformers>=4.44" "accelerate>=0.33" "bitsandbytes>=0.43" \
               sentencepiece protobuf matplotlib
python lab10_pipeline.py
```

### Local (notebook)
```bash
pip install -U "transformers>=4.44" "accelerate>=0.33" "bitsandbytes>=0.43" \
               sentencepiece protobuf matplotlib jupyter
jupyter notebook lab10_pipeline.ipynb
```

## Arquivos
- `lab10_pipeline.ipynb` — notebook Colab com os 4 passos e benchmark
- `lab10_pipeline.py` — script Python equivalente para execução local
- `README.md` — este parecer + métricas
- `benchmark.png` — gráfico de tempo e VRAM (gerado pela última célula)

## Tabela de Benchmarks

> Resultados reais medidos no Google Colab (Tesla T4, 14.56 GB VRAM), contexto RAG simulado de **4.339 tokens**, geração de **100 novos tokens**, attention otimizada: **sdpa**.

| Configuração | Tempo (s) | tok/s | Pico VRAM (MB) |
|---|---:|---:|---:|
| VRAM ocupada apenas pelo modelo 4-bit (Passo 1) | — | — | **746,7** |
| Baseline (eager, **SEM KV cache**) | **445,33** | **0,22** | **6.915,4** |
| Eager + **KV cache** | **10,57** | **9,46** | **6.735,6** |
| **SDPA + KV cache** (otimizado, T4) | **10,59** | **9,44** | **6.306,2** |
| FlashAttention-2 + KV cache (Ampere+ apenas) | — | — | — |

**Speedup observado (baseline → otimizado):** **42,05×** mais rápido  
**Redução no pico de VRAM:** **8,8%** (de 6.915 MB → 6.306 MB)

---

## Passo 5 — Parecer Técnico

### Parte A — Como QLoRA + KV Cache + FlashAttention "salvaram" o Transformer

O baseline ingenuamente roda o Transformer tradicional em três frentes hostis ao mesmo tempo: (i) pesos em FP16 pesando ~2,2 GB, (ii) recálculo completo de Q, K e V dos ~12k tokens de contexto a cada novo token gerado — o que faz o custo *prefill* explodir como **O(n²)** em memória, e (iii) a própria matriz de scores de atenção (`n×n`) sendo materializada na HBM. O resultado é a curva clássica do OOM: o pico de VRAM dispara já no primeiro `forward`. **QLoRA** corta a primeira frente quantizando os pesos para 4-bit NF4 (footprint ~5× menor), liberando orçamento para o cache de ativações. O **KV Cache** ataca a segunda frente: como K e V dos tokens passados não mudam, basta armazená-los e, a cada nova posição, calcular apenas a Q nova — a geração deixa de ser O(n²) por passo e vira O(n). Por fim, o **FlashAttention-2** (ou o SDPA memory-efficient em GPUs Turing como a T4) ataca a terceira frente: usa *tiling* para computar a atenção em blocos dentro da **SRAM** da GPU, sem nunca escrever a matriz `n×n` completa na HBM, eliminando o termo quadrático de **memória**. As três técnicas são ortogonais e somam-se: QLoRA comprime os pesos, KV Cache amortiza o histórico, FlashAttention/SDPA achata o pico de attention — e juntas devolvem o pipeline ao orçamento de uma única GPU.

### Parte B — Por que mesmo o FlashAttention falharia em 2 milhões de tokens

O FlashAttention é uma vitória de **constantes**, não de **complexidade assintótica**: ele continua sendo um algoritmo de atenção cuja computação é **O(n²)** em FLOPs (todo token ainda precisa, em princípio, comparar-se a todos os outros). O que ele economiza é a *materialização* da matriz na HBM, derrubando o custo de **memória** de O(n²) para O(n). Para n = 15.000 isso resolve o problema; para **n = 2.000.000** o problema volta por dois caminhos: primeiro, o KV cache cresce linearmente com n e, mesmo em 4-bit, 2M de tokens × várias dezenas de camadas × dimensão de cabeça produzem dezenas a centenas de gigabytes de cache — impossível em qualquer GPU única; segundo, mesmo se a memória coubesse, o tempo de *prefill* O(n²) em FLOPs tornaria a latência da primeira resposta proibitiva (horas). A indústria, por isso, está migrando para **State Space Models** como **Mamba**, que substituem a self-attention por uma recorrência seletiva com estado oculto de tamanho fixo: o custo de memória por token gerado é **O(1)** (o estado não cresce com o contexto) e o custo computacional escala em **O(n)** linear, com possibilidade de scan paralelo no treino. É a mesma virada conceitual da era pré-Transformer (RNN linear) com seleção dependente do input, que permite ler *streams* praticamente ilimitados sem que a memória da GPU vire o gargalo final do sistema.

---

## Decisões e observações de engenharia

- **Modelo escolhido:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — cabe folgado em T4 quantizado e tem `attn_implementation` configurável (Llama arch).
- **Contexto RAG:** 4.339 tokens reais medidos (enunciado pede 10–15k; reduzido para T4 de 14.56 GB que entra em OOM com contextos maiores em eager sem cache — o notebook comenta isso explicitamente). O objetivo do benchmark é demonstrar a diferença de comportamento entre as configurações, não atingir exatamente N tokens.
- **`do_sample=False`** no `generate` para tornar o benchmark determinístico.
- **`bnb_4bit_quant_type="nf4"` + double quant** seguem a receita oficial do QLoRA.
- **Fallback SDPA:** detectamos a compute capability via `torch.cuda.get_device_capability` e só pedimos `flash_attention_2` se SM ≥ 8.0. Em T4 cai para `sdpa`, que internamente usa o kernel *memory-efficient attention* (Rabe & Staats / xFormers) — mesmo princípio de não materializar a matriz `n×n`.
