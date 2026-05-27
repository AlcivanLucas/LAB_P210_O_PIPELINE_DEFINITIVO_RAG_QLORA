"""
Laboratório 10 — Pipeline Definitivo: RAG + QLoRA + Otimização de Inferência na GPU
Versão: script Python para execução local (equivalente ao lab10_pipeline.ipynb).

Requisitos:
    pip install -U "transformers>=4.44" "accelerate>=0.33" "bitsandbytes>=0.43" \
                   sentencepiece protobuf matplotlib

GPU recomendada: ≥ 8 GB VRAM.
  - Ampere+ (A100, RTX 30/40, L4): usa FlashAttention-2 nativo.
  - Turing / outros (T4, RTX 20): fallback automático para SDPA (memory-efficient).
  - Sem GPU: roda em CPU apenas para demonstração (muito lento, sem métricas VRAM).
"""

import gc
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---------------------------------------------------------------------------
# Configuração global
# ---------------------------------------------------------------------------
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
TARGET_TOKENS = 4000   # tokens de contexto RAG simulado
NEW_TOKENS = 100       # tokens a gerar em cada benchmark

HAS_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if HAS_CUDA else "cpu")


def free_vram():
    gc.collect()
    if HAS_CUDA:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def mb(x: int) -> float:
    return x / (1024 ** 2)


# ---------------------------------------------------------------------------
# Passo 0 — Informações da GPU
# ---------------------------------------------------------------------------
print("=" * 60)
print("LABORATÓRIO 10 — Pipeline Definitivo")
print("=" * 60)

if HAS_CUDA:
    gpu_name = torch.cuda.get_device_name(0)
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    total_vram = mb(torch.cuda.get_device_properties(0).total_memory)
    print(f"GPU : {gpu_name}")
    print(f"Compute Capability : {cc_major}.{cc_minor}")
    print(f"VRAM total : {total_vram:,.1f} MB")
    SUPPORTS_FA2 = cc_major >= 8
    print(f"Suporta FlashAttention-2 : {SUPPORTS_FA2}")
else:
    print("AVISO: CUDA não disponível. Rodando em CPU (sem métricas de VRAM).")
    SUPPORTS_FA2 = False

print()

# ---------------------------------------------------------------------------
# Passo 1 — Ingestão Eficiente (QLoRA 4-bit)
# ---------------------------------------------------------------------------
print("[Passo 1] Carregando modelo quantizado em 4-bit (QLoRA)...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

free_vram()

model_baseline = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config if HAS_CUDA else None,
    device_map={"": 0} if HAS_CUDA else "cpu",
    attn_implementation="eager",
)
model_baseline.eval()

vram_load_mb = mb(torch.cuda.memory_allocated()) if HAS_CUDA else 0.0
print(f"[Passo 1] VRAM ocupada pelo modelo 4-bit : {vram_load_mb:,.1f} MB\n")

# ---------------------------------------------------------------------------
# Passo 2 — Simulando o RAG Massivo
# ---------------------------------------------------------------------------
print("[Passo 2] Gerando contexto RAG simulado...")

fake_chapter = (
    "Capítulo de Manual Médico. Paciente do sexo masculino, 58 anos, hipertenso, "
    "apresenta dispneia aos médios esforços há três semanas, com episódios de dor "
    "precordial em aperto, irradiando para o membro superior esquerdo. Ausculta "
    "cardíaca revela sopro sistólico em foco aórtico. Eletrocardiograma evidencia "
    "alterações de repolarização ventricular em parede inferior. Ecocardiograma "
    "transtorácico demonstra hipertrofia concêntrica do ventrículo esquerdo, com "
    "fração de ejeção preservada em 58%. Cateterismo coronariano mostra lesão "
    "obstrutiva de 80% em artéria descendente anterior proximal. Conduta: "
    "angioplastia transluminal coronariana com implante de stent farmacológico, "
    "associada a dupla antiagregação plaquetária por 12 meses, estatina de alta "
    "intensidade, betabloqueador e inibidor da ECA. Acompanhamento ambulatorial "
    "trimestral com avaliação de função renal, perfil lipídico e ecocardiograma anual. "
)

rag_text = fake_chapter
while len(tokenizer(rag_text, return_tensors="pt").input_ids[0]) < TARGET_TOKENS:
    rag_text += fake_chapter

prompt = (
    "Você é um assistente clínico. Com base nos capítulos abaixo, gere um resumo "
    "clínico estruturado.\n\n--- CONTEXTO RAG ---\n"
    + rag_text
    + "\n--- FIM ---\n\nResumo clínico:\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)
n_ctx = inputs.input_ids.shape[1]
print(f"[Passo 2] Tokens no contexto (prompt + RAG) : {n_ctx:,}\n")

# ---------------------------------------------------------------------------
# Função de benchmark
# ---------------------------------------------------------------------------
def benchmark_generate(model, inputs, use_cache: bool, label: str) -> dict:
    model.config.use_cache = use_cache
    free_vram()
    if HAS_CUDA:
        torch.cuda.synchronize()
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=NEW_TOKENS,
                do_sample=False,
                use_cache=use_cache,
                pad_token_id=tokenizer.eos_token_id,
            )
        if HAS_CUDA:
            torch.cuda.synchronize()
        dt = time.time() - t0
        peak_mb = mb(torch.cuda.max_memory_allocated()) if HAS_CUDA else 0.0
        n_gen = out.shape[1] - inputs.input_ids.shape[1]
        print(f"[{label}]")
        print(f"  tempo={dt:.2f}s | tokens/s={n_gen/dt:.2f} | pico VRAM={peak_mb:,.1f} MB")
        return {"label": label, "time_s": dt, "tok_per_s": n_gen / dt,
                "peak_vram_mb": peak_mb, "oom": False}
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower():
            raise
        peak_mb = mb(torch.cuda.max_memory_allocated()) if HAS_CUDA else 0.0
        free_vram()
        print(f"[{label}] *** OOM *** pico VRAM antes do crash={peak_mb:,.1f} MB")
        print(f"  -> {str(e).splitlines()[0]}")
        return {"label": label, "time_s": float("nan"), "tok_per_s": 0.0,
                "peak_vram_mb": peak_mb, "oom": True}


# ---------------------------------------------------------------------------
# Passo 3 — Gargalo de Geração (SEM KV Cache)
# ---------------------------------------------------------------------------
print("[Passo 3] Benchmark baseline — eager SEM KV cache (espere OOM ou lentidão)...")
results = []
results.append(
    benchmark_generate(model_baseline, inputs, use_cache=False,
                       label="Baseline (eager, NO KV cache)")
)
print()

# ---------------------------------------------------------------------------
# Passo 4 — Engenharia de Otimização
# ---------------------------------------------------------------------------
print("[Passo 4a] Benchmark eager COM KV cache...")
results.append(
    benchmark_generate(model_baseline, inputs, use_cache=True,
                       label="Eager + KV Cache")
)
print()

# Recarrega modelo com attention hardware-aware
del model_baseline
free_vram()

chosen_attn = "flash_attention_2" if SUPPORTS_FA2 else "sdpa"
print(f"[Passo 4b] Recarregando modelo com attn_implementation='{chosen_attn}'...")
try:
    model_opt = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config if HAS_CUDA else None,
        device_map={"": 0} if HAS_CUDA else "cpu",
        attn_implementation=chosen_attn,
    )
except Exception as e:
    print(f"  Falha com {chosen_attn}: {e}\n  Fazendo fallback para 'sdpa'.")
    chosen_attn = "sdpa"
    model_opt = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config if HAS_CUDA else None,
        device_map={"": 0} if HAS_CUDA else "cpu",
        attn_implementation="sdpa",
    )
model_opt.eval()
print(f"  Modelo recarregado com '{chosen_attn}'.")

results.append(
    benchmark_generate(model_opt, inputs, use_cache=True,
                       label=f"{chosen_attn.upper()} + KV Cache (OTIMIZADO)")
)
print()

# ---------------------------------------------------------------------------
# Resumo final
# ---------------------------------------------------------------------------
print("=" * 82)
print(f"VRAM do modelo 4-bit carregado : {vram_load_mb:,.1f} MB")
print(f"Tokens de contexto (RAG simulado) : {n_ctx:,}")
print(f"Attention otimizada usada : {chosen_attn}")
print()
print(f"{'Configuração':<45} {'Tempo(s)':>10} {'tok/s':>8} {'Pico VRAM(MB)':>16}")
print("-" * 82)
for r in results:
    t = "OOM" if r["oom"] else f"{r['time_s']:.2f}"
    ts = "—" if r["oom"] else f"{r['tok_per_s']:.2f}"
    print(f"{r['label']:<45} {t:>10} {ts:>8} {r['peak_vram_mb']:>16,.1f}")

base = results[0]
opt = results[-1]
if not base["oom"] and not opt["oom"] and opt["time_s"] > 0:
    print(f"\nSpeedup tempo (baseline → otimizado) : {base['time_s']/opt['time_s']:.2f}x")
if base["peak_vram_mb"] > 0 and opt["peak_vram_mb"] > 0:
    reducao = (1 - opt["peak_vram_mb"] / base["peak_vram_mb"]) * 100
    print(f"Redução pico VRAM                    : {reducao:.1f}%")

# ---------------------------------------------------------------------------
# Gráfico (opcional — requer matplotlib)
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    times = [r["time_s"] for r in results]
    vrams = [r["peak_vram_mb"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(range(len(labels)), times)
    axes[0].set_title("Tempo de geração (s)")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, rotation=20, ha="right")

    axes[1].bar(range(len(labels)), vrams)
    axes[1].set_title("Pico VRAM (MB)")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig("benchmark.png", dpi=120)
    print("\nGráfico salvo em benchmark.png")
    plt.show()
except ImportError:
    print("\n(matplotlib não instalado — gráfico não gerado)")
