

LABORATÓRIO  10:  O  Pipeline  Definitivo  (RAG,  QLoRA  e  Otimização  de
Inferência na GPU)
- Objetivo do Laboratório
Este  é  o  laboratório  integrador  da  disciplina.  Vocês  deverão  orquestrar  um
pipeline  de  IA  ponta  a  ponta.  O  objetivo  é  simular  um  ambiente  de  produção
onde  um  modelo  ajustado  por  QLoRA  (Unidade  II)  precisa  ler  um  contexto
gigantesco  recuperado  por  um  RAG  (Unidade  III),  exigindo  que  vocês
manipulem  a  arquitetura  do  Self-Attention  (Unidade  I)  utilizando  FlashAttention
e KV Cache  para evitar o erro de  Out-Of-Memory (OOM)  na GPU.
- Contexto do Problema Corporativo
A  HealthTech  adorou  o  sistema  RAG  de  buscas  médicas  que  vocês  fiz eram  no
Lab  09.  Agora  eles  querem  colocar  o  sistema  em  produção  para  gerar  relatórios
automatizados. O fluxo definido pela arquitetura é:
## 1.
O  RAG  recupera  5  capítulos  inteiros  de  manuais  médicos  (aprox.  30.000
tokens).
## 2.
Esse  contexto  massivo  é  injetado  em  um  modelo  Llama-3  que  foi
fine-tunado  para o jargão médico.
-  O modelo precisa gerar um resumo clínico de 500 palavras.
O  Desastre:  Quando  o  time  de  testes  tentou  rodar  isso  na  nuvem,  a
complexidade  O(n²)  do  Self-Attention  estourou  a  memória  VRAM  da  GPU  e  o
servidor  travou.  Sua  missão  como  Arquiteto  de  IA  é  consertar  o  código  de
inferência combinando quantização, cache e algoritmos  Hardware-Aware  .
- Roteiro de Implementação (Passo a Passo)
Passo 1: Ingestão Eficiente (Revisando a Unidade II)
## ●
Não  podemos  carregar  o  LLM  base  em  16-bits  (Float16)  pois  ele
ocuparia muita memória logo de cara.
## ●
Utilize  a  biblioteca  bitsandbytes  para  carregar  um  modelo  gerador
auto-regressivo  (ex:  TinyLlama/TinyLlama-1.1B-Chat-v1.0  ou  similar)
utilizando  a  configuração  QLoRA  em  4-bits  (load_in_4bit=True  e
bnb_4bit_compute_dtype=torch.float16).

## ●
Métrica:  Registre  no  seu  relatório  quantos  Megabytes  de  VRAM  o  modelo
ocupou ao ser carregado quantizado.
Passo 2: Simulando o RAG Massivo (Revisando o Lab 09)
## ●
Gere  uma  string  de  texto  fictícia  contendo  cerca  de  10.000  a  15.000
tokens  simulando  os  PDFs  médicos  recuperados  pelo  seu  banco
vetorial.
●  Passe esse texto pelo  Tokenizador  (AutoTokenizer) do modelo.
Passo 3: O Gargalo de Geração (O Problema do Decoder)
## ●
Escreva  um  loop  simples  solicitando  ao  modelo  a  geração  de  100  novos
tokens com base no contexto massivo.
## ●
A  Pegadinha:  Force  o  modelo  a  NÃO  usar  cache  de  memória
## (model.config.use_cache = False).
## ●
Métrica:  Monitore  e  anote  o  tempo  total  de  geração  e  o  pico  de  memória
VRAM  utilizado  (torch.cuda.max_memory_allocated()).  Observe  a
lentidão  causada  pelo  recálculo  redundante  de  Q,  K,  V  a  cada  nova
palavra gerada.
Passo 4: A Engenharia de Otimização
## ●
Refatore  sua  função  de  geração  para  ativar  a  otimização  de  software:  KV
## Cache  (use_cache = True).
## ●
Refatore  o  carregamento  do  modelo  inicial  para  usar  a  otimização  de
hardware  baseada  na  memória  SRAM  da  GPU:  FlashAttention-2
(adicione attn_implementation="flash_attention_2").
●  Execute a mesma geração de 100 tokens.
## ●
Métrica:  Registre  o  novo  tempo  de  geração  e  a  drástica  redução  no  pico
de memória VRAM durante a fase de  prompting  .
Passo 5: Análise Arquitetural no README
●  No arquivo README.md, redija um parecer técnico de 2 parágrafos:
## ○
Parte  A:  Explique  como  a  combinação  de  QLoRA,  KV  Cache  e
FlashAttention  "salvou"  o  Transformer  tradicional  do  colapso  da
VRAM neste laboratório.
## ○
Parte  B:  Argumente  por  que,  se  o  cliente  exigisse  o
processamento  de  2  milhões  de  tokens  (em  vez  de  15.000),  até
mesmo  o  FlashAttention  falharia,  e  por  que  a  indústria  precisaria

migrar  para  State  Space  Models  (como  a  arquitetura  Mamba)
com sua complexidade de memória O(1).
- Critérios de Avaliação e Contrato Pedagógico
Este é o último laboratório antes da Avaliação P3. O rigor será máximo.
4.1. Formato de Entrega e Versionamento:
## ●
Plataforma:  O  código  fonte  (arquivo  .ipynb  ou  scripts  Python)  e  o
relatório  (README.md  com  as  métricas  de  benchmark  )  devem  ser
submetidos  via  repositório  no  GitHub  .  O  link  deve  ser  enviado  na
plataforma iCEV Digital.
## ●
Versionamento:  A  versão  final  a  ser  corrigida  deve  conter
obrigatoriamente a  tag  ou release  "v1.0"  .
4.2. Política de Integridade Acadêmica e Uso de IA:
## ●
Uso  Permitido:  É  liberado  o  uso  de  IAs  generativas  para  geração  do  texto
base fictício e estrutura de plotagem dos gráficos de memória.
## ●
Declaração  Obrigatória:  É  OBRIGATÓRIO  constar  no  topo  do
README.md:  "Partes  deste  laboratório  foram  geradas/complementadas
com IA, revisadas e validadas por [Seu Nome]"  .
## ●
Punição:  Submeter  código  não  compreendido  ou  copiado  de  colegas
sem  o  devido  crédito  resultará  em  Nota  0  (zero)  imediata  e  registro  da
ocorrência.
4.3. Política de Prazos e Atrasos:
●  1 dia de atraso:  -20% da nota contratual.
●  2 a 3 dias de atraso:  -50% da nota contratual.
## ●
Acima  de  3  dias:  Nota  0  (zero),  exceto  com  justificativa  oficial  de
saúde/trabalho.