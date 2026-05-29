export interface Qwen3Config {
  model: string
  source: string
  hidden_size: number
  num_hidden_layers: number
  num_attention_heads: number
  num_key_value_heads: number
  head_dim: number
  intermediate_size: number
  vocab_size: number
  max_position_embeddings: number
  rms_norm_eps: number
  tie_word_embeddings: boolean
  rope_theta?: number
  torch_dtype?: string
}

export interface BlockMetrics {
  shape: string
  flops: number // for a single invocation (one layer)
  params: number
  kvBytes?: number // per layer
}

export function dtypeBytes(cfg: Qwen3Config): number {
  const d = (cfg.torch_dtype ?? 'bfloat16').toLowerCase()
  if (d.includes('fp8') || d.includes('float8')) return 1
  if (d.includes('32')) return 4
  return 2 // bf16 / fp16
}

/** Per-block metrics for a forward pass over T tokens. */
export function computeMetrics(cfg: Qwen3Config, T: number): Record<string, BlockMetrics> {
  const d = cfg.hidden_size
  const h = cfg.num_attention_heads
  const kv = cfg.num_key_value_heads
  const hd = cfg.head_dim
  const qd = h * hd
  const kvd = kv * hd
  const inter = cfg.intermediate_size
  const V = cfg.vocab_size
  const bytes = dtypeBytes(cfg)

  const lin = (inDim: number, outDim: number) => 2 * inDim * outDim * T
  const norm = (dim: number) => 4 * dim * T

  return {
    embed: { shape: `[${T}] → [${T}, ${d}]`, flops: 0, params: V * d },
    input_norm: { shape: `[${T}, ${d}]`, flops: norm(d), params: d },
    qkv_proj: {
      shape: `[${T}, ${d}] → [${T}, ${qd + 2 * kvd}]`,
      flops: lin(d, qd + 2 * kvd),
      params: d * (qd + 2 * kvd),
    },
    q_norm: { shape: `[${T}, ${h}, ${hd}]`, flops: norm(qd), params: hd },
    k_norm: { shape: `[${T}, ${kv}, ${hd}]`, flops: norm(kvd), params: hd },
    rope: { shape: `Q[${T}, ${qd}] · K[${T}, ${kvd}]`, flops: 6 * (qd + kvd) * T, params: 0 },
    attn: {
      shape: `Q[${T}, ${h}, ${hd}] → [${T}, ${qd}]`,
      flops: 4 * h * hd * T * T,
      params: 0,
      kvBytes: 2 * kvd * T * bytes,
    },
    o_proj: { shape: `[${T}, ${qd}] → [${T}, ${d}]`, flops: lin(qd, d), params: qd * d },
    post_norm: { shape: `[${T}, ${d}]`, flops: norm(d), params: d },
    gate_up: {
      shape: `[${T}, ${d}] → [${T}, ${2 * inter}]`,
      flops: lin(d, 2 * inter),
      params: d * 2 * inter,
    },
    silu: { shape: `[${T}, ${2 * inter}] → [${T}, ${inter}]`, flops: 2 * inter * T, params: 0 },
    down: { shape: `[${T}, ${inter}] → [${T}, ${d}]`, flops: lin(inter, d), params: inter * d },
    final_norm: { shape: `[${T}, ${d}]`, flops: norm(d), params: d },
    lm_head: { shape: `[${T}, ${d}] → [${T}, ${V}]`, flops: lin(d, V), params: V * d },
    logits: { shape: `[${T}, ${V}]`, flops: 0, params: 0 },
  }
}

const LAYER_BLOCK_IDS = new Set([
  'input_norm', 'qkv_proj', 'q_norm', 'k_norm', 'rope', 'attn',
  'o_proj', 'post_norm', 'gate_up', 'silu', 'down',
])

export interface ModelSummary {
  totalParams: number
  layerParams: number
  totalFlops: number
  kvBytesTotal: number
  dtype: number
}

export function summarize(cfg: Qwen3Config, T: number): ModelSummary {
  const m = computeMetrics(cfg, T)
  const L = cfg.num_hidden_layers
  let layerParams = 0
  let layerFlops = 0
  let onceParams = 0
  let onceFlops = 0
  let kvPerLayer = 0
  for (const [id, met] of Object.entries(m)) {
    if (LAYER_BLOCK_IDS.has(id)) {
      layerParams += met.params
      layerFlops += met.flops
      kvPerLayer += met.kvBytes ?? 0
    } else {
      onceParams += met.params
      onceFlops += met.flops
    }
  }
  // Tied embeddings: embed and lm_head share weights — count once.
  if (cfg.tie_word_embeddings) onceParams -= cfg.vocab_size * cfg.hidden_size

  return {
    totalParams: onceParams + L * layerParams,
    layerParams,
    totalFlops: onceFlops + L * layerFlops,
    kvBytesTotal: L * kvPerLayer,
    dtype: dtypeBytes(cfg),
  }
}

// Formatters -----------------------------------------------------------------
export function fmtCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return String(n)
}

export function fmtFlops(n: number): string {
  if (n === 0) return '—'
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TFLOP`
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} GFLOP`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} MFLOP`
  return `${n.toFixed(0)} FLOP`
}

export function fmtBytes(n: number): string {
  if (!n) return '—'
  if (n >= 1 << 30) return `${(n / (1 << 30)).toFixed(2)} GiB`
  if (n >= 1 << 20) return `${(n / (1 << 20)).toFixed(2)} MiB`
  if (n >= 1 << 10) return `${(n / (1 << 10)).toFixed(1)} KiB`
  return `${n} B`
}
