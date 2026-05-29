import type { BlockKind, BlockType, ModelArch, ModelConfig } from './modelArch'

// --- Formatters (ported from blockMath.ts) -----------------------------------

export function dtypeBytes(cfg: ModelConfig): number {
  const d = (cfg.torch_dtype ?? 'bfloat16').toLowerCase()
  if (d.includes('fp8') || d.includes('float8')) return 1
  if (d.includes('32')) return 4
  return 2 // bf16 / fp16
}

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

// --- Kind colors (port from Qwen3Circuit + Phase 2/3 additions) --------------

export const KIND_COLOR: Record<BlockKind, string> = {
  embed:  '#06b6d4',
  norm:   '#8b5cf6',
  proj:   '#6366f1',
  rope:   '#38bdf8',
  attn:   '#10b981',
  act:    '#f59e0b',
  mlp:    '#6366f1',
  head:   '#ef4444',
  moe:    '#f472b6',
  router: '#f59e0b',
  latent: '#22d3ee',
}

// --- Block metrics types -----------------------------------------------------

export interface BlockMetrics {
  shape: string
  flops: number
  params: number
  activeParams?: number
  kvBytes?: number
}

// --- Per-block metric functions ----------------------------------------------

type MetricFn = (cfg: ModelConfig, T: number) => BlockMetrics

const lin = (inDim: number, outDim: number, T: number) => 2 * inDim * outDim * T
const normFlops = (dim: number, T: number) => 4 * dim * T

export const BLOCK_METRICS: Record<BlockType, MetricFn> = {
  embed: (cfg, T) => ({
    shape: `[${T}] → [${T}, ${cfg.hidden_size}]`,
    flops: 0,
    params: cfg.vocab_size * cfg.hidden_size,
  }),

  rmsnorm: (cfg, T) => ({
    shape: `[${T}, ${cfg.hidden_size}]`,
    flops: normFlops(cfg.hidden_size, T),
    params: cfg.hidden_size,
  }),

  qk_norm: (cfg, T) => {
    const hd = cfg.head_dim
    return {
      shape: `[${T}, *, ${hd}]`,
      flops: normFlops(hd, T),
      params: hd,
    }
  },

  qkv_linear: (cfg, T) => {
    const d = cfg.hidden_size
    const h = cfg.num_attention_heads
    const kv = cfg.num_key_value_heads
    const hd = cfg.head_dim
    const qd = h * hd
    const kvd = kv * hd
    return {
      shape: `[${T}, ${d}] → [${T}, ${qd + 2 * kvd}]`,
      flops: lin(d, qd + 2 * kvd, T),
      params: d * (qd + 2 * kvd),
    }
  },

  rope: (cfg, T) => {
    const h = cfg.num_attention_heads
    const kv = cfg.num_key_value_heads
    const hd = cfg.head_dim
    const qd = h * hd
    const kvd = kv * hd
    return {
      shape: `Q[${T}, ${qd}] · K[${T}, ${kvd}]`,
      flops: 6 * (qd + kvd) * T,
      params: 0,
    }
  },

  attention_gqa: (cfg, T) => {
    const h = cfg.num_attention_heads
    const kv = cfg.num_key_value_heads
    const hd = cfg.head_dim
    const qd = h * hd
    const kvd = kv * hd
    const bytes = dtypeBytes(cfg)
    return {
      shape: `Q[${T}, ${h}, ${hd}] → [${T}, ${qd}]`,
      flops: 4 * h * hd * T * T,
      params: 0,
      kvBytes: 2 * kvd * T * bytes,
    }
  },

  o_proj: (cfg, T) => {
    const d = cfg.hidden_size
    const qd = cfg.num_attention_heads * cfg.head_dim
    return {
      shape: `[${T}, ${qd}] → [${T}, ${d}]`,
      flops: lin(qd, d, T),
      params: qd * d,
    }
  },

  gate_up: (cfg, T) => {
    const d = cfg.hidden_size
    const inter = cfg.intermediate_size
    return {
      shape: `[${T}, ${d}] → [${T}, ${2 * inter}]`,
      flops: lin(d, 2 * inter, T),
      params: d * 2 * inter,
    }
  },

  activation: (cfg, T) => {
    const inter = cfg.intermediate_size
    return {
      shape: `[${T}, ${2 * inter}] → [${T}, ${inter}]`,
      flops: 2 * inter * T,
      params: 0,
    }
  },

  down: (cfg, T) => {
    const d = cfg.hidden_size
    const inter = cfg.intermediate_size
    return {
      shape: `[${T}, ${inter}] → [${T}, ${d}]`,
      flops: lin(inter, d, T),
      params: inter * d,
    }
  },

  lm_head: (cfg, T) => {
    const d = cfg.hidden_size
    const V = cfg.vocab_size
    return {
      shape: `[${T}, ${d}] → [${T}, ${V}]`,
      flops: lin(d, V, T),
      params: V * d,
    }
  },

  logits: (cfg, T) => ({
    shape: `[${T}, ${cfg.vocab_size}]`,
    flops: 0,
    params: 0,
  }),

  // Phase 2 MoE blocks — derived from ModelConfig MoE fields
  moe_router: (cfg, T) => {
    const d = cfg.hidden_size
    const E = cfg.num_experts ?? 1
    return {
      shape: `[${T}, ${d}] → [${T}, ${E}]`,
      flops: 2 * d * E * T,
      params: d * E,
    }
  },

  moe_experts: (cfg, T) => {
    const d = cfg.hidden_size
    const E = cfg.num_experts ?? 1
    const k = cfg.num_experts_per_tok ?? 1
    const I = cfg.moe_intermediate_size ?? cfg.intermediate_size
    // Each expert is a SwiGLU FFN: gate_up (d→2I) + down (I→d) = 3·d·I params
    const paramsPerExpert = 3 * d * I
    return {
      shape: `[${T}, ${d}] → [${T}, ${d}]  (top-${k}/${E})`,
      flops: k * 6 * d * I * T,        // only top-k experts run per token
      params: E * paramsPerExpert,      // total across all experts
      activeParams: k * paramsPerExpert, // only top-k run per token
    }
  },

  shared_expert: (cfg, T) => {
    // Used by some MoE models (e.g. DeepSeek); Qwen3-MoE has no shared expert.
    // Width = num_shared_experts * moe_intermediate_size
    const d = cfg.hidden_size
    const ns = cfg.num_shared_experts ?? 1
    const I = cfg.moe_intermediate_size ?? cfg.intermediate_size
    const width = ns * I
    const p = 3 * d * width
    return {
      shape: `[${T}, ${d}] → [${T}, ${d}]  (${ns} shared)`,
      flops: 6 * d * width * T,
      params: p,
      activeParams: p,                  // shared expert always runs
    }
  },
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  q_a_linear: (_cfg, _T) => ({ shape: '—', flops: 0, params: 0 }), // Phase 2/3
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  q_b_linear: (_cfg, _T) => ({ shape: '—', flops: 0, params: 0 }), // Phase 2/3
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  kv_a_linear: (_cfg, _T) => ({ shape: '—', flops: 0, params: 0 }), // Phase 2/3
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  kv_b_linear: (_cfg, _T) => ({ shape: '—', flops: 0, params: 0 }), // Phase 2/3
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  attention_mla: (_cfg, _T) => ({ shape: '—', flops: 0, params: 0 }), // Phase 2/3
}

// --- computeMetrics ----------------------------------------------------------

/** Compute per-block metrics for a forward pass over T tokens. */
export function computeMetrics(m: ModelArch, T: number): Record<string, BlockMetrics> {
  const cfg = m.config
  const result: Record<string, BlockMetrics> = {}

  const processBlock = (block: { id: string; type: BlockType }) => {
    const fn = BLOCK_METRICS[block.type]
    result[block.id] = fn ? fn(cfg, T) : { shape: '—', flops: 0, params: 0 }
  }

  for (const block of m.prelude) processBlock(block)

  for (const group of m.layers) {
    for (const branch of group.branches) {
      processBlock(branch.preNorm)
      for (const step of branch.steps) processBlock(step)
    }
  }

  for (const block of m.head) processBlock(block)

  return result
}

// --- summarize ---------------------------------------------------------------

export interface ModelSummary {
  totalParams: number
  activeParams: number
  kvBytesTotal: number
  totalFlops: number
  dtype: number
}

/** Aggregate model-level summary for T tokens. */
export function summarize(m: ModelArch, T: number): ModelSummary {
  const cfg = m.config
  const metrics = computeMetrics(m, T)

  let totalParams = 0
  let activeParams = 0
  let totalFlops = 0
  let kvBytesTotal = 0

  const addBlock = (id: string, times: number) => {
    const met = metrics[id]
    if (!met) return
    totalParams += met.params * times
    activeParams += (met.activeParams ?? met.params) * times
    totalFlops += met.flops * times
    kvBytesTotal += (met.kvBytes ?? 0) * times
  }

  for (const block of m.prelude) addBlock(block.id, 1)

  for (const group of m.layers) {
    const r = group.repeat
    for (const branch of group.branches) {
      addBlock(branch.preNorm.id, r)
      for (const step of branch.steps) addBlock(step.id, r)
    }
  }

  for (const block of m.head) addBlock(block.id, 1)

  // Tied embeddings: subtract once if lm_head shares weights with embed
  if (cfg.tie_word_embeddings) {
    const tiedParams = cfg.vocab_size * cfg.hidden_size
    totalParams -= tiedParams
    activeParams -= tiedParams
  }

  return {
    totalParams,
    activeParams,
    kvBytesTotal,
    totalFlops,
    dtype: dtypeBytes(cfg),
  }
}
