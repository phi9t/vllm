// The Qwen3 dense decoder-layer inference forward pass, as implemented in vLLM.
// file:line references grounded against the current source on phi9t-mainline
// (resolve by symbol if they drift). Inference only — no training/backward.

export type BlockKind = 'embed' | 'norm' | 'proj' | 'rope' | 'attn' | 'act' | 'mlp' | 'head'

export interface Qwen3Block {
  id: string
  label: string
  symbol: string
  ref: string // file:line
  kind: BlockKind
  desc: string
  qwen3Note?: string // what's distinctive about Qwen3 here
}

/** Pre-layer blocks (run once before the stack). */
export const PRELUDE: Qwen3Block[] = [
  {
    id: 'embed',
    label: 'Token embeddings',
    symbol: 'VocabParallelEmbedding (embed_tokens)',
    ref: 'vllm/model_executor/models/qwen2.py:358',
    kind: 'embed',
    desc: 'input_ids → hidden_states. Looks up a row of the embedding matrix per token.',
    qwen3Note: 'tie_word_embeddings reuses this matrix as the lm_head.',
  },
]

/** The repeating decoder layer (×num_hidden_layers). */
export const DECODER_LAYER: Qwen3Block[] = [
  {
    id: 'input_norm',
    label: 'Input RMSNorm',
    symbol: 'RMSNorm (input_layernorm)',
    ref: 'vllm/model_executor/models/qwen3.py:211',
    kind: 'norm',
    desc: 'Pre-attention normalization, fused with the residual add on all but the first layer.',
  },
  {
    id: 'qkv_proj',
    label: 'QKV projection',
    symbol: 'QKVParallelLinear (qkv_proj)',
    ref: 'vllm/model_executor/models/qwen3.py:98',
    kind: 'proj',
    desc: 'One fused linear producing Q, K, V. Split into q_dim + 2·kv_dim (GQA).',
  },
  {
    id: 'q_norm',
    label: 'Q head-norm',
    symbol: 'RMSNorm (q_norm, head_dim)',
    ref: 'vllm/model_executor/models/qwen3.py:142',
    kind: 'norm',
    desc: 'Per-head RMSNorm over head_dim applied to queries before RoPE.',
    qwen3Note: 'Qwen3-specific: QK-Norm stabilizes attention logits.',
  },
  {
    id: 'k_norm',
    label: 'K head-norm',
    symbol: 'RMSNorm (k_norm, head_dim)',
    ref: 'vllm/model_executor/models/qwen3.py:143',
    kind: 'norm',
    desc: 'Per-head RMSNorm over head_dim applied to keys before RoPE.',
    qwen3Note: 'Qwen3-specific: paired with q_norm.',
  },
  {
    id: 'rope',
    label: 'RoPE (Q, K)',
    symbol: 'RotaryEmbedding (get_rope)',
    ref: 'vllm/model_executor/layers/rotary_embedding/__init__.py:33',
    kind: 'rope',
    desc: 'Rotary position embedding rotates Q and K by position. Uses a cached cos/sin table.',
    qwen3Note: 'rope_theta = 1e6 for long context.',
  },
  {
    id: 'attn',
    label: 'Attention (paged KV)',
    symbol: 'Attention.forward',
    ref: 'vllm/model_executor/layers/attention/attention.py:409',
    kind: 'attn',
    desc: 'Scaled dot-product attention over the paged KV cache; backend chosen at init. GQA: kv_heads < heads.',
    qwen3Note: 'KV cache lives inside this layer (block tables, slot mapping).',
  },
  {
    id: 'o_proj',
    label: 'Output projection',
    symbol: 'RowParallelLinear (o_proj)',
    ref: 'vllm/model_executor/models/qwen3.py:107',
    kind: 'proj',
    desc: 'Projects the attention output (q_dim) back to hidden_size.',
  },
  {
    id: 'post_norm',
    label: 'Post-attn RMSNorm',
    symbol: 'RMSNorm (post_attention_layernorm)',
    ref: 'vllm/model_executor/models/qwen3.py:212',
    kind: 'norm',
    desc: 'Pre-MLP normalization, fused with the residual add.',
  },
  {
    id: 'gate_up',
    label: 'Gate+Up projection',
    symbol: 'MergedColumnParallelLinear (gate_up_proj)',
    ref: 'vllm/model_executor/models/qwen2.py:93',
    kind: 'mlp',
    desc: 'Fused gate and up projection: hidden_size → 2·intermediate_size.',
  },
  {
    id: 'silu',
    label: 'SiLU + gating',
    symbol: 'SiluAndMul',
    ref: 'vllm/model_executor/layers/activation.py:118',
    kind: 'act',
    desc: 'SwiGLU activation: silu(gate) · up → intermediate_size.',
  },
  {
    id: 'down',
    label: 'Down projection',
    symbol: 'RowParallelLinear (down_proj)',
    ref: 'vllm/model_executor/models/qwen2.py:100',
    kind: 'mlp',
    desc: 'Projects intermediate_size back to hidden_size.',
  },
]

/** Post-stack blocks (run once after the layers). */
export const HEAD: Qwen3Block[] = [
  {
    id: 'final_norm',
    label: 'Final RMSNorm',
    symbol: 'RMSNorm (norm)',
    ref: 'vllm/model_executor/models/qwen2.py:382',
    kind: 'norm',
    desc: 'Normalizes the final hidden state before the LM head.',
  },
  {
    id: 'lm_head',
    label: 'LM head',
    symbol: 'ParallelLMHead',
    ref: 'vllm/model_executor/models/qwen3.py:298',
    kind: 'head',
    desc: 'Projects hidden_size → vocab_size to produce logits.',
    qwen3Note: 'Weight tied to embed_tokens when tie_word_embeddings is set.',
  },
  {
    id: 'logits',
    label: 'Logits processor',
    symbol: 'LogitsProcessor',
    ref: 'vllm/model_executor/models/qwen3.py:307',
    kind: 'head',
    desc: 'Gathers logits for the sampled positions; hands off to the Sampler.',
  },
]

export const ALL_BLOCKS: Qwen3Block[] = [...PRELUDE, ...DECODER_LAYER, ...HEAD]
