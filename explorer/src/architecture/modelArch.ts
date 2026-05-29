export type BlockKind =
  | 'embed' | 'norm' | 'proj' | 'rope' | 'attn'
  | 'act' | 'mlp' | 'moe' | 'router' | 'latent' | 'head'

export type BlockType =
  | 'embed' | 'rmsnorm' | 'qk_norm' | 'qkv_linear' | 'rope'
  | 'attention_gqa' | 'o_proj' | 'gate_up' | 'activation' | 'down'
  | 'lm_head' | 'logits'
  | 'moe_router' | 'moe_experts' | 'shared_expert'
  | 'q_a_linear' | 'q_b_linear' | 'kv_a_linear' | 'kv_b_linear' | 'attention_mla'

export interface Block {
  id: string; type: BlockType; label: string; symbol: string
  ref: string  // "path/to/file.py:123"
  kind: BlockKind; desc: string; note?: string
}

export interface Branch { name: string; accent: string; preNorm: Block; steps: Block[] }

export interface LayerGroup { repeat: number; label?: string; branches: Branch[] }

export interface ModelConfig {
  hidden_size: number; num_hidden_layers: number; num_attention_heads: number
  num_key_value_heads: number; head_dim: number; intermediate_size: number
  vocab_size: number; tie_word_embeddings: boolean; torch_dtype?: string; rope_theta?: number
  num_experts?: number; num_experts_per_tok?: number; moe_intermediate_size?: number; num_shared_experts?: number
  q_lora_rank?: number; kv_lora_rank?: number; qk_nope_head_dim?: number
  qk_rope_head_dim?: number; v_head_dim?: number; first_k_dense_replace?: number
}

export interface ModelArch {
  model: string; slug: string; family: string; source: string
  config: ModelConfig; prelude: Block[]; layers: LayerGroup[]; head: Block[]
}

export interface ModelIndexEntry { slug: string; label: string; family: string; totalParams: number }
