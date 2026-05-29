export interface FinewebColumn {
  name: string
  type: string
  desc: string
}

export interface FinewebSchema {
  dataset: string
  columns: FinewebColumn[]
}

export interface FinewebRow {
  text: string
  id?: string
  dump?: string
  url?: string
  date?: string
  file_path?: string
  language?: string
  language_score?: number
  token_count?: number
  score?: number
  int_score?: number
}

export interface FinewebSample {
  source: string
  count: number
  rows: FinewebRow[]
}

export interface TokenPiece {
  id: number
  text: string
  special: boolean
}

export interface TokenSample {
  id?: string
  text: string
  tokens: TokenPiece[]
  qwen3_token_count: number
  dataset_token_count?: number | null
}

export interface Tokenization {
  tokenizer: string
  tokenizer_source: string
  note: string
  samples: TokenSample[]
}
