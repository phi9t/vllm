import { useState } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { TokenPiece, Tokenization } from './types'

function pieceClass(p: TokenPiece): string {
  if (p.special) return 'token-piece token-piece--special'
  if (/\n/.test(p.text)) return 'token-piece token-piece--newline'
  if (/^\s+$/.test(p.text)) return 'token-piece token-piece--space'
  return 'token-piece'
}

// Visualize the byte-level BPE convention: leading spaces become "·", newlines "⏎".
function display(text: string): string {
  return text.replace(/ /g, '·').replace(/\n/g, '⏎')
}

export default function TokenizationView({ tok }: { tok: Tokenization }) {
  const [idx, setIdx] = useState(0)
  const sample = tok.samples[idx]
  if (!sample) return null

  const qwen = sample.qwen3_token_count
  const gpt2 = sample.dataset_token_count ?? null
  const delta = gpt2 != null ? qwen - gpt2 : null

  return (
    <div className="flex flex-col gap-5">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <CardTitle>Qwen3 tokenization · {tok.tokenizer}</CardTitle>
            <label className="flex items-center gap-2 text-xs text-ink-soft">
              sample
              <select
                className="rounded-md border border-panelborder bg-panel px-2 py-1 font-mono text-xs text-ink"
                value={idx}
                onChange={(e) => setIdx(Number(e.target.value))}
              >
                {tok.samples.map((s, i) => (
                  <option key={s.id ?? i} value={i}>
                    {s.id ?? `sample ${i + 1}`}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-wrap gap-3">
            <Stat label="Qwen3 tokens" value={qwen} accent="cyan" />
            <Stat label="GPT-2 tokens (dataset)" value={gpt2 ?? '—'} />
            {delta != null && (
              <Stat
                label="difference"
                value={`${delta > 0 ? '+' : ''}${delta}`}
                accent={delta === 0 ? undefined : 'primary'}
              />
            )}
          </div>
          <div className="flex flex-wrap gap-1 rounded-lg border border-panelborder bg-black/20 p-3">
            {sample.tokens.map((p, i) => (
              <span key={i} className={pieceClass(p)} title={`id ${p.id}`}>
                {display(p.text) || '∅'}
              </span>
            ))}
          </div>
          <p className="text-xs leading-relaxed text-ink-muted">{tok.note}</p>
        </CardContent>
      </Card>
    </div>
  )
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string
  value: string | number
  accent?: 'cyan' | 'primary'
}) {
  const color =
    accent === 'cyan' ? 'text-cyan' : accent === 'primary' ? 'text-primary' : 'text-ink'
  return (
    <div className="rounded-lg border border-panelborder bg-panel px-4 py-2">
      <div className="text-[11px] uppercase tracking-wide text-ink-muted">{label}</div>
      <div className={`font-mono text-xl font-bold ${color}`}>{value}</div>
    </div>
  )
}
