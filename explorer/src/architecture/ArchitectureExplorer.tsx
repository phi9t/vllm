import { useEffect, useMemo, useState } from 'react'
import { ExternalLink } from 'lucide-react'
import { fetchExplorerJson, errorMessage } from '@/lib/fetch'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { sourceUrl } from '@/lib/assets'
import { ALL_BLOCKS, type Qwen3Block } from './qwen3Blocks'
import Qwen3Circuit from './Qwen3Circuit'
import {
  computeMetrics,
  fmtBytes,
  fmtCount,
  fmtFlops,
  summarize,
  type Qwen3Config,
} from './blockMath'

type Lens = 'flow' | 'shapes' | 'compute' | 'memory'
const LENSES: { id: Lens; label: string }[] = [
  { id: 'flow', label: 'Flow' },
  { id: 'shapes', label: 'Shapes' },
  { id: 'compute', label: 'Compute' },
  { id: 'memory', label: 'Memory' },
]

export default function ArchitectureExplorer() {
  const [cfg, setCfg] = useState<Qwen3Config | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [lens, setLens] = useState<Lens>('shapes')
  const [tokens, setTokens] = useState(512)
  const [selectedId, setSelectedId] = useState<string>('attn')

  useEffect(() => {
    fetchExplorerJson<Qwen3Config>('qwen3_config.json')
      .then(setCfg)
      .catch((e) => setError(errorMessage(e)))
  }, [])

  const metrics = useMemo(() => (cfg ? computeMetrics(cfg, tokens) : null), [cfg, tokens])
  const summary = useMemo(() => (cfg ? summarize(cfg, tokens) : null), [cfg, tokens])

  if (error) {
    return (
      <div className="panel p-6 text-danger">
        Failed to load qwen3_config.json: {error}
        <p className="mt-2 text-sm text-ink-soft">
          Generate it first: <code className="code-ref">./scripts/workflow.sh gen-data</code>
        </p>
      </div>
    )
  }
  if (!cfg || !metrics || !summary) {
    return <div className="panel p-6 text-ink-soft">Loading Qwen3 config…</div>
  }

  const lensValue = (id: string): string => {
    const m = metrics[id]
    if (!m) return ''
    switch (lens) {
      case 'shapes':
        return m.shape
      case 'compute':
        return fmtFlops(m.flops)
      case 'memory':
        return m.kvBytes ? `KV ${fmtBytes(m.kvBytes)}` : m.params ? `${fmtCount(m.params)} p` : '—'
      default:
        return ''
    }
  }

  const block = ALL_BLOCKS.find((b) => b.id === selectedId) ?? null

  return (
    // Diagram fills the entire left panel; config selectors + details on the right.
    <div className="grid items-start gap-5 lg:grid-cols-2">
      <Card className="flex flex-col">
        <CardHeader>
          <CardTitle>Qwen3 decoder — inference forward pass</CardTitle>
          <p className="text-sm text-ink-soft">
            Residual mainline, input ↑ output. One decoder layer (×{cfg.num_hidden_layers}); click
            any block.
          </p>
        </CardHeader>
        <CardContent className="flex flex-1 justify-center">
          <Qwen3Circuit
            lens={lens}
            selectedId={selectedId}
            onSelect={setSelectedId}
            lensValue={lensValue}
            layers={cfg.num_hidden_layers}
          />
        </CardContent>
      </Card>

      <div className="flex flex-col gap-5 lg:sticky lg:top-6">
        {/* Model config selectors + summary */}
        <Card>
          <CardContent className="flex flex-col gap-4 p-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex flex-wrap gap-2" role="tablist" aria-label="Lens">
                {LENSES.map(({ id, label }) => (
                  <button
                    key={id}
                    role="tab"
                    aria-selected={lens === id}
                    className={cn(
                      'rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors',
                      lens === id
                        ? 'border-panelborder-active bg-panel-hover text-ink'
                        : 'border-panelborder bg-panel text-ink-soft hover:text-ink',
                    )}
                    onClick={() => setLens(id)}
                  >
                    {label}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-3 text-xs text-ink-soft">
                <span className="whitespace-nowrap">
                  tokens <span className="font-mono text-cyan">{tokens}</span>
                </span>
                <div className="w-32">
                  <Slider
                    min={8}
                    max={4096}
                    step={8}
                    value={[tokens]}
                    onValueChange={([v]) => setTokens(v)}
                  />
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <Stat label="model" value={cfg.model.split('/').pop() ?? cfg.model} mono />
              <Stat label="params" value={fmtCount(summary.totalParams)} />
              <Stat label={`KV cache @ ${tokens}`} value={fmtBytes(summary.kvBytesTotal)} />
              <Stat label={`forward FLOPs @ ${tokens}`} value={fmtFlops(summary.totalFlops)} />
            </div>
            <p className="text-xs text-ink-muted">
              {cfg.num_hidden_layers} layers · d={cfg.hidden_size} · heads={cfg.num_attention_heads}/
              {cfg.num_key_value_heads} (GQA {cfg.num_attention_heads / cfg.num_key_value_heads}×) ·
              head_dim={cfg.head_dim} · config source <code className="code-ref">{cfg.source}</code>
            </p>
          </CardContent>
        </Card>

        <BlockDrawer
          block={block}
          metricLine={selectedId ? lensValue(selectedId) : ''}
          lens={lens}
        />
      </div>
    </div>
  )
}

// --- Drawer -----------------------------------------------------------------
function BlockDrawer({
  block,
  metricLine,
  lens,
}: {
  block: Qwen3Block | null
  metricLine: string
  lens: Lens
}) {
  if (!block) {
    return <Card className="drawer text-sm text-ink-muted">Select a block to inspect it.</Card>
  }
  return (
    <Card className="drawer">
      <div>
        <div className="text-[11px] uppercase tracking-wide text-ink-muted">{block.kind}</div>
        <h3 className="mt-1 text-lg font-bold text-ink">{block.label}</h3>
      </div>
      <a
        href={sourceUrl(block.ref)}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1.5 text-sm text-cyan hover:underline"
      >
        <code className="code-ref">{block.ref}</code>
        <ExternalLink size={13} aria-hidden="true" />
      </a>
      <div className="text-sm">
        <span className="text-ink-muted">symbol </span>
        <code className="code-ref">{block.symbol}</code>
      </div>
      <p className="text-sm leading-relaxed text-ink-soft">{block.desc}</p>
      {block.qwen3Note && (
        <p className="rounded-lg border border-cyan/30 bg-cyan/5 px-3 py-2 text-xs text-cyan">
          {block.qwen3Note}
        </p>
      )}
      {lens !== 'flow' && metricLine && (
        <div className="rounded-lg border border-panelborder bg-panel px-3 py-2">
          <div className="text-[11px] uppercase tracking-wide text-ink-muted">{lens}</div>
          <div className="font-mono text-sm text-ink">{metricLine}</div>
        </div>
      )}
    </Card>
  )
}

function Stat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="rounded-lg border border-panelborder bg-panel px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-ink-muted">{label}</div>
      <div className={cn('text-lg font-bold text-ink', mono && 'font-mono text-base')}>{value}</div>
    </div>
  )
}
