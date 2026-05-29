import { useEffect, useMemo, useState } from 'react'
import { ExternalLink } from 'lucide-react'
import { fetchExplorerJson, errorMessage } from '@/lib/fetch'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { sourceUrl } from '@/lib/assets'
import { ALL_BLOCKS, DECODER_LAYER, HEAD, PRELUDE, type Qwen3Block } from './qwen3Blocks'
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

const KIND_COLOR: Record<string, string> = {
  embed: '#06b6d4',
  norm: '#8b5cf6',
  proj: '#6366f1',
  rope: '#38bdf8',
  attn: '#10b981',
  act: '#f59e0b',
  mlp: '#6366f1',
  head: '#ef4444',
}

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
    <div className="flex flex-col gap-5">
      {/* Controls + summary */}
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
              <div className="w-40">
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
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
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

      <div className="dashboard-grid">
        <Card>
          <CardHeader>
            <CardTitle>Qwen3 decoder — inference forward pass</CardTitle>
            <p className="text-sm text-ink-soft">
              One decoder layer (repeated ×{cfg.num_hidden_layers}). Click a block for details.
            </p>
          </CardHeader>
          <CardContent>
            <BlockDiagram
              lens={lens}
              selectedId={selectedId}
              onSelect={setSelectedId}
              lensValue={lensValue}
              layers={cfg.num_hidden_layers}
            />
          </CardContent>
        </Card>

        <BlockDrawer block={block} metricLine={selectedId ? lensValue(selectedId) : ''} lens={lens} />
      </div>
    </div>
  )
}

// --- SVG vertical diagram ---------------------------------------------------
const BW = 250
const BH = 40
const GAP = 14
const X = 70

function BlockDiagram({
  lens,
  selectedId,
  onSelect,
  lensValue,
  layers,
}: {
  lens: Lens
  selectedId: string
  onSelect: (id: string) => void
  lensValue: (id: string) => string
  layers: number
}) {
  const ordered: { block: Qwen3Block; group: 'prelude' | 'layer' | 'head' }[] = [
    ...PRELUDE.map((block) => ({ block, group: 'prelude' as const })),
    ...DECODER_LAYER.map((block) => ({ block, group: 'layer' as const })),
    ...HEAD.map((block) => ({ block, group: 'head' as const })),
  ]
  const yOf = (i: number) => 16 + i * (BH + GAP)
  const layerIdx = ordered.map((o, i) => ({ ...o, i })).filter((o) => o.group === 'layer')
  const layerTop = yOf(layerIdx[0].i) - 8
  const layerBot = yOf(layerIdx[layerIdx.length - 1].i) + BH + 8
  const height = yOf(ordered.length - 1) + BH + 16
  const railX = X - 18

  return (
    <svg viewBox={`0 0 ${BW + 160} ${height}`} className="w-full" role="img" aria-label="Qwen3 decoder layer">
      {/* decoder-layer bracket */}
      <rect
        x={X - 28}
        y={layerTop}
        width={BW + 56}
        height={layerBot - layerTop}
        rx={12}
        fill="rgba(99,102,241,0.04)"
        stroke="rgba(99,102,241,0.25)"
        strokeDasharray="4 4"
      />
      <text x={X - 22} y={layerTop - 4} className="diagram-label" fill="#9ca3af">
        decoder layer × {layers}
      </text>

      {/* residual rail */}
      <line x1={railX} y1={layerTop + 6} x2={railX} y2={layerBot - 6} className="diagram-edge" />

      {ordered.map(({ block }, i) => {
        const y = yOf(i)
        const active = block.id === selectedId
        const color = KIND_COLOR[block.kind] ?? '#6366f1'
        const metric = lens === 'flow' ? '' : lensValue(block.id)
        return (
          <g
            key={block.id}
            className={cn('diagram-node', active && 'active')}
            transform={`translate(${X},${y})`}
            role="button"
            tabIndex={0}
            aria-label={block.label}
            onClick={() => onSelect(block.id)}
            onKeyDown={(ev) => {
              if (ev.key === 'Enter' || ev.key === ' ') onSelect(block.id)
            }}
          >
            {i > 0 && <line x1={BW / 2} y1={-GAP} x2={BW / 2} y2={0} className="diagram-edge" />}
            <rect
              width={BW}
              height={BH}
              rx={9}
              fill="rgba(17,24,39,0.92)"
              stroke={active ? '#38bdf8' : color}
              strokeWidth={active ? 2.5 : 1.4}
            />
            <rect width={4} height={BH} rx={2} fill={color} />
            <text className="diagram-label" x={14} y={BH / 2 - 1} fontSize={12} fill="#f3f4f6">
              {block.label}
            </text>
            {metric && (
              <text x={14} y={BH / 2 + 12} fontSize={9} fill="#9ca3af" className="diagram-label">
                {metric.length > 34 ? metric.slice(0, 33) + '…' : metric}
              </text>
            )}
          </g>
        )
      })}
    </svg>
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
