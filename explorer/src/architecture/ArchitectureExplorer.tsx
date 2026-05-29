import { useEffect, useMemo, useState } from 'react'
import { ExternalLink } from 'lucide-react'
import { fetchExplorerJson, errorMessage } from '@/lib/fetch'
import { cn } from '@/lib/utils'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Slider } from '@/components/ui/slider'
import { sourceUrl } from '@/lib/assets'
import { computeMetrics, fmtBytes, fmtCount, fmtFlops, summarize } from './blockTypes'
import ModelCircuit, { type Lens } from './ModelCircuit'
import type { Block, ModelArch, ModelIndexEntry } from './modelArch'

const LENSES: { id: Lens; label: string }[] = [
  { id: 'flow', label: 'Flow' },
  { id: 'shapes', label: 'Shapes' },
  { id: 'compute', label: 'Compute' },
  { id: 'memory', label: 'Memory' },
]

export default function ArchitectureExplorer() {
  const [index, setIndex] = useState<ModelIndexEntry[] | null>(null)
  const [indexError, setIndexError] = useState<string | null>(null)

  const [slug, setSlug] = useState<string | null>(null)
  const [manifest, setManifest] = useState<ModelArch | null>(null)
  const [manifestError, setManifestError] = useState<string | null>(null)
  const [manifestLoading, setManifestLoading] = useState(false)

  const [lens, setLens] = useState<Lens>('shapes')
  const [tokens, setTokens] = useState(512)
  const [selectedId, setSelectedId] = useState<string>('')

  // Load the model index on mount
  useEffect(() => {
    fetchExplorerJson<ModelIndexEntry[]>('models/index.json')
      .then((entries) => {
        setIndex(entries)
        if (entries.length > 0) setSlug(entries[0].slug)
      })
      .catch((e) => setIndexError(errorMessage(e)))
  }, [])

  // Load the selected model manifest when slug changes
  useEffect(() => {
    if (!slug) return
    setManifest(null)
    setManifestError(null)
    setManifestLoading(true)
    fetchExplorerJson<ModelArch>(`models/${slug}.json`)
      .then((m) => {
        setManifest(m)
        setManifestLoading(false)
        // Default selection: first block in first branch's first step (or first prelude)
        const firstId =
          m.layers[0]?.branches[0]?.steps[0]?.id ??
          m.layers[0]?.branches[0]?.preNorm?.id ??
          m.prelude[0]?.id ??
          ''
        setSelectedId(firstId)
      })
      .catch((e) => {
        setManifestError(errorMessage(e))
        setManifestLoading(false)
      })
  }, [slug])

  const metrics = useMemo(
    () => (manifest ? computeMetrics(manifest, tokens) : null),
    [manifest, tokens],
  )
  const summary = useMemo(
    () => (manifest ? summarize(manifest, tokens) : null),
    [manifest, tokens],
  )

  if (indexError) {
    return (
      <div className="panel p-6 text-danger">
        Failed to load models/index.json: {indexError}
        <p className="mt-2 text-sm text-ink-soft">
          Generate it first: <code className="code-ref">./scripts/workflow.sh gen-data</code>
        </p>
      </div>
    )
  }

  if (!index) {
    return <div className="panel p-6 text-ink-soft">Loading model index…</div>
  }

  const lensValue = (id: string): string => {
    if (!metrics) return ''
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

  // Find the selected block across all block collections in the manifest
  const findBlock = (m: ModelArch, id: string): Block | null => {
    for (const b of m.prelude) if (b.id === id) return b
    for (const group of m.layers) {
      for (const branch of group.branches) {
        if (branch.preNorm.id === id) return branch.preNorm
        for (const step of branch.steps) if (step.id === id) return step
      }
    }
    for (const b of m.head) if (b.id === id) return b
    return null
  }

  const selectedBlock = manifest && selectedId ? findBlock(manifest, selectedId) : null

  const modelLabel =
    index.find((e) => e.slug === slug)?.label ?? slug ?? ''

  return (
    <div className="grid items-start gap-5 lg:grid-cols-2">
      <Card className="flex flex-col">
        <CardHeader>
          <CardTitle>
            {manifest ? manifest.model.split('/').pop() ?? manifest.model : modelLabel} — inference forward pass
          </CardTitle>
          <p className="text-sm text-ink-soft">
            Residual mainline, input ↑ output.{manifest ? ` One decoder layer (×${manifest.config.num_hidden_layers});` : ''} click any block.
          </p>
        </CardHeader>
        <CardContent className="flex flex-1 justify-center">
          {manifestLoading && (
            <div className="py-12 text-sm text-ink-soft">Loading {modelLabel}…</div>
          )}
          {manifestError && (
            <div className="py-6 text-sm text-danger">
              Failed to load {slug}.json: {manifestError}
            </div>
          )}
          {manifest && metrics && (
            <ModelCircuit
              manifest={manifest}
              lens={lens}
              selectedId={selectedId}
              onSelect={setSelectedId}
              lensValue={lensValue}
            />
          )}
        </CardContent>
      </Card>

      <div className="flex flex-col gap-5 lg:sticky lg:top-6">
        {/* Model config selectors + summary */}
        <Card>
          <CardContent className="flex flex-col gap-4 p-5">
            {/* Model switcher */}
            {index.length > 1 && (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-xs text-ink-muted">Model</span>
                <select
                  value={slug ?? ''}
                  onChange={(e) => setSlug(e.target.value)}
                  className="rounded-lg border border-panelborder bg-panel px-2 py-1.5 text-xs text-ink focus:outline-none focus:border-panelborder-active"
                >
                  {index.map((entry) => (
                    <option key={entry.slug} value={entry.slug}>
                      {entry.label} ({fmtCount(entry.totalParams)})
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="flex flex-wrap items-center justify-between gap-4">
              {/* Lens tabs */}
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
              {/* Token slider */}
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

            {manifest && summary ? (
              <>
                <div className="grid grid-cols-2 gap-3">
                  <Stat label="params" value={fmtCount(summary.totalParams)} />
                  <Stat label="active params/token" value={fmtCount(summary.activeParams)} />
                  <Stat label={`KV cache @ ${tokens}`} value={fmtBytes(summary.kvBytesTotal)} />
                  <Stat label={`forward FLOPs @ ${tokens}`} value={fmtFlops(summary.totalFlops)} />
                </div>
                <p className="text-xs text-ink-muted">
                  {manifest.config.num_hidden_layers} layers · d={manifest.config.hidden_size} ·
                  heads={manifest.config.num_attention_heads}/{manifest.config.num_key_value_heads}
                  {' '}(GQA {manifest.config.num_attention_heads / manifest.config.num_key_value_heads}×) ·
                  head_dim={manifest.config.head_dim} · source{' '}
                  <code className="code-ref">{manifest.source}</code>
                </p>
              </>
            ) : (
              <p className="text-xs text-ink-muted">—</p>
            )}
          </CardContent>
        </Card>

        <BlockDrawer
          block={selectedBlock}
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
  block: Block | null
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
      {block.note && (
        <p className="rounded-lg border border-cyan/30 bg-cyan/5 px-3 py-2 text-xs text-cyan">
          {block.note}
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
