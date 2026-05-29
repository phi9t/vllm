import { useEffect, useState } from 'react'
import { fetchExplorerJson, errorMessage } from '@/lib/fetch'
import { cn } from '@/lib/utils'
import type { FinewebSample, FinewebSchema, Tokenization } from './types'
import SchemaView from './SchemaView'
import SampleBrowser from './SampleBrowser'
import StatsView from './StatsView'
import TokenizationView from './TokenizationView'

type Tab = 'format' | 'samples' | 'stats' | 'tokenization'

const TABS: { id: Tab; label: string }[] = [
  { id: 'format', label: 'Format / schema' },
  { id: 'samples', label: 'Samples' },
  { id: 'stats', label: 'Distributions' },
  { id: 'tokenization', label: 'Tokenization' },
]

export default function DataExplorer() {
  const [tab, setTab] = useState<Tab>('format')
  const [schema, setSchema] = useState<FinewebSchema | null>(null)
  const [sample, setSample] = useState<FinewebSample | null>(null)
  const [tok, setTok] = useState<Tokenization | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      fetchExplorerJson<FinewebSchema>('fineweb_schema.json'),
      fetchExplorerJson<FinewebSample>('fineweb_sample.json'),
      fetchExplorerJson<Tokenization>('tokenization.json'),
    ])
      .then(([s, sm, t]) => {
        setSchema(s)
        setSample(sm)
        setTok(t)
      })
      .catch((e) => setError(errorMessage(e)))
  }, [])

  if (error) {
    return (
      <div className="panel p-6 text-danger">
        Failed to load data: {error}
        <p className="mt-2 text-sm text-ink-soft">
          Generate it first: <code className="code-ref">./scripts/workflow.sh gen-data</code>
        </p>
      </div>
    )
  }

  if (!schema || !sample || !tok) {
    return <div className="panel p-6 text-ink-soft">Loading fineweb-edu sample…</div>
  }

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex flex-wrap gap-2" role="tablist" aria-label="Data views">
          {TABS.map(({ id, label }) => (
            <button
              key={id}
              role="tab"
              aria-selected={tab === id}
              className={cn(
                'rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors',
                tab === id
                  ? 'border-panelborder-active bg-panel-hover text-ink'
                  : 'border-panelborder bg-panel text-ink-soft hover:text-ink',
              )}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </div>
        <span className="text-xs text-ink-muted">
          {sample.count} rows · source <code className="code-ref">{sample.source}</code>
        </span>
      </div>

      {tab === 'format' && <SchemaView schema={schema} />}
      {tab === 'samples' && <SampleBrowser rows={sample.rows} columns={schema.columns} />}
      {tab === 'stats' && <StatsView rows={sample.rows} />}
      {tab === 'tokenization' && <TokenizationView tok={tok} />}
    </div>
  )
}
