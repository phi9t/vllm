import { useEffect, useState } from 'react'
import { fetchExplorerJson, errorMessage } from '@/lib/fetch'
import { AsyncBoundary } from '@/explorer-kit/AsyncBoundary'
import { ViewTabs } from '@/explorer-kit/ViewTabs'
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

  const ready = schema !== null && sample !== null && tok !== null

  return (
    <AsyncBoundary
      loading={!ready}
      error={error}
      loadingLabel="Loading fineweb-edu sample…"
      errorPrefix="Failed to load data"
    >
      {ready && (
        <div className="flex flex-col gap-5">
          <div className="flex flex-wrap items-center gap-3">
            <ViewTabs
              ariaLabel="Data views"
              value={tab}
              onChange={setTab}
              options={TABS.map((t) => ({ value: t.id, label: t.label }))}
            />
            <span className="text-xs text-ink-muted">
              {sample.count} rows · source <code className="code-ref">{sample.source}</code>
            </span>
          </div>

          {tab === 'format' && <SchemaView schema={schema} />}
          {tab === 'samples' && <SampleBrowser rows={sample.rows} columns={schema.columns} />}
          {tab === 'stats' && <StatsView rows={sample.rows} />}
          {tab === 'tokenization' && <TokenizationView tok={tok} />}
        </div>
      )}
    </AsyncBoundary>
  )
}
