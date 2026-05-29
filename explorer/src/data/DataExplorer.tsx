import { useEffect, useState } from 'react'
import { fetchExplorerJson, errorMessage } from '@/lib/fetch'
import { AsyncBoundary } from '@/explorer-kit/AsyncBoundary'
import { ViewTabs } from '@/explorer-kit/ViewTabs'
import { SubjectSwitcher } from '@/explorer-kit/SubjectSwitcher'
import type { FinewebSample, FinewebSchema, Tokenization } from './types'
import SchemaView from './SchemaView'
import SampleBrowser from './SampleBrowser'
import StatsView from './StatsView'
import TokenizationView from './TokenizationView'

/** One switchable dataset; data file names are relative to public/data/. */
interface DatasetEntry {
  slug: string
  label: string
  schema: string
  sample: string
  tokenization: string
}

type Tab = 'format' | 'samples' | 'stats' | 'tokenization'

const TABS: { id: Tab; label: string }[] = [
  { id: 'format', label: 'Format / schema' },
  { id: 'samples', label: 'Samples' },
  { id: 'stats', label: 'Distributions' },
  { id: 'tokenization', label: 'Tokenization' },
]

export default function DataExplorer() {
  const [index, setIndex] = useState<DatasetEntry[] | null>(null)
  const [indexError, setIndexError] = useState<string | null>(null)
  const [slug, setSlug] = useState<string | null>(null)

  const [tab, setTab] = useState<Tab>('format')
  const [schema, setSchema] = useState<FinewebSchema | null>(null)
  const [sample, setSample] = useState<FinewebSample | null>(null)
  const [tok, setTok] = useState<Tokenization | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Load the dataset index on mount.
  useEffect(() => {
    fetchExplorerJson<DatasetEntry[]>('datasets/index.json')
      .then((idx) => {
        setIndex(idx)
        setSlug(idx[0]?.slug ?? null)
      })
      .catch((e) => setIndexError(errorMessage(e)))
  }, [])

  const entry = index && slug ? index.find((d) => d.slug === slug) ?? null : null

  // Load the active dataset's files when the selection changes.
  useEffect(() => {
    if (!entry) return
    setSchema(null)
    setSample(null)
    setTok(null)
    setError(null)
    Promise.all([
      fetchExplorerJson<FinewebSchema>(entry.schema),
      fetchExplorerJson<FinewebSample>(entry.sample),
      fetchExplorerJson<Tokenization>(entry.tokenization),
    ])
      .then(([s, sm, t]) => {
        setSchema(s)
        setSample(sm)
        setTok(t)
      })
      .catch((e) => setError(errorMessage(e)))
  }, [entry])

  if (!index) {
    return (
      <AsyncBoundary
        loading={indexError === null}
        error={indexError}
        loadingLabel="Loading dataset index…"
        errorPrefix="Failed to load datasets/index.json"
      />
    )
  }

  const ready = schema !== null && sample !== null && tok !== null

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-3">
        {index.length > 1 && (
          <SubjectSwitcher
            label="Dataset"
            ariaLabel="Dataset"
            value={slug ?? ''}
            options={index.map((d) => ({ value: d.slug, label: d.label }))}
            onChange={setSlug}
          />
        )}
        <ViewTabs
          ariaLabel="Data views"
          value={tab}
          onChange={setTab}
          options={TABS.map((t) => ({ value: t.id, label: t.label }))}
        />
        {ready && (
          <span className="text-xs text-ink-muted">
            {sample.count} rows · source <code className="code-ref">{sample.source}</code>
          </span>
        )}
      </div>

      <AsyncBoundary
        loading={!ready}
        error={error}
        loadingLabel="Loading sample…"
        errorPrefix="Failed to load data"
      >
        {ready && (
          <>
            {tab === 'format' && <SchemaView schema={schema} />}
            {tab === 'samples' && <SampleBrowser rows={sample.rows} columns={schema.columns} />}
            {tab === 'stats' && <StatsView rows={sample.rows} />}
            {tab === 'tokenization' && <TokenizationView tok={tok} />}
          </>
        )}
      </AsyncBoundary>
    </div>
  )
}
