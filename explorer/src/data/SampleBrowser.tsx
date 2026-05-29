import { useState } from 'react'
import { cn } from '@/lib/utils'
import { Card, CardContent } from '@/components/ui/card'
import type { FinewebColumn, FinewebRow } from './types'

export default function SampleBrowser({
  rows,
}: {
  rows: FinewebRow[]
  columns: FinewebColumn[]
}) {
  const [selected, setSelected] = useState(0)
  const row = rows[selected]

  return (
    <div className="dashboard-grid">
      <Card className="max-h-[70vh] overflow-y-auto">
        <CardContent className="p-2">
          <ul className="flex flex-col gap-1">
            {rows.map((r, i) => (
              <li key={r.id ?? i}>
                <button
                  className={cn(
                    'w-full rounded-lg px-3 py-2 text-left text-sm transition-colors',
                    i === selected
                      ? 'bg-panel-hover text-ink'
                      : 'text-ink-soft hover:bg-panel-hover/60',
                  )}
                  onClick={() => setSelected(i)}
                  aria-pressed={i === selected}
                >
                  <div className="truncate font-medium">{r.text.slice(0, 48)}…</div>
                  <div className="mt-0.5 flex gap-2 text-[11px] text-ink-muted">
                    <span>score {r.score ?? '—'}</span>
                    <span>· {r.token_count ?? '—'} tok</span>
                    <span>· {r.language ?? '—'}</span>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {row && (
        <Card>
          <CardContent className="flex flex-col gap-4 p-5">
            <div className="flex flex-wrap gap-2">
              <Badge label="int_score" value={row.int_score} accent="cyan" />
              <Badge label="score" value={row.score} accent="primary" />
              <Badge label="token_count" value={row.token_count} />
              <Badge label="language_score" value={row.language_score} />
              <Badge label="language" value={row.language} />
            </div>
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
              <Meta label="id" value={row.id} />
              <Meta label="dump" value={row.dump} />
              <Meta label="url" value={row.url} />
              <Meta label="date" value={row.date} />
            </dl>
            <div>
              <div className="mb-1 text-xs font-semibold text-ink-muted">text</div>
              <p className="max-h-[42vh] overflow-y-auto whitespace-pre-wrap rounded-lg border border-panelborder bg-black/20 p-3 text-sm leading-relaxed text-ink">
                {row.text}
              </p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function Badge({
  label,
  value,
  accent,
}: {
  label: string
  value: string | number | undefined
  accent?: 'cyan' | 'primary'
}) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-md border border-panelborder px-2 py-1 text-[11px]',
        accent === 'cyan' && 'border-cyan/40 text-cyan',
        accent === 'primary' && 'border-primary/40 text-primary',
      )}
    >
      <span className="text-ink-muted">{label}</span>
      <span className="font-mono font-semibold">{value ?? '—'}</span>
    </span>
  )
}

function Meta({ label, value }: { label: string; value?: string }) {
  return (
    <>
      <dt className="font-mono text-ink-muted">{label}</dt>
      <dd className="truncate font-mono text-ink-soft">{value ?? '—'}</dd>
    </>
  )
}
