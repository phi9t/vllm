import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { FinewebRow } from './types'

interface Bin {
  label: string
  count: number
}

function countBy(rows: FinewebRow[], key: (r: FinewebRow) => string | undefined): Bin[] {
  const map = new Map<string, number>()
  for (const r of rows) {
    const k = key(r)
    if (k === undefined) continue
    map.set(k, (map.get(k) ?? 0) + 1)
  }
  return [...map.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : 1))
    .map(([label, count]) => ({ label, count }))
}

function histogram(
  rows: FinewebRow[],
  value: (r: FinewebRow) => number | undefined,
  edges: number[],
): Bin[] {
  const bins: Bin[] = edges.slice(0, -1).map((lo, i) => ({
    label: `${lo}–${edges[i + 1]}`,
    count: 0,
  }))
  for (const r of rows) {
    const v = value(r)
    if (v === undefined) continue
    for (let i = 0; i < edges.length - 1; i++) {
      if (v >= edges[i] && (v < edges[i + 1] || i === edges.length - 2)) {
        bins[i].count += 1
        break
      }
    }
  }
  return bins
}

function Chart({ title, data }: { title: string; data: Bin[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -16 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
            <XAxis dataKey="label" tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="rgba(255,255,255,0.1)" />
            <YAxis allowDecimals={false} tick={{ fill: '#9ca3af', fontSize: 11 }} stroke="rgba(255,255,255,0.1)" />
            <Tooltip
              cursor={{ fill: 'rgba(99,102,241,0.08)' }}
              contentStyle={{
                background: '#0d0f15',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: 8,
                fontSize: 12,
              }}
            />
            <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  )
}

export default function StatsView({ rows }: { rows: FinewebRow[] }) {
  const intScore = useMemo(() => countBy(rows, (r) => r.int_score?.toString()), [rows])
  const tokenCount = useMemo(
    () => histogram(rows, (r) => r.token_count, [0, 50, 100, 200, 400, 800, 2000]),
    [rows],
  )
  const langScore = useMemo(
    () => histogram(rows, (r) => r.language_score, [0.9, 0.92, 0.94, 0.96, 0.98, 1.0]),
    [rows],
  )

  return (
    <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
      <Chart title="Edu int_score distribution" data={intScore} />
      <Chart title="token_count (GPT-2) distribution" data={tokenCount} />
      <Chart title="language_score distribution" data={langScore} />
      <Card>
        <CardHeader>
          <CardTitle>Reading these</CardTitle>
        </CardHeader>
        <CardContent className="text-sm leading-relaxed text-ink-soft">
          fineweb-edu is filtered by an education-quality classifier; <code className="code-ref">int_score</code>{' '}
          (the rounded <code className="code-ref">score</code>) is the usual filter threshold. The{' '}
          <code className="code-ref">token_count</code> column is computed with the GPT-2 tokenizer — compare it
          against the Qwen3 count in the Tokenization tab.
        </CardContent>
      </Card>
    </div>
  )
}
