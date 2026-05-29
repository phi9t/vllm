import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import type { FinewebSchema } from './types'

export default function SchemaView({ schema }: { schema: FinewebSchema }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>{schema.dataset} — record format</CardTitle>
        <p className="text-sm text-ink-soft">
          Each row is one cleaned web document plus crawl provenance and quality scores.
        </p>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="text-ink-muted">
                <th className="border-b border-panelborder px-3 py-2 font-semibold">Column</th>
                <th className="border-b border-panelborder px-3 py-2 font-semibold">Type</th>
                <th className="border-b border-panelborder px-3 py-2 font-semibold">Description</th>
              </tr>
            </thead>
            <tbody>
              {schema.columns.map((c) => (
                <tr key={c.name} className="align-top">
                  <td className="border-b border-panelborder/60 px-3 py-2">
                    <code className="code-ref">{c.name}</code>
                  </td>
                  <td className="border-b border-panelborder/60 px-3 py-2 font-mono text-xs text-secondary">
                    {c.type}
                  </td>
                  <td className="border-b border-panelborder/60 px-3 py-2 text-ink-soft">
                    {c.desc}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  )
}
