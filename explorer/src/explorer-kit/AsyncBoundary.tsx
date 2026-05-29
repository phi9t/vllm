import type { ReactNode } from 'react'

export function AsyncBoundary({
  loading,
  error,
  loadingLabel = 'Loading…',
  errorPrefix = 'Failed to load',
  children,
}: {
  loading: boolean
  error: string | null
  loadingLabel?: string
  errorPrefix?: string
  children?: ReactNode
}) {
  if (error) {
    return (
      <div className="panel p-6 text-danger">
        {errorPrefix}: {error}
        <p className="mt-2 text-sm text-ink-soft">
          Generate it first:{' '}
          <code className="code-ref">./scripts/workflow.sh gen-data</code>
        </p>
      </div>
    )
  }
  if (loading) {
    return <div className="panel p-6 text-ink-soft">{loadingLabel}</div>
  }
  return <>{children}</>
}
