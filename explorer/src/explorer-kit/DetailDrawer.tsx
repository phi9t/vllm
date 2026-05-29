import type { ReactNode } from 'react'
import { ExternalLink } from 'lucide-react'
import { Card } from '@/components/ui/card'
import { sourceUrl } from '@/lib/assets'

/**
 * Shared detail-drawer chrome used by every mode: the glass `.drawer` Card, an
 * empty state, an eyebrow + title, and an optional grounded source link
 * (`file:line` → GitHub blob via `sourceUrl`). Mode-specific body content is
 * passed as children, so each drawer reuses the chrome without a rigid schema.
 */
export function DetailDrawer({
  empty,
  emptyText,
  eyebrow,
  title,
  sourceRef,
  children,
}: {
  empty?: boolean
  emptyText?: string
  eyebrow?: ReactNode
  title?: string
  /** A "path/to/file.py:line" (or bare path) — rendered as a code-ref link. */
  sourceRef?: string | null
  children?: ReactNode
}) {
  if (empty) {
    return <Card className="drawer text-sm text-ink-muted">{emptyText}</Card>
  }
  return (
    <Card className="drawer">
      <div>
        {eyebrow != null && (
          <div className="text-[11px] uppercase tracking-wide text-ink-muted">{eyebrow}</div>
        )}
        {title && <h3 className="mt-1 text-lg font-bold text-ink">{title}</h3>}
      </div>
      {sourceRef && <SourceLink refStr={sourceRef} />}
      {children}
    </Card>
  )
}

/** A grounded source reference rendered as a code-ref → GitHub blob link. */
export function SourceLink({ refStr }: { refStr: string }) {
  return (
    <a
      href={sourceUrl(refStr)}
      target="_blank"
      rel="noopener noreferrer"
      className="inline-flex items-center gap-1.5 text-sm text-cyan hover:underline"
    >
      <code className="code-ref">{refStr}</code>
      <ExternalLink size={13} aria-hidden="true" />
    </a>
  )
}
