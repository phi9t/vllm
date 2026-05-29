import { ArrowRight, ExternalLink, Terminal } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { DetailDrawer } from '@/explorer-kit/DetailDrawer'
import { HACKERS_GUIDE_URL, sourceUrl } from '@/lib/assets'
import type { DrawerDetail } from './types'

export default function ComponentDrawer({
  detail,
  onOpenArchitecture,
}: {
  detail: DrawerDetail | null
  onOpenArchitecture: () => void
}) {
  const sourceRef = detail?.file
    ? detail.line
      ? `${detail.file}:${detail.line}`
      : detail.file
    : null

  return (
    <DetailDrawer
      empty={!detail}
      emptyText="Select a component in the graph or a subsystem below to see its source location and runnable hack."
      eyebrow={detail?.section ? `§${detail.section} · ${detail.sectionTitle ?? ''}` : 'component'}
      title={detail?.title}
      sourceRef={sourceRef}
    >
      {detail && (
        <>
          {detail.symbol && (
            <div className="text-sm">
              <span className="text-ink-muted">symbol </span>
              <code className="code-ref">{detail.symbol}</code>
            </div>
          )}

          {detail.hacks.length > 0 && (
            <div className="flex flex-col gap-2">
              <div className="text-[11px] uppercase tracking-wide text-ink-muted">▶ Try it</div>
              {detail.hacks.map((h) => (
                <a
                  key={h.script}
                  href={sourceUrl(h.script)}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-start gap-2 rounded-lg border border-panelborder bg-panel px-3 py-2 text-sm hover:border-panelborder-active"
                >
                  <Terminal size={14} className="mt-0.5 shrink-0 text-success" aria-hidden="true" />
                  <span>
                    <code className="code-ref">{h.script.replace('hacks/', '')}</code>
                    <span className="mt-0.5 block text-xs text-ink-soft">{h.desc}</span>
                  </span>
                </a>
              ))}
            </div>
          )}

          <a
            href={HACKERS_GUIDE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 text-xs text-ink-soft hover:text-cyan"
          >
            Open HACKERS_GUIDE.md <ExternalLink size={12} aria-hidden="true" />
          </a>

          {detail.isModelRunner && (
            <Button onClick={onOpenArchitecture} className="mt-1 w-full">
              Open model architecture deep dive <ArrowRight size={15} aria-hidden="true" />
            </Button>
          )}
        </>
      )}
    </DetailDrawer>
  )
}
