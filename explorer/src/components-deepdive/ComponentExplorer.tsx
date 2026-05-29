import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { fetchExplorerJson, errorMessage } from '@/lib/fetch'
import { cn } from '@/lib/utils'
import { AsyncBoundary } from '@/explorer-kit/AsyncBoundary'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import ArchitectureGraph from './ArchitectureGraph'
import ComponentDrawer from './ComponentDrawer'
import type {
  ComponentManifest,
  CoreNode,
  DrawerDetail,
  Hack,
  SectionNode,
} from './types'

type Page = 'flow' | 'subsystems'

export default function ComponentExplorer({
  onOpenArchitecture,
}: {
  onOpenArchitecture: () => void
}) {
  const [manifest, setManifest] = useState<ComponentManifest | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selected, setSelected] = useState<DrawerDetail | null>(null)
  const [page, setPage] = useState<Page>('flow')

  useEffect(() => {
    fetchExplorerJson<ComponentManifest>('components.json')
      .then(setManifest)
      .catch((e) => setError(errorMessage(e)))
  }, [])

  // Map a file -> the guide section that documents it (for attaching hacks).
  const sectionByFile = useMemo(() => {
    const m = new Map<string, SectionNode>()
    manifest?.sections.forEach((s) => {
      if (s.file && !m.has(s.file)) m.set(s.file, s)
    })
    return m
  }, [manifest])

  function hacksForSection(section: string | null): Hack[] {
    if (!manifest || !section) return []
    return manifest.hacks.filter((h) => h.section === section)
  }

  function fromNode(n: CoreNode): DrawerDetail {
    const sec = sectionByFile.get(n.file) ?? null
    return {
      title: n.label,
      file: n.file,
      line: n.line,
      symbol: n.symbol,
      section: sec?.section ?? null,
      sectionTitle: sec?.title ?? null,
      hacks: hacksForSection(sec?.section ?? null),
      isModelRunner: n.id === 'gpumodelrunner',
    }
  }

  function fromSection(s: SectionNode): DrawerDetail {
    return {
      title: s.title,
      file: s.file,
      line: s.line,
      symbol: s.symbol,
      section: s.section,
      sectionTitle: s.title,
      hacks: hacksForSection(s.section),
      isModelRunner: s.file?.includes('gpu_model_runner') ?? false,
    }
  }

  if (!manifest) {
    return (
      <AsyncBoundary
        loading={error === null}
        error={error}
        loadingLabel="Loading component manifest…"
        errorPrefix="Failed to load components.json"
      />
    )
  }

  const selectedId =
    selected && manifest.nodes.find((n) => n.label === selected.title)?.id
      ? manifest.nodes.find((n) => n.label === selected.title)!.id
      : null

  return (
    <div className="flex flex-col gap-5">
      {/* Pager — flow on one page, every subsystem on the next */}
      <div className="flex flex-wrap gap-2" role="tablist" aria-label="Component view">
        <PagerTab active={page === 'flow'} onClick={() => setPage('flow')}>
          Request flow
        </PagerTab>
        <PagerTab active={page === 'subsystems'} onClick={() => setPage('subsystems')}>
          All subsystems ({manifest.sections.length})
        </PagerTab>
      </div>

      {page === 'flow' ? (
        // Diagram fills the entire left half; detail sits at the top of the right half.
        <div className="grid items-start gap-5 lg:grid-cols-2">
          <Card className="flex flex-col">
            <CardHeader>
              <CardTitle>V1 engine — request flow</CardTitle>
              <p className="text-sm text-ink-soft">
                The §2 architecture. Click a box for its source location and runnable hack.
              </p>
            </CardHeader>
            <CardContent className="flex flex-1 items-center justify-center">
              <ArchitectureGraph
                nodes={manifest.nodes}
                edges={manifest.edges}
                selectedId={selectedId}
                onSelect={(n) => setSelected(fromNode(n))}
              />
            </CardContent>
          </Card>
          <div className="lg:sticky lg:top-6">
            <ComponentDrawer detail={selected} onOpenArchitecture={onOpenArchitecture} />
          </div>
        </div>
      ) : (
        // Subsystems list on the left; the same detail drawer on the right.
        <div className="grid items-start gap-5 lg:grid-cols-[1fr_360px]">
          <Card>
            <CardHeader>
              <CardTitle>All subsystems ({manifest.sections.length})</CardTitle>
              <p className="text-sm text-ink-soft">
                Every section of the guide — the major components beyond the core flow.
              </p>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {manifest.sections.map((s) => (
                  <button
                    key={s.id}
                    className={cn(
                      'rounded-lg border px-3 py-2 text-left transition-colors',
                      selected?.title === s.title
                        ? 'border-panelborder-active bg-panel-hover'
                        : 'border-panelborder bg-panel hover:border-panelborder-active',
                    )}
                    onClick={() => setSelected(fromSection(s))}
                  >
                    <div className="flex items-baseline gap-2">
                      <span className="font-mono text-xs text-cyan">§{s.section}</span>
                      <span className="text-sm font-medium text-ink">{s.title}</span>
                    </div>
                    {s.hacks.length > 0 && (
                      <div className="mt-1 font-mono text-[11px] text-success">
                        ▶ {s.hacks.map((h) => h.replace('hacks/', '')).join(', ')}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
          <div className="lg:sticky lg:top-6">
            <ComponentDrawer detail={selected} onOpenArchitecture={onOpenArchitecture} />
          </div>
        </div>
      )}
    </div>
  )
}

function PagerTab({
  active,
  onClick,
  children,
}: {
  active: boolean
  onClick: () => void
  children: ReactNode
}) {
  return (
    <button
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={cn(
        'rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors',
        active
          ? 'border-panelborder-active bg-panel-hover text-ink'
          : 'border-panelborder bg-panel text-ink-soft hover:text-ink',
      )}
    >
      {children}
    </button>
  )
}
