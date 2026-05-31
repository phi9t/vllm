import { useEffect, useId, useMemo, useRef, useState } from 'react'
import ReactMarkdown, { type Components } from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSlug from 'rehype-slug'
import { fetchExplorerText, errorMessage } from '@/lib/fetch'
import { REPO_HOME } from '@/lib/assets'
import { AsyncBoundary } from '@/explorer-kit/AsyncBoundary'
import { Card } from '@/components/ui/card'
import type { ExplorerModeProps } from '@/explorer-kit/mode'

const REF = 'phi9t-mainline'

/** Rewrite the guide's repo-relative links to GitHub blob/tree URLs; keep
 *  in-page anchors and external links as-is. */
function rewriteHref(href?: string): { href: string; external: boolean } {
  if (!href) return { href: '#', external: false }
  if (href.startsWith('#')) return { href, external: false }
  if (/^https?:\/\//.test(href) || href.startsWith('mailto:')) {
    return { href, external: true }
  }
  const clean = href.replace(/^\.?\//, '')
  const kind = clean.endsWith('/') ? 'tree' : 'blob'
  return { href: `${REPO_HOME}/${kind}/${REF}/${clean}`, external: true }
}

/** Match GitHub's heading slugs (github-slugger, as used by rehype-slug):
 *  lowercase, drop everything but [a-z0-9 -], spaces -> '-', no collapsing. */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .replace(/[^a-z0-9 -]/g, '')
    .replace(/ /g, '-')
}

// --- Mermaid: lazy-loaded, themed to Observatory --------------------------------
let mermaidReady: Promise<typeof import('mermaid').default> | null = null
function loadMermaid() {
  if (!mermaidReady) {
    mermaidReady = import('mermaid').then(({ default: mermaid }) => {
      mermaid.initialize({
        startOnLoad: false,
        securityLevel: 'strict',
        theme: 'base',
        themeVariables: {
          darkMode: true,
          fontFamily: 'Fira Code Variable, ui-monospace, monospace',
          fontSize: '13px',
          background: '#0b0f19',
          primaryColor: '#161d2e',
          primaryBorderColor: '#6366f1',
          primaryTextColor: '#f3f4f6',
          secondaryColor: '#16233a',
          tertiaryColor: '#1d1838',
          lineColor: '#38bdf8',
          textColor: '#cbd5e1',
          titleColor: '#f3f4f6',
          clusterBkg: 'rgba(99,102,241,0.06)',
          clusterBorder: 'rgba(99,102,241,0.35)',
          edgeLabelBackground: '#0b0f19',
        },
      })
      return mermaid
    })
  }
  return mermaidReady
}

/** Render one ```mermaid block to SVG; fall back to the source on error. */
function Mermaid({ chart }: { chart: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [failed, setFailed] = useState(false)
  const rawId = useId()
  const id = 'mmd-' + rawId.replace(/[^a-zA-Z0-9]/g, '')

  useEffect(() => {
    let cancelled = false
    loadMermaid()
      .then((mermaid) => mermaid.render(id, chart))
      .then(({ svg }) => {
        if (!cancelled && ref.current) ref.current.innerHTML = svg
      })
      .catch(() => {
        if (!cancelled) setFailed(true)
      })
    return () => {
      cancelled = true
    }
  }, [chart, id])

  if (failed) {
    return <pre className="guide-mermaid-src">{chart}</pre>
  }
  return <div className="guide-mermaid" role="img" aria-label="diagram" ref={ref} />
}

function isMermaidNode(node: unknown): boolean {
  // hast <pre> node whose first child is <code class="language-mermaid">
  const child = (node as { children?: { properties?: { className?: unknown } }[] })
    ?.children?.[0]
  const cls = child?.properties?.className
  return Array.isArray(cls) && cls.some((c) => String(c).includes('language-mermaid'))
}

const components: Components = {
  a({ href, children, node: _node, ...rest }) {
    const r = rewriteHref(href)
    const ext = r.external ? { target: '_blank', rel: 'noopener noreferrer' } : {}
    return (
      <a href={r.href} {...ext} {...rest}>
        {children}
      </a>
    )
  },
  code({ className, children, node: _node, ...rest }) {
    if (/\blanguage-mermaid\b/.test(className || '')) {
      return <Mermaid chart={String(children).trim()} />
    }
    return (
      <code className={className} {...rest}>
        {children}
      </code>
    )
  },
  pre({ node, children, ...rest }) {
    // Mermaid blocks render their own <div>; don't wrap them in <pre>.
    if (isMermaidNode(node)) return <>{children}</>
    return <pre {...rest}>{children}</pre>
  },
}

export default function GuideExplorer(_: ExplorerModeProps) {
  const [md, setMd] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchExplorerText('guide.md')
      .then(setMd)
      .catch((e) => setError(errorMessage(e)))
  }, [])

  // Sidebar TOC built from the top-level `## N.` headings.
  const toc = useMemo(() => {
    if (!md) return []
    const out: { num: string; title: string; id: string }[] = []
    for (const line of md.split('\n')) {
      const m = /^## (\d+)\. (.+)$/.exec(line)
      if (m) {
        out.push({
          num: m[1],
          title: m[2].replace(/`/g, ''),
          id: slugify(`${m[1]}. ${m[2].replace(/`/g, '')}`),
        })
      }
    }
    return out
  }, [md])

  if (!md) {
    return (
      <AsyncBoundary
        loading={error === null}
        error={error}
        loadingLabel="Loading the Hacker's Guide…"
        errorPrefix="Failed to load guide.md (run gen-data)"
      />
    )
  }

  return (
    <div className="grid items-start gap-5 lg:grid-cols-[248px_1fr]">
      <nav aria-label="Guide contents" className="hidden lg:block lg:sticky lg:top-6">
        <Card className="p-4">
          <div className="mb-2 text-[11px] uppercase tracking-wide text-ink-muted">
            Contents
          </div>
          <ol className="flex flex-col gap-1">
            {toc.map((s) => (
              <li key={s.id}>
                <a
                  href={`#${s.id}`}
                  className="block truncate text-xs text-ink-soft transition-colors hover:text-cyan"
                  title={s.title}
                >
                  <span className="text-ink-muted">{s.num}.</span> {s.title}
                </a>
              </li>
            ))}
          </ol>
        </Card>
      </nav>

      <Card className="guide p-6 md:p-8">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          rehypePlugins={[rehypeSlug]}
          components={components}
        >
          {md}
        </ReactMarkdown>
      </Card>
    </div>
  )
}
