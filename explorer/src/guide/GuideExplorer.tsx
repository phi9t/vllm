import { useEffect, useMemo, useState } from 'react'
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
