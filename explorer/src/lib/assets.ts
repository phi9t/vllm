// Asset + repo URL helpers. Mirrors the sibling explorers' lib/assets.ts so the
// same data-loading contract holds: everything resolves against Vite's BASE_URL,
// which makes the app portable to sub-path static hosting (GitHub Pages, etc.).

export const REPO_HOME = 'https://github.com/phi9t/vllm'
export const HACKERS_GUIDE_URL = `${REPO_HOME}/blob/phi9t-mainline/HACKERS_GUIDE.md`

/** Resolve a public asset / data path against the configured base URL. */
export function dataUrl(path: string): string {
  const clean = path.replace(/^\//, '')
  return `${import.meta.env.BASE_URL}${clean}`
}

export function logoMarkUrl(): string {
  return dataUrl('logo-mark.svg')
}

/** Build a GitHub blob URL for a `path` (optionally `path:line`) on the fork. */
export function sourceUrl(fileLine: string, ref = 'phi9t-mainline'): string {
  const [file, line] = fileLine.split(':')
  const anchor = line ? `#L${line}` : ''
  return `${REPO_HOME}/blob/${ref}/${file}${anchor}`
}
