import { dataUrl } from './assets'

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err)
}

/** Fetch a JSON manifest from public/data/, typed as T. */
export async function fetchExplorerJson<T>(path: string): Promise<T> {
  const url = dataUrl(path.startsWith('data/') ? path : `data/${path}`)
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status} ${response.statusText}`)
  }
  return response.json() as Promise<T>
}

/** Format a numeric metric with magnitude-aware precision. */
export function formatMetric(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (Math.abs(value) >= 1000) return value.toLocaleString('en-US', { maximumFractionDigits: 0 })
  if (Math.abs(value) >= 100) return value.toFixed(1)
  if (Math.abs(value) >= 1) return value.toFixed(3)
  return value.toFixed(4)
}

/** Compact integer formatting, e.g. 12_500 -> "12.5K". */
export function formatCount(value: number): string {
  if (!Number.isFinite(value)) return '—'
  if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(1)}B`
  if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(1)}M`
  if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(1)}K`
  return String(value)
}
