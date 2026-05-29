import { cn } from '@/lib/utils'

/**
 * Pill tablist for switching sub-views within a mode (Data tabs, Component
 * pager, Architecture lens). The canonical control vocabulary of the explorer —
 * reuse this instead of hand-rolling pill buttons.
 */
export function ViewTabs<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
}: {
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
  ariaLabel?: string
}) {
  return (
    <div className="flex flex-wrap gap-2" role="tablist" aria-label={ariaLabel}>
      {options.map(({ value: v, label }) => (
        <button
          key={v}
          type="button"
          role="tab"
          aria-selected={value === v}
          onClick={() => onChange(v)}
          className={cn(
            'rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors',
            value === v
              ? 'border-panelborder-active bg-panel-hover text-ink'
              : 'border-panelborder bg-panel text-ink-soft hover:text-ink',
          )}
        >
          {label}
        </button>
      ))}
    </div>
  )
}
