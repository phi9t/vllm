import { cn } from '@/lib/utils'

export function SubjectSwitcher<T extends string>({
  label,
  value,
  options,
  onChange,
  ariaLabel,
}: {
  label?: string
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
  ariaLabel?: string
}) {
  const useSelect = options.length > 6

  return (
    <div className="flex flex-wrap items-center gap-2">
      {label && <span className="text-xs text-ink-muted">{label}</span>}
      {useSelect ? (
        <select
          value={value}
          onChange={(e) => onChange(e.target.value as T)}
          aria-label={ariaLabel}
          className="rounded-lg border border-panelborder bg-panel px-2 py-1.5 text-xs text-ink focus:outline-none focus:border-panelborder-active"
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>
      ) : (
        <div className="flex flex-wrap gap-2" role="tablist" aria-label={ariaLabel}>
          {options.map((opt) => (
            <button
              key={opt.value}
              type="button"
              role="tab"
              aria-selected={value === opt.value}
              onClick={() => onChange(opt.value)}
              className={cn(
                'rounded-lg border px-3 py-1.5 text-xs font-semibold transition-colors',
                value === opt.value
                  ? 'border-panelborder-active bg-panel-hover text-ink'
                  : 'border-panelborder bg-panel text-ink-soft hover:text-ink',
              )}
            >
              {opt.label}
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
