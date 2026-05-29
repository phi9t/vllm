import { cn } from '@/lib/utils'
import type { CoreNode, Edge } from './types'

// Stable layout for the §2 core nodes (viewBox 640 x 560). Falls back to a
// simple grid for any unknown ids so the graph still renders if the guide changes.
const POS: Record<string, { x: number; y: number }> = {
  'llm-asyncllm': { x: 245, y: 16 },
  enginecoreproc: { x: 245, y: 92 },
  'enginecore-step': { x: 245, y: 168 },
  outputprocessor: { x: 455, y: 168 },
  scheduler: { x: 35, y: 260 },
  kvcachemanager: { x: 35, y: 344 },
  executor: { x: 245, y: 260 },
  gpumodelrunner: { x: 245, y: 344 },
  sampler: { x: 245, y: 428 },
}

const W = 150
const H = 46

function center(id: string, idx: number) {
  const p = POS[id] ?? { x: 35 + (idx % 3) * 210, y: 16 + Math.floor(idx / 3) * 84 }
  return { cx: p.x + W / 2, cy: p.y + H / 2, ...p }
}

export default function ArchitectureGraph({
  nodes,
  edges,
  selectedId,
  onSelect,
}: {
  nodes: CoreNode[]
  edges: Edge[]
  selectedId: string | null
  onSelect: (n: CoreNode) => void
}) {
  const centers = new Map(nodes.map((n, i) => [n.id, center(n.id, i)]))

  return (
    <svg viewBox="0 0 640 500" className="w-full" role="img" aria-label="V1 engine component graph">
      {edges.map((e, i) => {
        const a = centers.get(e.from)
        const b = centers.get(e.to)
        if (!a || !b) return null
        return <line key={i} className="diagram-edge" x1={a.cx} y1={a.cy} x2={b.cx} y2={b.cy} />
      })}

      {nodes.map((n, i) => {
        const p = center(n.id, i)
        const active = n.id === selectedId
        return (
          <g
            key={n.id}
            className={cn('diagram-node', active && 'active')}
            transform={`translate(${p.x},${p.y})`}
            onClick={() => onSelect(n)}
            role="button"
            tabIndex={0}
            aria-label={n.label}
            onKeyDown={(ev) => {
              if (ev.key === 'Enter' || ev.key === ' ') onSelect(n)
            }}
          >
            <rect
              width={W}
              height={H}
              rx={10}
              fill="rgba(17,24,39,0.92)"
              stroke={active ? '#38bdf8' : 'rgba(99,102,241,0.5)'}
              strokeWidth={active ? 2.5 : 1.5}
            />
            <text className="diagram-label" x={W / 2} y={H / 2 + 4} textAnchor="middle">
              {n.label}
            </text>
          </g>
        )
      })}
    </svg>
  )
}
