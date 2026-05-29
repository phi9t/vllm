// Qwen3 decoder as a compact residual-mainline diagram.
// The residual stream is a vertical rail flowing bottom (input) → top (output).
// Attention and FFN/SwiGLU branches tee off it, run a pre-norm + a stack of
// step cards, and merge back at a residual-add (+) node. Cards are sized close
// to body text; every card is selectable and carries its lens metric.
import { cn } from '@/lib/utils'
import { DECODER_LAYER, HEAD, PRELUDE, type Qwen3Block } from './qwen3Blocks'

const ALL: Record<string, Qwen3Block> = Object.fromEntries(
  [...PRELUDE, ...DECODER_LAYER, ...HEAD].map((b) => [b.id, b]),
)

const KIND_COLOR: Record<string, string> = {
  embed: '#06b6d4',
  norm: '#8b5cf6',
  proj: '#6366f1',
  rope: '#38bdf8',
  attn: '#10b981',
  act: '#f59e0b',
  mlp: '#6366f1',
  head: '#ef4444',
}

// Branch composition (ids resolve into qwen3Blocks). Order = inference flow.
const ATTN = { preNorm: 'input_norm', steps: ['qkv_proj', 'q_norm', 'k_norm', 'rope', 'attn', 'o_proj'] }
const FFN = { preNorm: 'post_norm', steps: ['gate_up', 'silu', 'down'] }

// --- Geometry (viewBox units ≈ rendered px; the SVG is width-capped) --------
const VIEW_W = 420
const MAINLINE_X = 104
const CARD_X = 290
const CARD_W = 184
const CARD_H = 30
const MAIN_W = 152
const MAIN_H = 28
const CARD_PITCH = 42
const PRENORM_GAP = 16
const JUNC_GAP = 18
const MAIN_PITCH = 50
const PAD = 44
const SHELL_PAD = 13
const SHELL_LEFT = CARD_X - CARD_W / 2 - SHELL_PAD
const SHELL_RIGHT = CARD_X + CARD_W / 2 + SHELL_PAD

/** Vertical span a branch occupies between its tee (bottom) and junction (top). */
const branchHeight = (n: number) => PRENORM_GAP + CARD_H / 2 + (n - 1) * CARD_PITCH + CARD_H / 2 + JUNC_GAP

type Lens = 'flow' | 'shapes' | 'compute' | 'memory'

export default function Qwen3Circuit({
  lens,
  selectedId,
  onSelect,
  lensValue,
  layers,
}: {
  lens: Lens
  selectedId: string
  onSelect: (id: string) => void
  lensValue: (id: string) => string
  layers: number
}) {
  // Mainline stops, top (output) → bottom (input).
  const yLogits = PAD
  const yLmHead = yLogits + MAIN_PITCH
  const yFinalNorm = yLmHead + MAIN_PITCH
  const yJ2 = yFinalNorm + MAIN_PITCH
  const yFfnTee = yJ2 + branchHeight(FFN.steps.length + 1)
  const yJ1 = yFfnTee + 34
  const yAttnTee = yJ1 + branchHeight(ATTN.steps.length + 1)
  const yEmbed = yAttnTee + MAIN_PITCH
  const height = yEmbed + PAD

  const bracketTop = (yFinalNorm + yJ2) / 2
  const bracketBot = (yAttnTee + yEmbed) / 2
  const metric = (id: string) => (lens === 'flow' ? '' : lensValue(id))

  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${height}`}
      className="mx-auto w-full max-w-[480px]"
      role="group"
      aria-label="Qwen3 decoder as a residual-mainline diagram; input enters at the bottom, output exits at the top"
    >
      {/* decoder-layer bracket */}
      <rect
        x={8}
        y={bracketTop}
        width={VIEW_W - 16}
        height={bracketBot - bracketTop}
        rx={14}
        fill="rgba(99,102,241,0.025)"
        stroke="rgba(99,102,241,0.18)"
        strokeDasharray="3 5"
      />
      <text x={16} y={bracketTop - 6} className="module-shell-title">
        decoder layer × {layers}
      </text>

      {/* residual mainline rail (bottom → top) */}
      <line className="mainline-rail" x1={MAINLINE_X} y1={yEmbed} x2={MAINLINE_X} y2={yLogits} />
      {Array.from({ length: 6 }).map((_, i) => {
        const cy = yEmbed - ((yEmbed - yLogits) * (i + 0.5)) / 6
        return (
          <circle
            key={i}
            className="mainline-tick"
            cx={MAINLINE_X}
            cy={cy}
            r={2.2}
            style={{ animationDelay: `${i * 0.16}s` }}
          />
        )
      })}
      <text
        x={MAINLINE_X - 26}
        y={(yJ1 + yJ2) / 2}
        textAnchor="middle"
        className="module-shell-title"
        transform={`rotate(-90, ${MAINLINE_X - 26}, ${(yJ1 + yJ2) / 2})`}
      >
        residual stream ↑
      </text>

      {/* Branches (under the nodes) */}
      <Branch
        title="Attention"
        branch={ATTN}
        yTee={yAttnTee}
        yJunc={yJ1}
        selectedId={selectedId}
        onSelect={onSelect}
        metric={metric}
      />
      <Branch
        title="FFN · SwiGLU"
        branch={FFN}
        yTee={yFfnTee}
        yJunc={yJ2}
        selectedId={selectedId}
        onSelect={onSelect}
        metric={metric}
      />

      {/* Residual-add nodes */}
      <PlusNode x={MAINLINE_X} y={yJ1} label="attn" />
      <PlusNode x={MAINLINE_X} y={yJ2} label="ffn" />

      {/* Mainline nodes */}
      {[
        { id: 'logits', y: yLogits },
        { id: 'lm_head', y: yLmHead },
        { id: 'final_norm', y: yFinalNorm },
        { id: 'embed', y: yEmbed },
      ].map(({ id, y }) => (
        <NodeCard
          key={id}
          id={id}
          cx={MAINLINE_X}
          cy={y}
          w={MAIN_W}
          h={MAIN_H}
          selected={selectedId === id}
          onSelect={onSelect}
          metric={metric(id)}
        />
      ))}

      <text x={MAINLINE_X} y={yEmbed + 26} textAnchor="middle" className="module-shell-title">
        input_ids
      </text>
      <text x={MAINLINE_X} y={yLogits - 20} textAnchor="middle" className="module-shell-title">
        output logits
      </text>
    </svg>
  )
}

// --- One residual branch ----------------------------------------------------
function Branch({
  title,
  branch,
  yTee,
  yJunc,
  selectedId,
  onSelect,
  metric,
}: {
  title: string
  branch: { preNorm: string; steps: string[] }
  yTee: number
  yJunc: number
  selectedId: string
  onSelect: (id: string) => void
  metric: (id: string) => string
}) {
  const cards = [branch.preNorm, ...branch.steps] // bottom → top
  const bottomCenter = yTee - PRENORM_GAP - CARD_H / 2
  const ys = cards.map((_, i) => bottomCenter - i * CARD_PITCH)
  const topY = ys[ys.length - 1]
  const shellTop = topY - CARD_H / 2 - 24
  const shellBot = ys[0] + CARD_H / 2 + 10

  return (
    <g>
      <rect
        className="module-shell"
        x={SHELL_LEFT}
        y={shellTop}
        width={SHELL_RIGHT - SHELL_LEFT}
        height={shellBot - shellTop}
        rx={12}
      />
      <text x={SHELL_LEFT} y={shellTop - 5} className="module-shell-title">
        {title}
      </text>

      {/* tee → up the card column → merge into the + node; cards thread the line */}
      <circle cx={MAINLINE_X} cy={yTee} r={3.5} fill="#38bdf8" />
      <path
        className="branch-wire"
        d={`M ${MAINLINE_X} ${yTee} H ${CARD_X} V ${yJunc} H ${MAINLINE_X}`}
      />

      {cards.map((id, i) => (
        <NodeCard
          key={id}
          id={id}
          cx={CARD_X}
          cy={ys[i]}
          w={CARD_W}
          h={CARD_H}
          selected={selectedId === id}
          onSelect={onSelect}
          metric={metric(id)}
        />
      ))}
    </g>
  )
}

// --- A clickable block card -------------------------------------------------
function NodeCard({
  id,
  cx,
  cy,
  w,
  h,
  selected,
  onSelect,
  metric,
}: {
  id: string
  cx: number
  cy: number
  w: number
  h: number
  selected: boolean
  onSelect: (id: string) => void
  metric: string
}) {
  const block = ALL[id]
  if (!block) return null
  const color = KIND_COLOR[block.kind] ?? '#6366f1'
  const x = cx - w / 2
  const y = cy - h / 2
  const hasMetric = metric !== ''
  const trimmed = metric.length > 30 ? metric.slice(0, 29) + '…' : metric

  return (
    <g
      className={cn('diagram-node', selected && 'active')}
      role="button"
      tabIndex={0}
      aria-label={block.label}
      onClick={() => onSelect(id)}
      onKeyDown={(ev) => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault()
          onSelect(id)
        }
      }}
    >
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={8}
        fill="rgba(15,21,33,0.96)"
        stroke={selected ? '#38bdf8' : color}
        strokeWidth={selected ? 2 : 1.2}
        strokeOpacity={selected ? 1 : 0.75}
      />
      <rect x={x} y={y + 4} width={3} height={h - 8} rx={1.5} fill={color} />
      <text
        className="diagram-label"
        x={x + 12}
        y={hasMetric ? cy - 2 : cy + 4}
        fontSize={12}
        fill="#f3f4f6"
      >
        {block.label}
      </text>
      {hasMetric && (
        <text className="diagram-label" x={x + 12} y={cy + 9} fontSize={8.5} fill="#9ca3af">
          {trimmed}
        </text>
      )}
    </g>
  )
}

// --- Residual-add node (a crisp +, not a glyph) -----------------------------
function PlusNode({ x, y, label }: { x: number; y: number; label: string }) {
  return (
    <g aria-hidden="true">
      <circle cx={x} cy={y} r={8.5} fill="rgba(15,21,33,0.98)" stroke="#8b5cf6" strokeWidth={1.4} />
      <line className="junction-plus" x1={x - 4} y1={y} x2={x + 4} y2={y} />
      <line className="junction-plus" x1={x} y1={y - 4} x2={x} y2={y + 4} />
      <text x={x - 14} y={y + 3} textAnchor="end" className="module-shell-title">
        {label}
      </text>
    </g>
  )
}
