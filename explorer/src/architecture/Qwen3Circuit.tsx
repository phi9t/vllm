// Qwen3 decoder as a compact residual-mainline diagram (Observatory glass).
// Residual stream is a vertical rail flowing bottom (input) → top (output).
// Attention and FFN/SwiGLU branches tee off it, run a pre-norm + a stack of
// step cards inside a tinted panel, and merge back at a residual-add (+) node.
// Cards are sized to fit their full text; every card is selectable and carries
// its lens metric.
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
const ATTN_ACCENT = '#10b981'
const FFN_ACCENT = '#6366f1'

// --- Geometry (viewBox units ≈ rendered px; the SVG is width-capped) --------
const VIEW_W = 510
const MAINLINE_X = 98
const CARD_X = 326
const CARD_W = 252
const CARD_H = 32
const MAIN_W = 164
const MAIN_H = 30
const CARD_PITCH = 44
const PRENORM_GAP = 16
const JUNC_GAP = 18
const MAIN_PITCH = 52
const PAD = 46
const SHELL_PAD = 16
const SHELL_LEFT = CARD_X - CARD_W / 2 - SHELL_PAD
const SHELL_RIGHT = CARD_X + CARD_W / 2 + SHELL_PAD

/** Vertical span a branch occupies between its tee (bottom) and junction (top). */
const branchHeight = (n: number) => PRENORM_GAP + CARD_H / 2 + (n - 1) * CARD_PITCH + CARD_H / 2 + JUNC_GAP

/** Orthogonal SVG path with rounded (quadratic) corners. */
function roundedOrthPath(pts: [number, number][], r = 12): string {
  if (pts.length < 2) return ''
  let d = `M ${pts[0][0]} ${pts[0][1]}`
  for (let i = 1; i < pts.length - 1; i++) {
    const [px, py] = pts[i - 1]
    const [cx, cy] = pts[i]
    const [nx, ny] = pts[i + 1]
    const before: [number, number] = [cx - Math.sign(cx - px) * r, cy - Math.sign(cy - py) * r]
    const after: [number, number] = [cx + Math.sign(nx - cx) * r, cy + Math.sign(ny - cy) * r]
    d += ` L ${before[0]} ${before[1]} Q ${cx} ${cy} ${after[0]} ${after[1]}`
  }
  const last = pts[pts.length - 1]
  d += ` L ${last[0]} ${last[1]}`
  return d
}

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
      className="mx-auto w-full max-w-[560px]"
      role="group"
      aria-label="Qwen3 decoder as a residual-mainline diagram; input enters at the bottom, output exits at the top"
    >
      <defs aria-hidden="true">
        {Object.entries(KIND_COLOR).map(([kind, color]) => (
          <linearGradient key={kind} id={`grad-${kind}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.2} />
            <stop offset="100%" stopColor={color} stopOpacity={0.035} />
          </linearGradient>
        ))}
        <linearGradient id="rail-grad" gradientUnits="userSpaceOnUse" x1="0" y1={yEmbed} x2="0" y2={yLogits}>
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#38bdf8" />
        </linearGradient>
      </defs>

      {/* decoder-layer bracket */}
      <rect
        x={8}
        y={bracketTop}
        width={VIEW_W - 16}
        height={bracketBot - bracketTop}
        rx={16}
        fill="rgba(99,102,241,0.02)"
        stroke="rgba(99,102,241,0.16)"
        strokeDasharray="2 6"
      />
      <text x={18} y={bracketTop - 7} className="module-shell-title">
        decoder layer × {layers}
      </text>

      {/* residual mainline rail (bottom → top) */}
      <line
        className="mainline-rail"
        x1={MAINLINE_X}
        y1={yEmbed}
        x2={MAINLINE_X}
        y2={yLogits}
        stroke="url(#rail-grad)"
      />
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
        x={MAINLINE_X - 24}
        y={(yJ1 + yJ2) / 2}
        textAnchor="middle"
        className="module-shell-title"
        transform={`rotate(-90, ${MAINLINE_X - 24}, ${(yJ1 + yJ2) / 2})`}
      >
        residual stream ↑
      </text>

      {/* Branches (under the nodes) */}
      <Branch
        title="Attention"
        accent={ATTN_ACCENT}
        branch={ATTN}
        yTee={yAttnTee}
        yJunc={yJ1}
        selectedId={selectedId}
        onSelect={onSelect}
        metric={metric}
      />
      <Branch
        title="FFN · SwiGLU"
        accent={FFN_ACCENT}
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
          tag={false}
          selected={selectedId === id}
          onSelect={onSelect}
          metric={metric(id)}
        />
      ))}

      <text x={MAINLINE_X} y={yEmbed + 28} textAnchor="middle" className="module-shell-title">
        input_ids
      </text>
      <text x={MAINLINE_X} y={yLogits - 22} textAnchor="middle" className="module-shell-title">
        output logits
      </text>
    </svg>
  )
}

// --- One residual branch ----------------------------------------------------
function Branch({
  title,
  accent,
  branch,
  yTee,
  yJunc,
  selectedId,
  onSelect,
  metric,
}: {
  title: string
  accent: string
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
  const shellTop = topY - CARD_H / 2 - 26
  const shellBot = ys[0] + CARD_H / 2 + 12
  const pillW = title.length * 6.2 + 18

  return (
    <g>
      {/* tinted branch panel */}
      <rect
        x={SHELL_LEFT}
        y={shellTop}
        width={SHELL_RIGHT - SHELL_LEFT}
        height={shellBot - shellTop}
        rx={14}
        fill={accent}
        fillOpacity={0.05}
        stroke={accent}
        strokeOpacity={0.22}
      />
      {/* header pill straddling the top border */}
      <rect
        x={SHELL_LEFT + 14}
        y={shellTop - 9}
        width={pillW}
        height={18}
        rx={9}
        fill="rgba(11,15,25,0.98)"
        stroke={accent}
        strokeOpacity={0.45}
      />
      <text x={SHELL_LEFT + 14 + pillW / 2} y={shellTop + 3.5} textAnchor="middle" className="branch-pill-text" fill={accent}>
        {title}
      </text>

      {/* tee → up the card column → merge into the + node; cards thread the line */}
      <circle cx={MAINLINE_X} cy={yTee} r={3.5} fill="#38bdf8" />
      <path
        className="branch-wire"
        fill="none"
        d={roundedOrthPath(
          [
            [MAINLINE_X, yTee],
            [CARD_X, yTee],
            [CARD_X, yJunc],
            [MAINLINE_X, yJunc],
          ],
          12,
        )}
      />

      {cards.map((id, i) => (
        <NodeCard
          key={id}
          id={id}
          cx={CARD_X}
          cy={ys[i]}
          w={CARD_W}
          h={CARD_H}
          tag
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
  tag,
  selected,
  onSelect,
  metric,
}: {
  id: string
  cx: number
  cy: number
  w: number
  h: number
  tag: boolean
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
      {/* dark base + kind-tinted gradient + crisp border */}
      <rect x={x} y={y} width={w} height={h} rx={10} fill="rgba(13,18,30,0.96)" />
      <rect x={x} y={y} width={w} height={h} rx={10} fill={`url(#grad-${block.kind})`} />
      <rect x={x + 6} y={y + 6} width={3} height={h - 12} rx={1.5} fill={color} />
      <rect
        className="card-border"
        x={x}
        y={y}
        width={w}
        height={h}
        rx={10}
        fill="none"
        stroke={selected ? '#38bdf8' : color}
        strokeWidth={selected ? 2 : 1.2}
        strokeOpacity={selected ? 1 : 0.7}
      />
      <text
        className="diagram-label"
        x={x + 16}
        y={hasMetric ? cy - 2 : cy + 4}
        fontSize={12}
        fontWeight={600}
        fill="#f3f4f6"
      >
        {block.label}
      </text>
      {hasMetric && (
        <text className="diagram-label" x={x + 16} y={cy + 10} fontSize={9} fill="#9ca3af">
          {metric}
        </text>
      )}
      {tag && (
        <text
          className="kind-tag"
          x={x + w - 10}
          y={y + 13}
          textAnchor="end"
          fill={color}
        >
          {block.kind.toUpperCase()}
        </text>
      )}
    </g>
  )
}

// --- Residual-add node (a crisp +, not a glyph) -----------------------------
function PlusNode({ x, y, label }: { x: number; y: number; label: string }) {
  return (
    <g aria-hidden="true">
      <circle cx={x} cy={y} r={11} fill="none" stroke="#8b5cf6" strokeOpacity={0.25} strokeWidth={3} />
      <circle cx={x} cy={y} r={8.5} fill="rgba(13,18,30,0.98)" stroke="#8b5cf6" strokeWidth={1.4} />
      <line className="junction-plus" x1={x - 4} y1={y} x2={x + 4} y2={y} />
      <line className="junction-plus" x1={x} y1={y - 4} x2={x} y2={y + 4} />
      <text x={x - 16} y={y + 3} textAnchor="end" className="module-shell-title">
        {label}
      </text>
    </g>
  )
}
