// Qwen3 decoder rendered as a Lumina-style residual-mainline circuit.
// The residual stream is a vertical rail flowing bottom (input) → top (output).
// Two branches tee off it — Attention and FFN/SwiGLU — each a pre-norm plus a
// vertical chain of step cards inside a labelled shell, merging back at a ⊕
// junction. Every card is selectable and carries its lens metric.
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

// --- Geometry ---------------------------------------------------------------
const VIEW_W = 440
const MAINLINE_X = 116
const SPINE_X = 158
const CARD_X = 312
const CARD_W = 196
const CARD_H = 38
const MAIN_W = 168
const MAIN_H = 34
const CARD_PITCH = 54
const PRENORM_GAP = 20
const JUNC_GAP = 22
const MAIN_PITCH = 60
const PAD = 58
const SHELL_LEFT = SPINE_X - 16
const SHELL_RIGHT = CARD_X + CARD_W / 2 + 14

/** Vertical span a branch occupies between its tee (bottom) and junction (top). */
function branchHeight(cardCount: number): number {
  return PRENORM_GAP + CARD_H / 2 + (cardCount - 1) * CARD_PITCH + CARD_H / 2 + JUNC_GAP
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
  // Lay out mainline stops from top (output) to bottom (input).
  const yLogits = PAD
  const yLmHead = yLogits + MAIN_PITCH
  const yFinalNorm = yLmHead + MAIN_PITCH
  const yJ2 = yFinalNorm + MAIN_PITCH // x ⊕ ffn
  const ffnH = branchHeight(FFN.steps.length + 1)
  const yFfnTee = yJ2 + ffnH
  const yJ1 = yFfnTee + 40 // x ⊕ attn
  const attnH = branchHeight(ATTN.steps.length + 1)
  const yAttnTee = yJ1 + attnH
  const yEmbed = yAttnTee + MAIN_PITCH
  const height = yEmbed + PAD

  // decoder-layer bracket spans the attention+ffn region
  const bracketTop = (yFinalNorm + yJ2) / 2
  const bracketBot = (yAttnTee + yEmbed) / 2

  const metric = (id: string) => (lens === 'flow' ? '' : lensValue(id))

  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${height}`}
      className="w-full"
      role="img"
      aria-label="Qwen3 decoder as a residual-mainline circuit; input enters at the bottom, output exits at the top"
    >
      <defs aria-hidden="true">
        <marker id="circuit-up" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto">
          <path d="M0,8 L4,0 L8,8 Z" className="circuit-arrow-head" />
        </marker>
      </defs>

      {/* decoder-layer bracket */}
      <rect
        x={8}
        y={bracketTop}
        width={VIEW_W - 16}
        height={bracketBot - bracketTop}
        rx={12}
        fill="rgba(99,102,241,0.03)"
        stroke="rgba(99,102,241,0.22)"
        strokeDasharray="4 4"
      />
      <text x={16} y={bracketTop - 6} className="module-shell-title" fill="#9ca3af">
        decoder layer × {layers}
      </text>

      {/* residual mainline rail (bottom → top) */}
      <line className="mainline-rail" x1={MAINLINE_X} y1={yEmbed} x2={MAINLINE_X} y2={yLogits} />
      {Array.from({ length: 7 }).map((_, i) => {
        const cy = yEmbed - ((yEmbed - yLogits) * (i + 0.5)) / 7
        return (
          <circle
            key={i}
            className="mainline-tick"
            cx={MAINLINE_X}
            cy={cy}
            r={2.4}
            style={{ animationDelay: `${i * 0.14}s` }}
          />
        )
      })}
      <text
        x={MAINLINE_X - 30}
        y={(yJ1 + yJ2) / 2}
        textAnchor="middle"
        className="module-shell-title"
        transform={`rotate(-90, ${MAINLINE_X - 30}, ${(yJ1 + yJ2) / 2})`}
      >
        residual stream ↑
      </text>

      {/* Branches (drawn before nodes so wires sit under cards) */}
      <Branch
        title="Attention"
        yTee={yAttnTee}
        yJunc={yJ1}
        preNormId={ATTN.preNorm}
        stepIds={ATTN.steps}
        selectedId={selectedId}
        onSelect={onSelect}
        metric={metric}
      />
      <Branch
        title="FFN · SwiGLU"
        yTee={yFfnTee}
        yJunc={yJ2}
        preNormId={FFN.preNorm}
        stepIds={FFN.steps}
        selectedId={selectedId}
        onSelect={onSelect}
        metric={metric}
      />

      {/* Junction (residual add) nodes */}
      <Junction x={MAINLINE_X} y={yJ1} label="x ⊕ attn" />
      <Junction x={MAINLINE_X} y={yJ2} label="x ⊕ ffn" />

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

      {/* IN / OUT captions */}
      <text x={MAINLINE_X} y={yEmbed + 30} textAnchor="middle" className="module-shell-title" fill="#6b7280">
        input_ids ↑
      </text>
      <text x={MAINLINE_X} y={yLogits - 24} textAnchor="middle" className="module-shell-title" fill="#6b7280">
        logits ↑ (output)
      </text>
    </svg>
  )
}

// --- One residual branch ----------------------------------------------------
function Branch({
  title,
  yTee,
  yJunc,
  preNormId,
  stepIds,
  selectedId,
  onSelect,
  metric,
}: {
  title: string
  yTee: number
  yJunc: number
  preNormId: string
  stepIds: string[]
  selectedId: string
  onSelect: (id: string) => void
  metric: (id: string) => string
}) {
  const cards = [preNormId, ...stepIds] // bottom → top
  const bottomCenter = yTee - PRENORM_GAP - CARD_H / 2
  const ys = cards.map((_, i) => bottomCenter - i * CARD_PITCH)
  const topY = ys[ys.length - 1]
  const shellTop = topY - CARD_H / 2 - 30
  const shellBot = ys[0] + CARD_H / 2 + 14
  const cardLeft = CARD_X - CARD_W / 2

  return (
    <g>
      {/* shell */}
      <rect
        className="module-shell"
        x={SHELL_LEFT}
        y={shellTop}
        width={SHELL_RIGHT - SHELL_LEFT}
        height={shellBot - shellTop}
        rx={10}
      />
      <text x={SHELL_LEFT + 12} y={shellTop + 18} className="module-shell-title">
        {title}
      </text>

      {/* tee off the mainline, spine, merge back into junction */}
      <circle cx={MAINLINE_X} cy={yTee} r={4} fill="#38bdf8" />
      <path className="branch-wire" d={`M ${MAINLINE_X} ${yTee} H ${SPINE_X}`} />
      <path className="branch-wire" d={`M ${SPINE_X} ${yJunc} H ${MAINLINE_X}`} />
      <line className="branch-spine" x1={SPINE_X} y1={yTee} x2={SPINE_X} y2={yJunc} />

      {/* upward flow arrows along the spine, between successive card taps */}
      {ys.slice(0, -1).map((y, i) => (
        <line
          key={i}
          className="branch-spine"
          x1={SPINE_X}
          y1={y - 6}
          x2={SPINE_X}
          y2={ys[i + 1] + 6}
          markerEnd="url(#circuit-up)"
        />
      ))}

      {/* taps + cards */}
      {cards.map((id, i) => (
        <g key={id}>
          <path className="branch-wire" d={`M ${SPINE_X} ${ys[i]} H ${cardLeft}`} />
          <NodeCard
            id={id}
            cx={CARD_X}
            cy={ys[i]}
            w={CARD_W}
            h={CARD_H}
            selected={selectedId === id}
            onSelect={onSelect}
            metric={metric(id)}
          />
        </g>
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
  const trimmed = metric.length > 28 ? metric.slice(0, 27) + '…' : metric

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
        rx={9}
        fill="rgba(17,24,39,0.92)"
        stroke={selected ? '#38bdf8' : color}
        strokeWidth={selected ? 2.5 : 1.4}
      />
      <rect x={x} y={y} width={4} height={h} rx={2} fill={color} />
      <text
        className="diagram-label"
        x={x + 14}
        y={metric ? cy - 3 : cy + 4}
        fontSize={12}
        fill="#f3f4f6"
      >
        {block.label}
      </text>
      {metric && (
        <text className="diagram-label" x={x + 14} y={cy + 11} fontSize={9} fill="#9ca3af">
          {trimmed}
        </text>
      )}
    </g>
  )
}

// --- Residual-add junction (decorative) -------------------------------------
function Junction({ x, y, label }: { x: number; y: number; label: string }) {
  return (
    <g aria-hidden="true">
      <circle cx={x} cy={y} r={15} fill="rgba(17,24,39,0.95)" stroke="#8b5cf6" strokeWidth={1.6} />
      <text x={x} y={y + 5} textAnchor="middle" className="junction-label" fontSize={14}>
        ⊕
      </text>
      <text x={x - 22} y={y + 4} textAnchor="end" className="module-shell-title">
        {label}
      </text>
    </g>
  )
}
