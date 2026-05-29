// Generic residual-mainline diagram (Observatory glass).
// Ported from Qwen3Circuit.tsx and generalized to render any ModelArch manifest.
// Residual stream flows bottom (input) → top (output).
// Branches tee off the mainline, run a pre-norm + step cards inside a tinted panel,
// and merge back at a residual-add (+) node. Cards are selectable and carry lens metrics.
import { cn } from '@/lib/utils'
import { KIND_COLOR } from './blockTypes'
import type { Block, LayerGroup, ModelArch } from './modelArch'

export type Lens = 'flow' | 'shapes' | 'compute' | 'memory'

// --- Geometry (viewBox units ≈ rendered px; the SVG is width-capped) ---------
const VIEW_W = 510
const MAINLINE_X = 106
const CARD_X = 326
const CARD_W = 252
const CARD_H = 32
const MAIN_W = 196
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
const branchHeight = (n: number) =>
  PRENORM_GAP + CARD_H / 2 + (n - 1) * CARD_PITCH + CARD_H / 2 + JUNC_GAP

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

// ---------------------------------------------------------------------------
// Props

interface CircuitProps {
  manifest: ModelArch
  lens: Lens
  selectedId: string
  onSelect: (id: string) => void
  lensValue: (id: string) => string
}

export default function ModelCircuit({
  manifest,
  lens,
  selectedId,
  onSelect,
  lensValue,
}: CircuitProps) {
  const metric = (id: string) => (lens === 'flow' ? '' : lensValue(id))

  // Build layout bottom (input) → top (output).
  // We accumulate y offsets as we walk prelude → layers → head.
  // The output direction is upward, so y decreases as we go up.
  // We first compute heights for all groups, then assign y positions.

  // Helper: total height of all branches in a layer group (the tallest branch
  // determines the tee-to-junction span; here every branch in a group is stacked
  // separately between its own tee and junction, but both share the same tee and
  // junction on the mainline — so the group spans max branch height).
  // Actually each branch has its own tee/junction pair stacked on the mainline,
  // so the group height = sum of per-branch spans + gap between branches.
  const BRANCH_GAP = 34 // gap between consecutive tee points within a group

  // Compute cumulative y positions starting from PAD (top of SVG, = output).
  // We lay out top-to-bottom in SVG space, then the SVG reads bottom=input top=output.

  // Head blocks (top of SVG = output side)
  const headYs: { block: Block; y: number }[] = []
  let y = PAD
  for (const block of [...manifest.head].reverse()) {
    headYs.unshift({ block, y })
    y += MAIN_PITCH
  }

  // Layer groups — each group has brackets and interleaved branches.
  // For each group we need: bracketTop, bracketBot, and per-branch tee/junction ys.
  interface GroupLayout {
    group: LayerGroup
    bracketTop: number
    bracketBot: number
    branches: { branch: import('./modelArch').Branch; yTee: number; yJunc: number }[]
  }
  const groupLayouts: GroupLayout[] = []

  for (const group of [...manifest.layers].reverse()) {
    const bracketTopY = y
    // For each branch (in reverse order since we're building top-to-bottom in SVG):
    // Junction (merge point) comes before tee (split point) when going upward.
    const branchLayouts: GroupLayout['branches'] = []
    for (const branch of [...group.branches].reverse()) {
      const cardCount = 1 + branch.steps.length // preNorm + steps
      const yJunc = y
      const span = branchHeight(cardCount)
      const yTee = y + span
      branchLayouts.unshift({ branch, yTee, yJunc })
      y = yTee + BRANCH_GAP
    }
    // Remove the last BRANCH_GAP (already ends the group)
    y -= BRANCH_GAP
    const bracketBotY = y + MAIN_PITCH / 2
    y += MAIN_PITCH / 2

    groupLayouts.push({
      group,
      bracketTop: bracketTopY,
      bracketBot: bracketBotY,
      branches: branchLayouts,
    })
  }
  groupLayouts.reverse()

  // Prelude blocks (bottom of SVG = input side)
  const preludeYs: { block: Block; y: number }[] = []
  for (const block of [...manifest.prelude].reverse()) {
    preludeYs.unshift({ block, y })
    y += MAIN_PITCH
  }
  preludeYs.reverse()
  // y now = total height
  const height = y + PAD

  const yTop = PAD
  const yBot = height - PAD

  return (
    <svg
      viewBox={`0 0 ${VIEW_W} ${height}`}
      className="mx-auto w-full max-w-[560px]"
      role="group"
      aria-label={`${manifest.model} decoder as a residual-mainline diagram; input enters at the bottom, output exits at the top`}
    >
      <defs aria-hidden="true">
        {Object.entries(KIND_COLOR).map(([kind, color]) => (
          <linearGradient key={kind} id={`grad-${kind}`} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity={0.2} />
            <stop offset="100%" stopColor={color} stopOpacity={0.035} />
          </linearGradient>
        ))}
        <linearGradient id="rail-grad" gradientUnits="userSpaceOnUse" x1="0" y1={yBot} x2="0" y2={yTop}>
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#38bdf8" />
        </linearGradient>
      </defs>

      {/* Layer group brackets */}
      {groupLayouts.map(({ group, bracketTop, bracketBot }, gi) => (
        <g key={gi}>
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
            {group.label ?? 'decoder layer'} × {group.repeat}
          </text>
        </g>
      ))}

      {/* Residual mainline rail (bottom → top) */}
      <line
        className="mainline-rail"
        x1={MAINLINE_X}
        y1={yBot}
        x2={MAINLINE_X}
        y2={yTop}
        stroke="url(#rail-grad)"
      />
      {Array.from({ length: 6 }).map((_, i) => {
        const cy = yBot - ((yBot - yTop) * (i + 0.5)) / 6
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
      {groupLayouts.length > 0 && (
        <text
          x={MAINLINE_X - 24}
          y={(groupLayouts[0].branches[groupLayouts[0].branches.length - 1].yTee +
            groupLayouts[groupLayouts.length - 1].branches[0].yJunc) / 2}
          textAnchor="middle"
          className="module-shell-title"
          transform={`rotate(-90, ${MAINLINE_X - 24}, ${(groupLayouts[0].branches[groupLayouts[0].branches.length - 1].yTee +
            groupLayouts[groupLayouts.length - 1].branches[0].yJunc) / 2})`}
        >
          residual stream ↑
        </text>
      )}

      {/* Layer group branches and plus nodes */}
      {groupLayouts.map(({ group: _g, branches }, gi) =>
        branches.map(({ branch, yTee, yJunc }) => (
          <g key={`${gi}-${branch.name}`}>
            <BranchPanel
              name={branch.name}
              accent={branch.accent}
              preNorm={branch.preNorm}
              steps={branch.steps}
              yTee={yTee}
              yJunc={yJunc}
              selectedId={selectedId}
              onSelect={onSelect}
              metric={metric}
            />
            <PlusNode x={MAINLINE_X} y={yJunc} label={branch.name} />
          </g>
        )),
      )}

      {/* Head mainline nodes (top = output) */}
      {headYs.map(({ block, y: by }) => (
        <NodeCard
          key={block.id}
          block={block}
          cx={MAINLINE_X}
          cy={by}
          w={MAIN_W}
          h={MAIN_H}
          tag={false}
          selected={selectedId === block.id}
          onSelect={onSelect}
          metric={metric(block.id)}
        />
      ))}

      {/* Prelude mainline nodes (bottom = input) */}
      {preludeYs.map(({ block, y: by }) => (
        <NodeCard
          key={block.id}
          block={block}
          cx={MAINLINE_X}
          cy={by}
          w={MAIN_W}
          h={MAIN_H}
          tag={false}
          selected={selectedId === block.id}
          onSelect={onSelect}
          metric={metric(block.id)}
        />
      ))}

      <text x={MAINLINE_X} y={yBot + 28} textAnchor="middle" className="module-shell-title">
        input_ids
      </text>
      <text x={MAINLINE_X} y={yTop - 22} textAnchor="middle" className="module-shell-title">
        output logits
      </text>
    </svg>
  )
}

// --- One residual branch -----------------------------------------------------
function BranchPanel({
  name,
  accent,
  preNorm,
  steps,
  yTee,
  yJunc,
  selectedId,
  onSelect,
  metric,
}: {
  name: string
  accent: string
  preNorm: Block
  steps: Block[]
  yTee: number
  yJunc: number
  selectedId: string
  onSelect: (id: string) => void
  metric: (id: string) => string
}) {
  const cards = [preNorm, ...steps] // bottom → top
  const bottomCenter = yTee - PRENORM_GAP - CARD_H / 2
  const ys = cards.map((_, i) => bottomCenter - i * CARD_PITCH)
  const topY = ys[ys.length - 1]
  const shellTop = topY - CARD_H / 2 - 26
  const shellBot = ys[0] + CARD_H / 2 + 12
  const pillW = name.length * 6.2 + 18

  return (
    <g>
      {/* Tinted branch panel */}
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
      {/* Header pill straddling the top border */}
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
      <text
        x={SHELL_LEFT + 14 + pillW / 2}
        y={shellTop + 3.5}
        textAnchor="middle"
        className="branch-pill-text"
        fill={accent}
      >
        {name}
      </text>

      {/* Tee → up the card column → merge into the + node */}
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

      {cards.map((block, i) => (
        <NodeCard
          key={block.id}
          block={block}
          cx={CARD_X}
          cy={ys[i]}
          w={CARD_W}
          h={CARD_H}
          tag
          selected={selectedId === block.id}
          onSelect={onSelect}
          metric={metric(block.id)}
        />
      ))}
    </g>
  )
}

// --- A clickable block card --------------------------------------------------
function NodeCard({
  block,
  cx,
  cy,
  w,
  h,
  tag,
  selected,
  onSelect,
  metric,
}: {
  block: Block
  cx: number
  cy: number
  w: number
  h: number
  tag: boolean
  selected: boolean
  onSelect: (id: string) => void
  metric: string
}) {
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
      onClick={() => onSelect(block.id)}
      onKeyDown={(ev) => {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault()
          onSelect(block.id)
        }
      }}
    >
      {/* Dark base + kind-tinted gradient + crisp border */}
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

// --- Residual-add node (a crisp +, not a glyph) ------------------------------
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
