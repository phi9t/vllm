export interface CoreNode {
  id: string
  label: string
  group: 'core'
  file: string
  line: number
  symbol: string | null
}

export interface Edge {
  from: string
  to: string
}

export interface SectionNode {
  id: string
  section: string
  title: string
  group: 'section'
  file: string | null
  line: number | null
  symbol: string | null
  hacks: string[]
}

export interface Hack {
  n: string
  script: string
  section: string
  desc: string
}

export interface ComponentManifest {
  generated_at: string
  guide: string
  nodes: CoreNode[]
  edges: Edge[]
  sections: SectionNode[]
  hacks: Hack[]
}

/** Normalized detail shown in the drawer (built from a core node or section). */
export interface DrawerDetail {
  title: string
  file: string | null
  line: number | null
  symbol: string | null
  section: string | null
  sectionTitle: string | null
  hacks: Hack[]
  isModelRunner: boolean
}
