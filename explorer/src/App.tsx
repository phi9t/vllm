import { useState } from 'react'
import { ArrowLeft, Database, Network, Boxes, BookOpen } from 'lucide-react'
import { REPO_HOME, logoMarkUrl } from './lib/assets'
import type { ExplorerMode } from './explorer-kit/mode'
import DataExplorer from './data/DataExplorer'
import ComponentExplorer from './components-deepdive/ComponentExplorer'
import ArchitectureExplorer from './architecture/ArchitectureExplorer'
import GuideExplorer from './guide/GuideExplorer'

// Typed mode registry — adding a mode is one entry here, no new conditional.
const MODES: ExplorerMode[] = [
  {
    id: 'data',
    label: 'Data Exploration',
    icon: Database,
    subtitle: 'fineweb-edu format & Qwen3 tokenization',
    View: DataExplorer,
  },
  {
    id: 'components',
    label: 'Component Deep Dive',
    icon: Network,
    subtitle: 'V1 engine internals — a HACKERS_GUIDE + hacks companion',
    View: ComponentExplorer,
  },
  {
    id: 'architecture',
    label: 'Model Architecture',
    icon: Boxes,
    subtitle: 'Inference forward pass — Qwen3 & DeepSeek (dense · MoE · MLA)',
    View: ArchitectureExplorer,
  },
  {
    id: 'guide',
    label: "Hacker's Guide",
    icon: BookOpen,
    subtitle: 'The vLLM V1 engine, code-first — rendered from HACKERS_GUIDE.md (v0.22.0)',
    View: GuideExplorer,
  },
]

export default function App() {
  const [activeId, setActiveId] = useState<string>(MODES[0].id)
  const active = MODES.find((m) => m.id === activeId) ?? MODES[0]
  const ActiveView = active.View

  return (
    <div className="relative min-h-screen">
      <div className="observatory-bg" aria-hidden="true" />

      <div className="explorer-container">
        <header className="explorer-header">
          <div>
            <a href="#main-content" className="skip-link">
              Skip to main content
            </a>
            <a href={REPO_HOME} className="back-home-link" target="_blank" rel="noopener noreferrer">
              <ArrowLeft size={14} aria-hidden="true" />
              <span>phi9t/vllm research fork</span>
            </a>
            <div className="header-title-row">
              <img src={logoMarkUrl()} alt="" className="header-logo" width={32} height={32} />
              <h1>vLLM Explorer</h1>
            </div>
            <p>{active.subtitle}</p>
          </div>

          <nav className="family-switch" aria-label="Explorer section">
            {MODES.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                className={`family-switch-btn ${activeId === id ? 'active' : ''}`}
                aria-pressed={activeId === id}
                onClick={() => setActiveId(id)}
              >
                <Icon size={14} aria-hidden="true" />
                {label}
              </button>
            ))}
          </nav>
        </header>

        <main id="main-content">
          <ActiveView navigate={setActiveId} />
        </main>
      </div>
    </div>
  )
}
