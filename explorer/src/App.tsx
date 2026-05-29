import { useState } from 'react'
import { ArrowLeft, Database, Network, Boxes } from 'lucide-react'
import { REPO_HOME, logoMarkUrl } from './lib/assets'
import DataExplorer from './data/DataExplorer'
import ComponentExplorer from './components-deepdive/ComponentExplorer'
import ArchitectureExplorer from './architecture/ArchitectureExplorer'

export type ExplorerFamily = 'data' | 'components' | 'architecture'

const FAMILIES: { id: ExplorerFamily; label: string; icon: typeof Database; subtitle: string }[] = [
  {
    id: 'data',
    label: 'Data Exploration',
    icon: Database,
    subtitle: 'fineweb-edu format & Qwen3 tokenization',
  },
  {
    id: 'components',
    label: 'Component Deep Dive',
    icon: Network,
    subtitle: 'V1 engine internals — a HACKERS_GUIDE + hacks companion',
  },
  {
    id: 'architecture',
    label: 'Model Architecture',
    icon: Boxes,
    subtitle: 'Inference forward pass — Qwen3 & DeepSeek (dense · MoE · MLA)',
  },
]

export default function App() {
  const [family, setFamily] = useState<ExplorerFamily>('data')
  const active = FAMILIES.find((f) => f.id === family)!

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
            {FAMILIES.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                className={`family-switch-btn ${family === id ? 'active' : ''}`}
                aria-pressed={family === id}
                onClick={() => setFamily(id)}
              >
                <Icon size={14} aria-hidden="true" />
                {label}
              </button>
            ))}
          </nav>
        </header>

        <main id="main-content">
          {family === 'data' && <DataExplorer />}
          {family === 'components' && (
            <ComponentExplorer onOpenArchitecture={() => setFamily('architecture')} />
          )}
          {family === 'architecture' && <ArchitectureExplorer />}
        </main>
      </div>
    </div>
  )
}
