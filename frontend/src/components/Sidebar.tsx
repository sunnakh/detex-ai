'use client';

import Link from 'next/link';
import LogoMark from '@/components/LogoMark';
import type { Session } from '@/types';

interface SidebarProps {
  sessions: Session[];
  activeId: number;
  onSelect: (id: number) => void;
  onNew: () => void;
  open: boolean;
  onToggle: () => void;
}

export default function Sidebar({ sessions, activeId, onSelect, onNew, open }: SidebarProps) {
  return (
    <aside className={`sidebar${open ? '' : ' sidebar-collapsed'}`}>
      <div className="sidebar-header">
        <LogoMark size={30} />
        {open && <span className="logo-name">detex<span style={{color:'var(--accent)'}}>.ai</span></span>}
      </div>

      {open && (
        <>
          <button className="new-chat-btn" onClick={onNew} id="new-chat-btn">
            <PlusIcon />
            New Analysis
          </button>

          <div className="sidebar-label">Recent</div>

          <div className="history-list">
            {sessions.map(s => (
              <div
                key={s.id}
                className={`history-item${s.id === activeId ? ' active' : ''}`}
                onClick={() => onSelect(s.id)}
                id={`session-${s.id}`}
              >
                <span
                  className="history-dot"
                  style={{ background: getDotColor(s.lastResult?.label) }}
                />
                <span style={{ overflow: 'hidden', textOverflow: 'ellipsis' }}>{s.title}</span>
              </div>
            ))}
          </div>

          <div className="sidebar-footer">
            <Link href="/app/settings" className="sidebar-footer-btn">
              <GearIcon />
              Settings
            </Link>
            <button className="sidebar-footer-btn">
              <HelpIcon />
              Help &amp; FAQ
            </button>
          </div>
        </>
      )}
    </aside>
  );
}

function getDotColor(label?: string) {
  if (!label) return 'var(--text-muted)';
  return label === 'AI-generated' ? 'var(--ai-color)' : 'var(--human-color)';
}

const PlusIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
  </svg>
);
const GearIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="12" cy="12" r="3"/>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06-.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
  </svg>
);
const HelpIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="12" cy="12" r="10"/>
    <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>
    <line x1="12" y1="17" x2="12.01" y2="17"/>
  </svg>
);
