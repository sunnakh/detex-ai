'use client';

import type { Theme } from '@/types';

interface HeaderProps {
  theme: Theme;
  onToggleTheme: () => void;
  sidebarOpen: boolean;
  onToggleSidebar: () => void;
  onSignOut?: () => void;
}

export default function Header({ theme, onToggleTheme, sidebarOpen, onToggleSidebar, onSignOut }: HeaderProps) {
  return (
    <header className="main-header">
      <div className="header-left">
        {/* Sidebar toggle */}
        <button
          className="sidebar-toggle-btn"
          onClick={onToggleSidebar}
          id="sidebar-toggle"
          aria-label={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
          title={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
        >
          {sidebarOpen ? <PanelCloseIcon /> : <PanelOpenIcon />}
        </button>

        <div className="header-title">
          <span className="status-indicator" />
          detex<span style={{color:'var(--accent)'}}>.ai</span>
        </div>
      </div>

      <div className="header-actions">
        <div className="theme-toggle" role="group" aria-label="Toggle theme">
          <button
            id="theme-day"
            className={`theme-btn${theme === 'light' ? ' active' : ''}`}
            onClick={() => theme === 'dark' && onToggleTheme()}
            aria-pressed={theme === 'light'}
          >
            <SunIcon />
            Day
          </button>
          <button
            id="theme-night"
            className={`theme-btn${theme === 'dark' ? ' active' : ''}`}
            onClick={() => theme === 'light' && onToggleTheme()}
            aria-pressed={theme === 'dark'}
          >
            <MoonIcon />
            Night
          </button>
        </div>
        {onSignOut && (
          <button
            onClick={onSignOut}
            id="sign-out-btn"
            style={{
              marginLeft: '8px', padding: '6px 12px',
              borderRadius: 'var(--radius)', border: '1px solid var(--border)',
              background: 'transparent', color: 'var(--text-secondary)',
              fontSize: '12.5px', cursor: 'pointer', display: 'flex',
              alignItems: 'center', gap: '5px',
              transition: 'background var(--transition), color var(--transition)',
            }}
            title="Sign out"
          >
            <SignOutIcon /> Sign out
          </button>
        )}
      </div>
    </header>
  );
}

const PanelCloseIcon = () => (
  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <rect x="3" y="3" width="18" height="18" rx="2"/>
    <line x1="9" y1="3" x2="9" y2="21"/>
    <line x1="14" y1="9" x2="16" y2="12"/>
    <line x1="14" y1="15" x2="16" y2="12"/>
  </svg>
);

const PanelOpenIcon = () => (
  <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <rect x="3" y="3" width="18" height="18" rx="2"/>
    <line x1="9" y1="3" x2="9" y2="21"/>
    <line x1="12" y1="9" x2="14" y2="12"/>
    <line x1="12" y1="15" x2="14" y2="12"/>
  </svg>
);

const SunIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <circle cx="12" cy="12" r="5"/>
    <line x1="12" y1="1"  x2="12" y2="3"/>
    <line x1="12" y1="21" x2="12" y2="23"/>
    <line x1="4.22" y1="4.22"  x2="5.64" y2="5.64"/>
    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
    <line x1="1"  y1="12" x2="3"  y2="12"/>
    <line x1="21" y1="12" x2="23" y2="12"/>
    <line x1="4.22"  y1="19.78" x2="5.64"  y2="18.36"/>
    <line x1="18.36" y1="5.64"  x2="19.78" y2="4.22"/>
  </svg>
);
const MoonIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
  </svg>
);

const SignOutIcon = () => (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
    <polyline points="16 17 21 12 16 7"/>
    <line x1="21" y1="12" x2="9" y2="12"/>
  </svg>
);
