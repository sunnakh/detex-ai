'use client';

import Link from 'next/link';
import { useState, useEffect } from 'react';
import LogoMark from '@/components/LogoMark';

export default function LandingPage() {
  const [theme, setTheme] = useState<'dark'|'light'>('light');

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  return (
    <div className="lp-root">
      {/* ── Nav ── */}
      <nav className="lp-nav">
        <div className="lp-nav-inner">
          <div className="lp-logo">
            <LogoMark size={30} />
            <span className="lp-logo-name">detex<span style={{color:'var(--accent)'}}>.ai</span></span>
          </div>
          <div className="lp-nav-links">
            <a href="#features" className="lp-nav-link">Features</a>
            <a href="#how" className="lp-nav-link">How it works</a>
            <Link href="/sign-in" className="lp-nav-link">Sign In</Link>
            <div className="theme-toggle" style={{marginLeft: '4px'}}>
              <button
                className={`theme-btn${theme === 'light' ? ' active' : ''}`}
                onClick={() => setTheme('light')}
              >☀️</button>
              <button
                className={`theme-btn${theme === 'dark' ? ' active' : ''}`}
                onClick={() => setTheme('dark')}
              >🌙</button>
            </div>
            <Link href="/sign-up" className="lp-cta-sm">Get Started</Link>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="lp-hero">

        <h1 className="lp-hero-h1">
          Know if it&apos;s written<br />
          <span className="lp-hero-gradient">by AI or a human.</span>
        </h1>
        <p className="lp-hero-sub">
          Paste any text and get an instant, confidence-scored analysis.
          Detect ChatGPT, Claude, Gemini, and other AI writing in seconds.
        </p>
        <div className="lp-hero-btns">
          <Link href="/sign-up" className="lp-btn-primary">
            Try it free →
          </Link>
          <Link href="/sign-in" className="lp-btn-ghost">
            Sign in
          </Link>
        </div>

        {/* Animated Demo Card */}
        <div className="lp-demo-card">
          {/* Window bar */}
          <div className="lp-demo-header">
            <div className="lp-demo-dots">
              <span style={{ background: '#ef4444' }} />
              <span style={{ background: '#f59e0b' }} />
              <span style={{ background: '#10b981' }} />
            </div>
            <span className="lp-demo-title">detex.ai — live analysis</span>
            <span className="lp-demo-live-dot" />
          </div>

          {/* Scan animation strip */}
          <div className="lp-demo-scan-wrap">
            <div className="lp-demo-scan-text">
              "The empirical evidence suggests a paradigm shift in contemporary discourse, 
              particularly in domains where nuanced perspectives facilitate holistic understanding 
              of multifaceted socioeconomic challenges..."
            </div>
            <div className="lp-demo-scan-line" />
          </div>

          {/* Result body */}
          <div className="lp-demo-body">
            {/* Verdict badge */}
            <div className="lp-demo-verdict-row">
              <div className="lp-demo-verdict ai">
                <span className="lp-demo-verdict-pulse" />
                ✦ AI-generated
              </div>
              <span className="lp-demo-confidence">78% confidence</span>
            </div>

            {/* Dual score bars */}
            <div className="lp-demo-bars">
              <div className="lp-demo-bar-item">
                <div className="lp-demo-bar-labels">
                  <span style={{ color: '#f87171' }}>✦ AI</span>
                  <span style={{ color: '#f87171', fontWeight: 600 }}>78%</span>
                </div>
                <div className="lp-demo-bar-track">
                  <div className="lp-demo-bar-fill ai-fill" style={{ '--target-w': '78%' } as React.CSSProperties} />
                </div>
              </div>
              <div className="lp-demo-bar-item">
                <div className="lp-demo-bar-labels">
                  <span style={{ color: '#34d399' }}>◎ Human</span>
                  <span style={{ color: '#34d399', fontWeight: 600 }}>22%</span>
                </div>
                <div className="lp-demo-bar-track">
                  <div className="lp-demo-bar-fill human-fill" style={{ '--target-w': '22%' } as React.CSSProperties} />
                </div>
              </div>
            </div>

            {/* Stats row */}
            <div className="lp-demo-stats">
              <div><span>Words</span><strong>142</strong></div>
              <div><span>Characters</span><strong>891</strong></div>
              <div><span>Time</span><strong>0.3ms</strong></div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section className="lp-section" id="features">
        <div className="lp-section-inner">
          <div className="lp-section-label">Features</div>
          <h2 className="lp-section-h2">Everything you need to<br />verify text authenticity</h2>
          <div className="lp-features-grid">
            {FEATURES.map(f => (
              <div className="lp-feature-card" key={f.title}>
                <div className="lp-feature-icon">{f.icon}</div>
                <h3 className="lp-feature-title">{f.title}</h3>
                <p className="lp-feature-desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section className="lp-section lp-section-alt" id="how">
        <div className="lp-section-inner">
          <div className="lp-section-label">How it works</div>
          <h2 className="lp-section-h2">Three steps to know the truth</h2>
          <div className="lp-steps">
            {STEPS.map((s, i) => (
              <div className="lp-step" key={s.title}>
                <div className="lp-step-num">{i + 1}</div>
                <h3 className="lp-step-title">{s.title}</h3>
                <p className="lp-step-desc">{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="lp-cta-section">
        <div className="lp-cta-inner">
          <h2 className="lp-cta-h2">Start detecting for free</h2>
          <p className="lp-cta-sub">No credit card required. Works instantly.</p>
          <div className="lp-hero-btns">
            <Link href="/app" className="lp-btn-primary">Try it now →</Link>
            <Link href="/sign-up" className="lp-btn-ghost-light">Create account</Link>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="lp-footer">
        <div className="lp-footer-inner">
          <div className="lp-footer-brand">
            <div className="lp-logo">
              <LogoMark size={28} />
              <span className="lp-logo-name" style={{color:'#8b949e'}}>detex<span style={{color:'var(--accent)'}}>
.ai</span></span>
            </div>
            <p className="lp-footer-tagline">
              AI Text Detection &nbsp;&middot;&nbsp; Built by{' '}
              <a href="https://sunnakh.com" target="_blank" rel="noopener noreferrer" className="lp-footer-brand-link">neurobrain-ai</a>
            </p>
          </div>
          <div className="lp-footer-right">
            <div className="lp-footer-links">
              <Link href="/privacy">Privacy</Link>
              <Link href="/terms">Terms</Link>
              <a href="https://sunnakh.com/contact/" target="_blank" rel="noopener noreferrer">Contact</a>
            </div>
            <p className="lp-footer-copy">© 2026 detex.ai. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

const FEATURES = [
  { icon: '⚡', title: 'Instant results', desc: 'Get confidence-scored detections in under a millisecond. No waiting, no queues.' },
  { icon: '🎯', title: 'High accuracy', desc: 'Multi-signal linguistic analysis combining 9 independent detection features.' },
  { icon: '📊', title: 'Detailed scores', desc: 'AI vs human percentages, word count, analysis time, and interpretation — all in one view.' },
  { icon: '🌙', title: 'Dark & light mode', desc: 'Beautiful catops-style interface that works in any lighting. Toggle with one click.' },
  { icon: '💬', title: 'Session history', desc: 'Every analysis is saved in your session history with color-coded results.' },
  { icon: '🔒', title: 'Privacy first', desc: 'Your text is never stored or logged. All analysis happens in real-time.' },
];

const STEPS = [
  { title: 'Paste your text', desc: 'Copy any text — an email, essay, article, or social post — and paste it into the detector.' },
  { title: 'Instant analysis', desc: 'Our multi-signal engine analyzes sentence variance, vocabulary, AI fingerprints, and more.' },
  { title: 'Read the verdict', desc: 'Get a clear AI vs Human score with confidence percentage and detailed interpretation.' },
];
