'use client';

import Link from 'next/link';
import { useState, useEffect, useRef } from 'react';
import LogoMark from '@/components/LogoMark';

const API = 'http://localhost:8000';

type ScanResult = { ai_score: number; human_score: number; label: string; confidence: number; analysis_time?: number };

export default function LandingPage() {
  const [theme, setTheme] = useState<'dark'|'light'>('light');
  const [activeTab, setActiveTab] = useState<'text' | 'file' | 'humanizer'>('text');
  const [text, setText] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [scanning, setScanning] = useState(false);
  const [result, setResult] = useState<ScanResult | null>(null);
  const [humanizedText, setHumanizedText] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  async function handleScan() {
    setError(null);
    setResult(null);
    setHumanizedText(null);
    setScanning(true);
    try {
      if (activeTab === 'file') {
        if (!selectedFile) { setError('Please select a file first.'); setScanning(false); return; }
        const fd = new FormData();
        fd.append('file', selectedFile);
        const res = await fetch(`${API}/detect-file`, { method: 'POST', body: fd });
        if (!res.ok) throw new Error((await res.json()).detail || 'Detection failed');
        setResult(await res.json());
      } else if (activeTab === 'text') {
        if (!text.trim()) { setError('Please paste some text first.'); setScanning(false); return; }
        const res = await fetch(`${API}/detect`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || 'Detection failed');
        setResult(await res.json());
      } else if (activeTab === 'humanizer') {
        if (!text.trim()) { setError('Please paste some text first.'); setScanning(false); return; }
        const res = await fetch(`${API}/humanize`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || 'Humanization failed');
        const data = await res.json();
        setHumanizedText(data.humanized_text);
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Something went wrong.');
    } finally {
      setScanning(false);
    }
  }

  return (
    <div className="lp-root">
      {/* ── Nav ── */}
      <nav className="lp-nav">
        <div className="lp-nav-inner">
          <Link href="/" className="lp-logo">
            <LogoMark size={28} />
            <span className="lp-logo-name">detex.ai</span>
          </Link>
          <div className="lp-nav-links">
            <a href="#features" className="lp-nav-link">Features</a>
            <a href="#how" className="lp-nav-link">How it works</a>
            <Link href="/sign-in" className="lp-nav-link">Log in</Link>
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
            <Link href="/sign-up" className="lp-cta-sm">Sign up</Link>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section className="lp-hero">
        <div style={{
          display: 'inline-block',
          padding: '6px 16px',
          borderRadius: '999px',
          background: 'var(--accent-light)',
          color: 'var(--accent)',
          fontSize: '13px',
          fontWeight: 600,
          marginBottom: '24px',
          border: '1px solid var(--accent)',
          opacity: 0.9
        }}>
          ✨ Try out our new AI text detector in beta
        </div>
        <h1 className="lp-hero-h1">
          The Gold Standard in<br />
          <span className="lp-hero-gradient">AI Detection</span>
        </h1>
        <p className="lp-hero-sub">
          Preserve what&apos;s human. Paste your text below to instantly detect AI-generated 
          content from ChatGPT, Claude, Gemini, and more.
        </p>

        <div id="scanner-section" className="lp-scanner-layout">
          <div className="lp-scanner-main" style={{ height: '100%' }}>
            <div className="lp-scanner-card" style={{ marginBottom: 0, height: '100%', display: 'flex', flexDirection: 'column' }}>
          <div className="lp-scanner-tabs">
            <button 
              className={`lp-scanner-tab ${activeTab === 'text' ? 'active' : ''}`}
              onClick={() => setActiveTab('text')}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
              Text
            </button>
            <div className="lp-scanner-tab-divider" />
            <button 
              className={`lp-scanner-tab ${activeTab === 'file' ? 'active' : ''}`}
              onClick={() => setActiveTab('file')}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path></svg>
              File Upload
            </button>
            <div className="lp-scanner-tab-divider" />
            <button 
              className={`lp-scanner-tab ${activeTab === 'humanizer' ? 'active' : ''}`}
              onClick={() => setActiveTab('humanizer')}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
              Humanizer
            </button>
          </div>
          <div className="lp-scanner-body" style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            {activeTab === 'file' ? (
              <label htmlFor="lp-file-input" style={{
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                height: '100%', minHeight: '180px', cursor: 'pointer', borderRadius: '12px',
                border: selectedFile ? '2px solid var(--accent)' : '2px dashed var(--border)',
                background: 'var(--bg-input)',
                transition: 'border-color 0.2s, background 0.2s',
                gap: '12px', padding: '32px', textAlign: 'center',
              }}
              onDragOver={(e) => { e.preventDefault(); (e.currentTarget as HTMLElement).style.borderColor = 'var(--accent)'; }}
              onDragLeave={(e) => { (e.currentTarget as HTMLElement).style.borderColor = selectedFile ? 'var(--accent)' : 'var(--border)'; }}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files[0];
                if (f) { setSelectedFile(f); setResult(null); }
                (e.currentTarget as HTMLElement).style.borderColor = 'var(--accent)';
              }}
              >
                <input
                  id="lp-file-input" ref={fileInputRef} type="file" accept=".pdf,.doc,.docx,.txt"
                  style={{ display: 'none' }}
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) { setSelectedFile(f); setResult(null); } }}
                />
                {selectedFile ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                    <p style={{ margin: 0, fontWeight: 600, color: 'var(--accent)', fontSize: '14px' }}>{selectedFile.name}</p>
                    <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '12px' }}>{(selectedFile.size / 1024).toFixed(1)} KB · <span style={{ color: 'var(--accent)', cursor: 'pointer', textDecoration: 'underline' }} onClick={(e) => { e.preventDefault(); setSelectedFile(null); if(fileInputRef.current) fileInputRef.current.value=''; }}>Remove</span></p>
                  </div>
                ) : (
                  <>
                    <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                    <div>
                      <p style={{ margin: 0, fontWeight: 600, color: 'var(--text-primary)', fontSize: '15px' }}>Drop your file here, or <span style={{ color: 'var(--accent)' }}>browse</span></p>
                      <p style={{ margin: '4px 0 0', color: 'var(--text-muted)', fontSize: '13px' }}>Supports PDF, DOC, DOCX, TXT — up to 10MB</p>
                    </div>
                  </>
                )}
              </label>
            ) : (
              <textarea
                className="lp-scanner-textarea"
                placeholder={activeTab === 'humanizer' ? 'Paste AI-generated text here to humanize it...' : 'Paste your text here...'}
                spellCheck={false}
                value={text}
                onChange={(e) => { setText(e.target.value); setResult(null); }}
                style={{ flex: 1, minHeight: '260px' }}
              />
            )}
          </div>
          <div className="lp-scanner-footer">
            <div className="lp-scanner-stats">
              {activeTab === 'file' ? (
                <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>PDF · DOC · DOCX · TXT</span>
              ) : (
                <>
                  <span>{text.trim().split(/\s+/).filter(w => w.length > 0).length} words</span>
                  <span>{text.length} characters</span>
                </>
              )}
            </div>
            {activeTab === 'humanizer' ? (
              <button 
                className="lp-btn-primary lp-btn-scan" 
                onClick={handleScan}
                disabled={scanning}
                style={{ opacity: scanning ? 0.7 : 1, cursor: scanning ? 'not-allowed' : 'pointer' }}
              >
                {scanning ? 'Humanizing…' : 'Humanize Text'}
              </button>
            ) : (
              <button
                className="lp-btn-primary lp-btn-scan"
                onClick={handleScan}
                disabled={scanning}
                style={{ opacity: scanning ? 0.7 : 1, cursor: scanning ? 'not-allowed' : 'pointer' }}
              >
                {scanning ? 'Scanning…' : activeTab === 'file' ? 'Upload & Scan' : 'Scan Text'}
              </button>
            )}
          </div>
        </div>
          </div>
        
        <div className="lp-scanner-side" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          {/* ── Scan Result ── */}
          {error && (
            <div style={{
              padding: '12px 16px', borderRadius: '10px',
              background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.3)',
              color: 'var(--ai-color)', fontSize: '14px', fontWeight: 500,
            }}>{error}</div>
          )}
          {!result && !error && !humanizedText && (
            <div style={{
              padding: '32px 24px', borderRadius: '16px',
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
              boxShadow: 'var(--shadow-md)', textAlign: 'center', flex: 1
            }}>
              <div style={{
                width: '64px', height: '64px', borderRadius: '50%', background: 'var(--bg-input)',
                display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '20px'
              }}>
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
              </div>
              <h3 style={{ margin: '0 0 8px', fontSize: '20px', fontWeight: 600, color: 'var(--text-primary)' }}>Awaiting Scan</h3>
              <p style={{ margin: 0, fontSize: '15px', color: 'var(--text-muted)', lineHeight: 1.5, maxWidth: '240px' }}>
                Enter text or upload a file and click scan to see the detailed AI probability breakdown.
              </p>
            </div>
          )}
          {result && (() => {
          const isAI = result.label === 'AI-generated';
          const probPercentage = (result.confidence * 100).toFixed(1);
          const themeColor = isAI ? '#ef4444' : '#10b981';
          const entityStr = isAI ? 'AI' : 'Human';
          
          const radius = 70;
          const strokeWidth = 10;
          const normalizedRadius = radius - strokeWidth * 0.5;
          const circumference = normalizedRadius * 2 * Math.PI;
          const strokeDashoffset = circumference - (result.confidence) * circumference;

          return (
            <div style={{
              padding: '32px 24px', borderRadius: '16px',
              background: 'var(--bg-card)', border: '1px solid var(--border)',
              boxShadow: 'var(--shadow-md)', textAlign: 'center',
              animation: 'rc2-in 0.4s cubic-bezier(0.34,1.2,0.64,1)',
              position: 'relative',
              flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center'
            }}>
              {/* Download Icon (Top Right) */}
              <button 
                onClick={() => alert('Download coming soon!')}
                style={{
                  position: 'absolute', top: '16px', right: '16px',
                  background: 'var(--bg-input)', border: '1px solid var(--border)',
                  borderRadius: '8px', padding: '8px', cursor: 'pointer',
                  color: 'var(--text-secondary)'
                }}
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
              </button>

              <div style={{ position: 'relative', width: '160px', height: '160px', margin: '0 auto 40px' }}>
                <svg height="160" width="160" style={{ transform: 'rotate(-90deg)' }}>
                  <circle
                    stroke="var(--bg-input)"
                    fill="transparent"
                    strokeWidth={strokeWidth}
                    r={normalizedRadius}
                    cx="80"
                    cy="80"
                  />
                  <circle
                    stroke={themeColor}
                    fill="transparent"
                    strokeWidth={strokeWidth}
                    strokeDasharray={circumference + ' ' + circumference}
                    style={{ strokeDashoffset, transition: 'stroke-dashoffset 1.5s ease-out' }}
                    strokeLinecap="round"
                    r={normalizedRadius}
                    cx="80"
                    cy="80"
                  />
                </svg>
                <div style={{
                  position: 'absolute', top: 0, left: 0, width: '100%', height: '100%',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '36px', fontWeight: 700, color: themeColor,
                  letterSpacing: '-1px'
                }}>
                  {probPercentage}%
                </div>
              </div>

              <h2 style={{ fontSize: '26px', fontWeight: 700, color: 'var(--text-primary)', margin: '0 0 12px', letterSpacing: '-0.5px' }}>
                This text was likely written by <span style={{ color: themeColor }}>{entityStr}</span>
              </h2>
              <p style={{ fontSize: '18px', color: 'var(--text-secondary)', margin: '0 0 32px' }}>
                There is a {probPercentage}% probability this text was entirely written by <span style={{ color: themeColor }}>{entityStr}</span>
              </p>

              {isAI && (
                <button 
                  onClick={() => setActiveTab('humanizer')}
                  style={{
                    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                    padding: '12px 32px', borderRadius: '10px',
                    background: 'var(--bg-input)', border: '1px solid var(--border)',
                    color: 'var(--text-primary)', fontSize: '16px', fontWeight: 600,
                    cursor: 'pointer', transition: 'background 0.2s, transform 0.2s',
                  }}
                  onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-hover)'}
                  onMouseOut={(e) => e.currentTarget.style.background = 'var(--bg-input)'}
                  onMouseDown={(e) => e.currentTarget.style.transform = 'scale(0.98)'}
                  onMouseUp={(e) => e.currentTarget.style.transform = 'scale(1)'}
                >
                  Humanize AI Text <span style={{ marginLeft: '8px', color: 'var(--text-muted)' }}>→</span>
                </button>
              )}
            </div>
          );
        })()}

        {/* ── Humanizer Result ── */}
        {humanizedText && (
          <div style={{
            padding: '24px', borderRadius: '16px',
            background: 'var(--bg-card)', border: '1px solid var(--border)',
            boxShadow: 'var(--shadow-md)', flex: 1, display: 'flex', flexDirection: 'column'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--human-color)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path></svg>
                <h3 style={{ margin: 0, fontSize: '16px', fontWeight: 600, color: 'var(--text-primary)' }}>Humanized Text</h3>
              </div>
              <button
                onClick={() => { navigator.clipboard.writeText(humanizedText); alert('Copied to clipboard!'); }}
                style={{
                  background: 'var(--bg-input)', border: '1px solid var(--border)', borderRadius: '6px',
                  padding: '6px 12px', fontSize: '13px', fontWeight: 500, color: 'var(--text-secondary)',
                  cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px'
                }}
              >
                Copy Text
              </button>
            </div>
            <div style={{
              background: 'var(--bg-input)', padding: '16px', borderRadius: '8px',
              color: 'var(--text-primary)', fontSize: '15px', lineHeight: 1.6,
              whiteSpace: 'pre-wrap'
            }}>
              {humanizedText}
            </div>
          </div>
        )}
          </div>
        </div>

        <div className="lp-hero-agreement" style={{ marginTop: '24px' }}>
          <label>
            <input type="checkbox" defaultChecked />
            <span>I agree to the <Link href="/terms">Terms of Service</Link> and <Link href="/privacy">Privacy Policy</Link>.</span>
          </label>
        </div>
      </section>

      {/* ── Logo Strip ── */}
      <section className="lp-logos">
        <p className="lp-logos-text">Trusted by professionals and organizations worldwide</p>
        <div style={{
          display: 'flex', 
          flexWrap: 'nowrap', 
          justifyContent: 'center', 
          alignItems: 'center', 
          gap: '4rem', 
          margin: '0 auto',
          width: '100%'
        }}>
          
          {/* Uzinfocom */}
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/logos/uzinfocom-dark.svg" alt="Uzinfocom" width={160} className="transition-transform hover:scale-105" style={{ height: 'auto', display: 'block' }} />
          </div>

          {/* Zucco.tech */}
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/logos/zucco.svg" alt="Zucco.tech" width={180} className="transition-transform hover:scale-105" style={{ height: 'auto', display: 'block' }} />
          </div>

          {/* zehnmind.ai */}
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src="/logos/zehnmind.svg" alt="zehnmind.ai" width={160} className="transition-transform hover:scale-105" style={{ height: 'auto', display: 'block' }} />
          </div>

        </div>
      </section>

      {/* ── Instructions / How it Works ── */}
      <section className="lp-section" id="how-it-works">
        <div className="lp-section-inner">
          <div className="lp-features-grid lp-features-grid-clean" style={{marginBottom: '5rem'}}>
            <div className="lp-feature-text">
              <h2 className="lp-section-h2">How to Use<br />Detex.ai</h2>
              <p className="lp-feature-desc-large">
                Detecting AI content and bypassing detectors is incredibly simple. 
                Follow these three steps to analyze and humanize your text.
              </p>
              <ul className="lp-feature-list">
                <li><span className="lp-check">1</span> <b>Paste or Upload:</b> Enter any text or drop a document into the scanner.</li>
                <li><span className="lp-check">2</span> <b>Scan for AI:</b> Click scan to see the exact AI vs Human probability breakdown.</li>
                <li><span className="lp-check">3</span> <b>Humanize:</b> If it's flagged as AI, use the Humanizer tab to completely rewrite the text to bypass detectors.</li>
              </ul>
            </div>
            <div className="lp-feature-visual">
              {/* Simplified visual */}
              <div className="lp-visual-card">
                <div className="lp-visual-header">Detection Results</div>
                <div className="lp-visual-score">
                  <div className="lp-score-circle">
                    <span className="lp-score-num">89%</span>
                    <span className="lp-score-label">AI Probability</span>
                  </div>
                </div>
                <div className="lp-visual-bars">
                  <div className="lp-vbar"><div className="lp-vbar-fill ai" style={{width: '89%'}}></div></div>
                  <div className="lp-vbar"><div className="lp-vbar-fill human" style={{width: '11%'}}></div></div>
                </div>
              </div>
            </div>
          </div>

          <div className="lp-features-grid lp-features-grid-clean" style={{flexDirection: 'row-reverse'}}>
            <div className="lp-feature-text">
              <h2 className="lp-section-h2">Bypass Detectors<br />with the Humanizer.</h2>
              <p className="lp-feature-desc-large">
                Need to guarantee your work passes AI filters? Our integrated Humanizer intelligently rewrites text to introduce human-like variance and flow, effortlessly bypassing standard detection models.
              </p>
              <ul className="lp-feature-list">
                <li><span className="lp-check">✓</span> Retains your original meaning and tone</li>
                <li><span className="lp-check">✓</span> Introduces natural burstiness and perplexity</li>
                <li><span className="lp-check">✓</span> Produces 100% human-scored output</li>
              </ul>
            </div>
            <div className="lp-feature-visual">
              <div className="lp-visual-card">
                <div className="lp-visual-header">Humanizer Output</div>
                <div className="lp-visual-score">
                  <div className="lp-score-circle" style={{borderColor: 'var(--accent)', background: 'transparent'}}>
                    <span className="lp-score-num" style={{color: 'var(--accent)'}}>100%</span>
                    <span className="lp-score-label">Human Score</span>
                  </div>
                </div>
                <div className="lp-visual-bars">
                  <div className="lp-vbar"><div className="lp-vbar-fill" style={{width: '100%', background: 'var(--accent)'}}></div></div>
                  <div className="lp-vbar"><div className="lp-vbar-fill ai" style={{width: '0%'}}></div></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── CTA Bottom ── */}
      <section className="lp-cta-section">
        <div className="lp-cta-inner">
          <h2>Ready to Verify Your Text?</h2>
          <p>Join thousands of professionals using Detex.ai to ensure authenticity.</p>
          <a href="#scanner-section" className="lp-btn-cta" onClick={(e) => {
            e.preventDefault();
            document.getElementById('scanner-section')?.scrollIntoView({ behavior: 'smooth' });
          }}>
            Start Scanning Free
          </a>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="lp-footer">
        <div className="lp-footer-inner">
          <div className="lp-footer-brand">
            <Link href="/" className="lp-logo" style={{textDecoration: 'none'}}>
              <LogoMark size={24} />
              <span className="lp-logo-name" style={{color:'var(--text-secondary)'}}>detex.ai</span>
            </Link>
            <p className="lp-footer-tagline">
              Preserving human ingenuity.
            </p>
          </div>
          <div className="lp-footer-right">
            <div className="lp-nav-links">
              <Link href="/privacy">Privacy Policy</Link>
              <Link href="/terms">Terms of Service</Link>
              <a href="mailto:contact@detex.ai">Contact Us</a>
            </div>
            <p className="lp-footer-copy">© 2026 detex.ai. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}

