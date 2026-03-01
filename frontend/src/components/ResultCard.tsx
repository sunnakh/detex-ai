'use client';

import { useState, useEffect } from 'react';
import type { DetectResult } from '@/types';

interface ResultCardProps { result: DetectResult; }

export default function ResultCard({ result }: ResultCardProps) {
  const isAI = result.label === 'AI-generated';
  const [animated, setAnimated] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 80);
    return () => clearTimeout(t);
  }, []);

  const aiPct   = Math.round(result.ai_score   * 100);
  const humPct  = Math.round(result.human_score * 100);
  const confPct = Math.round(result.confidence  * 100);

  // Circular ring math
  const R = 42, CIRC = 2 * Math.PI * R;
  const dash = animated ? CIRC * (1 - confPct / 100) : CIRC;

  return (
    <div className="rc2-wrap">
      {/* ── Top verdict banner ── */}
      <div className={`rc2-banner ${isAI ? 'rc2-ai' : 'rc2-human'}`}>
        <span className="rc2-banner-icon">{isAI ? '🤖' : '✍️'}</span>
        <div>
          <div className="rc2-verdict">{result.label}</div>
          <div className="rc2-subtext">{interp(result)}</div>
        </div>
      </div>

      {/* ── Body ── */}
      <div className="rc2-body">
        {/* Confidence ring */}
        <div className="rc2-ring-col">
          <svg width="110" height="110" viewBox="0 0 100 100">
            {/* track */}
            <circle cx="50" cy="50" r={R} fill="none" stroke="var(--border)" strokeWidth="9"/>
            {/* fill */}
            <circle
              cx="50" cy="50" r={R} fill="none"
              stroke={isAI ? 'var(--ai-color)' : 'var(--human-color)'}
              strokeWidth="9"
              strokeLinecap="round"
              strokeDasharray={CIRC}
              strokeDashoffset={dash}
              transform="rotate(-90 50 50)"
              style={{ transition: animated ? 'stroke-dashoffset 1s cubic-bezier(0.34,1.2,0.64,1)' : 'none' }}
            />
            <text x="50" y="46" textAnchor="middle" fill="currentColor" fontSize="18" fontWeight="700"
              style={{ fill: 'var(--text-primary)' }}>{confPct}%</text>
            <text x="50" y="60" textAnchor="middle" fontSize="9"
              style={{ fill: 'var(--text-muted)' }}>confidence</text>
          </svg>
        </div>

        {/* Split bar + labels */}
        <div className="rc2-scores">
          <div className="rc2-split-labels">
            <span className="rc2-ai-label">🤖 AI — {aiPct}%</span>
            <span className="rc2-human-label">✍️ Human — {humPct}%</span>
          </div>
          <div className="rc2-split-track">
            <div
              className="rc2-split-ai"
              style={{
                width: animated ? `${aiPct}%` : '0%',
                transition: animated ? 'width 0.9s cubic-bezier(0.34,1.2,0.64,1)' : 'none'
              }}
            />
            <div
              className="rc2-split-human"
              style={{
                width: animated ? `${humPct}%` : '0%',
                transition: animated ? 'width 0.9s 0.1s cubic-bezier(0.34,1.2,0.64,1)' : 'none',
                marginLeft: 'auto'
              }}
            />
          </div>
          {/* Stat pills row */}
          <div className="rc2-pills">
            <span className="rc2-pill">📝 {result.word_count.toLocaleString()} words</span>
            <span className="rc2-pill">🔤 {result.char_count.toLocaleString()} chars</span>
            <span className="rc2-pill">⚡ {result.analysis_time_ms} ms</span>
          </div>
        </div>
      </div>
    </div>
  );
}

function interp({ label, confidence }: DetectResult): string {
  const p = Math.round(confidence * 100);
  if (label === 'AI-generated') {
    if (p >= 90) return 'Very likely AI-generated — strong LLM patterns detected.';
    if (p >= 70) return 'Probably AI-generated — shows typical LLM structure.';
    return 'Likely AI-assisted or heavily edited — mixed signals.';
  }
  if (p >= 90) return 'Very likely human-written — natural variation throughout.';
  if (p >= 70) return 'Probably human-written — some predictable phrasing.';
  return 'Leans human — but not conclusive.';
}
