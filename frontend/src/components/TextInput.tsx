'use client';

import { useState, useRef, useCallback } from 'react';

interface TextInputProps {
  onSend: (text: string) => Promise<void>;
  disabled: boolean;
}

export default function TextInput({ onSend, disabled }: TextInputProps) {
  const [value, setValue] = useState('');
  const taRef = useRef<HTMLTextAreaElement>(null);

  const autoResize = () => {
    const el = taRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 190) + 'px';
  };

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    autoResize();
  };

  const submit = useCallback(async () => {
    const t = value.trim();
    if (!t || disabled) return;
    setValue('');
    if (taRef.current) taRef.current.style.height = 'auto';
    await onSend(t);
  }, [value, disabled, onSend]);

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const canSend = value.trim().length > 0 && !disabled;

  return (
    <div className="input-area">
      <div className="input-outer">
        <div className="input-box">
          <textarea
            ref={taRef}
            id="text-input"
            className="input-ta"
            placeholder="Paste or type any text to analyze…"
            value={value}
            onChange={handleChange}
            onKeyDown={handleKey}
            rows={1}
            disabled={disabled}
          />
          <button
            id="send-btn"
            className="send-btn"
            onClick={submit}
            disabled={!canSend}
            aria-label="Analyze text"
          >
            {disabled ? <SpinnerIcon /> : <SendIcon />}
          </button>
        </div>
        <p className="input-hint">
          Press <kbd className="kbd">Enter</kbd> to analyze ·{' '}
          <kbd className="kbd">Shift+Enter</kbd> for a new line
        </p>
      </div>
    </div>
  );
}

const SendIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="19" x2="12" y2="5"/>
    <polyline points="5 12 12 5 19 12"/>
  </svg>
);

const SpinnerIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round"
    style={{ animation: 'spin 0.75s linear infinite', transformOrigin: 'center' }}>
    <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"/>
    <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
  </svg>
);
