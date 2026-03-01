'use client';

import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import LogoMark from '@/components/LogoMark';
import { createClient } from '@/lib/supabase/client';

interface PasswordStrength {
  score: number;  // 0-4
  label: string;
  color: string;
  checks: { label: string; passed: boolean }[];
}

function checkPassword(pw: string): PasswordStrength {
  const checks = [
    { label: 'At least 8 characters', passed: pw.length >= 8 },
    { label: 'Uppercase letter (A–Z)',  passed: /[A-Z]/.test(pw) },
    { label: 'Number (0–9)',            passed: /[0-9]/.test(pw) },
    { label: 'Special character (!@#…)',passed: /[^A-Za-z0-9]/.test(pw) },
  ];
  const score = checks.filter(c => c.passed).length;
  const labels = ['', 'Weak', 'Weak', 'Fair', 'Strong'];
  const colors = ['', '#ef4444', '#ef4444', '#f59e0b', '#22c55e'];
  return { score, label: labels[score] || '', color: colors[score] || '#ef4444', checks };
}

export default function SignUpPage() {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [form, setForm] = useState({
    username: '',
    email: '',
    password: '',
    phone: '',
  });

  const strength = checkPassword(form.password);
  const isPasswordValid = strength.score === 4;

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setForm(prev => ({ ...prev, [e.target.name]: e.target.value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isPasswordValid) {
      setError('Please choose a stronger password that meets all requirements.');
      return;
    }

    const supabase = createClient();
    const { error } = await supabase.auth.signUp({
      email: form.email,
      password: form.password,
      options: {
        data: {
          username: form.username,
          phone: form.phone || null,
        },
        emailRedirectTo: `${window.location.origin}/auth/callback`,
      },
    });

    if (error && !error.message.includes('confirmation email')) {
      setError(error.message);
      setLoading(false);
    } else {
      router.replace('/');
    }
  };

  const signInGoogle = async () => {
    const supabase = createClient();
    await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: { redirectTo: `${window.location.origin}/auth/callback` },
    });
  };

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <Link href="/" className="auth-logo" style={{textDecoration:'none'}}>
          <LogoMark size={34} />
          <span className="auth-logo-name">detex<span style={{color:'var(--accent)'}}>.ai</span></span>
        </Link>

        <h1 className="auth-title">Create your account</h1>
        <p className="auth-sub">Start detecting AI-generated text for free</p>

        {error && <p className="auth-error">{error}</p>}

        <form className="auth-form" onSubmit={handleSubmit} noValidate>
          <div className="auth-field">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              name="username"
              type="text"
              placeholder="username"
              autoComplete="username"
              required
              value={form.username}
              onChange={handleChange}
            />
          </div>

          <div className="auth-field">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              name="email"
              type="email"
              placeholder="you@gmail.com"
              autoComplete="email"
              required
              value={form.email}
              onChange={handleChange}
            />
          </div>

          <div className="auth-field">
            <label htmlFor="password">Password</label>
            <input
              id="password"
              name="password"
              type="password"
              placeholder="Create a strong password"
              autoComplete="new-password"
              required
              value={form.password}
              onChange={handleChange}
            />
            {form.password.length > 0 && (
              <div className="pw-strength">
                <div className="pw-strength-bar">
                  {[1,2,3,4].map(i => (
                    <div
                      key={i}
                      className="pw-strength-seg"
                      style={{ background: i <= strength.score ? strength.color : 'var(--border)' }}
                    />
                  ))}
                </div>
                <span className="pw-strength-label" style={{ color: strength.color }}>
                  {strength.label}
                </span>
                <ul className="pw-checks">
                  {strength.checks.map(c => (
                    <li key={c.label} style={{ color: c.passed ? '#22c55e' : 'var(--text-muted)' }}>
                      {c.passed ? '✓' : '✗'} {c.label}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="auth-field">
            <label htmlFor="phone">
              Phone number <span className="auth-optional">(optional)</span>
            </label>
            <input
              id="phone"
              name="phone"
              type="tel"
              placeholder="+1 555 000 0000"
              autoComplete="tel"
              value={form.phone}
              onChange={handleChange}
            />
          </div>

          <button
            type="submit"
            className="auth-submit-btn"
            disabled={loading || !isPasswordValid}
            id="btn-signup"
          >
            {loading ? 'Creating account…' : 'Create account'}
          </button>
        </form>

        <div className="auth-divider">or</div>

        <div className="auth-btns">
          <button
            className="oauth-btn oauth-google"
            onClick={signInGoogle}
            id="btn-google"
          >
            <GoogleIcon />
            Continue with Google
          </button>
        </div>

        <p className="auth-switch">
          Already have an account?{' '}
          <Link href="/sign-in" className="auth-link">Sign in</Link>
        </p>
        <p className="auth-terms">
          By signing up you agree to our{' '}
          <a href="#" className="auth-link">Terms</a> and{' '}
          <a href="#" className="auth-link">Privacy Policy</a>.
        </p>
      </div>
    </div>
  );
}

function GoogleIcon() {
  return (
    <svg width="17" height="17" viewBox="0 0 24 24">
      <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"/>
      <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"/>
      <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"/>
      <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"/>
    </svg>
  );
}
