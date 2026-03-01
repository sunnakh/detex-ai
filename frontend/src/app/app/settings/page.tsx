'use client';

import { useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import LogoMark from '@/components/LogoMark';
import { createClient } from '@/lib/supabase/client';

type Tab = 'profile' | 'account' | 'danger';

export default function SettingsPage() {
  const router = useRouter();
  const supabase = createClient();
  const [tab, setTab] = useState<Tab>('profile');
  const [user, setUser] = useState<any>(null);
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<{ type: 'ok' | 'err'; text: string } | null>(null);

  // Profile fields
  const [fullName, setFullName] = useState('');
  const [phone, setPhone]       = useState('');
  const [bio, setBio]           = useState('');

  // Account fields
  const [newEmail, setNewEmail] = useState('');

  const avatarRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    supabase.auth.getUser().then(({ data }) => {
      if (!data.user) { router.push('/sign-in'); return; }
      setUser(data.user);
      const m = data.user.user_metadata;
      setFullName(m?.full_name || m?.name || '');
      setPhone(m?.phone || '');
      setBio(m?.bio || '');
      setNewEmail(data.user.email || '');
    });
  }, []);

  const flash = (type: 'ok' | 'err', text: string) => {
    setMsg({ type, text });
    setTimeout(() => setMsg(null), 4000);
  };

  const saveProfile = async () => {
    setSaving(true);
    const { error } = await supabase.auth.updateUser({
      data: { full_name: fullName, phone, bio },
    });
    setSaving(false);
    error ? flash('err', error.message) : flash('ok', 'Profile updated!');
  };

  const saveEmail = async () => {
    if (!newEmail || newEmail === user?.email) return;
    setSaving(true);
    const { error } = await supabase.auth.updateUser({ email: newEmail });
    setSaving(false);
    error ? flash('err', error.message) : flash('ok', 'Confirmation email sent — check your inbox.');
  };

  const sendPasswordReset = async () => {
    if (!user?.email) return;
    setSaving(true);
    const { error } = await supabase.auth.resetPasswordForEmail(user.email);
    setSaving(false);
    error ? flash('err', error.message) : flash('ok', 'Password reset email sent.');
  };

  const signOut = async () => {
    await supabase.auth.signOut();
    router.push('/sign-in');
  };

  const deleteAccount = async () => {
    if (!confirm('Are you sure? This cannot be undone.')) return;
    flash('err', 'Account deletion requires backend support — contact support@detex.ai');
  };

  const avatar = user?.user_metadata?.avatar_url;
  const initials = (fullName || user?.email || '?').slice(0, 2).toUpperCase();

  return (
    <div className="settings-shell">
      {/* Sidebar */}
      <aside className="settings-sidebar">
        <Link href="/app" className="settings-back">
          <BackIcon /> Back to app
        </Link>
        <div className="settings-logo">
          <LogoMark size={28} />
          <span>detex<span style={{ color: 'var(--accent)' }}>.ai</span></span>
        </div>
        <nav className="settings-nav">
          {(['profile', 'account', 'danger'] as Tab[]).map(t => (
            <button key={t} className={`settings-nav-btn${tab === t ? ' active' : ''}`} onClick={() => setTab(t)}>
              {tabIcon(t)} {tabLabel(t)}
            </button>
          ))}
        </nav>
      </aside>

      {/* Content */}
      <main className="settings-main">
        {/* Flash message */}
        {msg && (
          <div className={`settings-flash ${msg.type}`}>{msg.text}</div>
        )}

        {/* ── Profile tab ── */}
        {tab === 'profile' && (
          <div className="settings-section">
            <h2 className="settings-h2">Profile</h2>
            <p className="settings-desc">Update your display name, bio, and contact info.</p>

            {/* Avatar */}
            <div className="settings-avatar-row">
              <div className="settings-avatar">
                {avatar ? <img src={avatar} alt="avatar" className="settings-avatar-img" /> : <span>{initials}</span>}
              </div>
              <div>
                <p className="settings-avatar-hint">Profile photo comes from your Google / GitHub account.</p>
              </div>
            </div>

            <div className="settings-form">
              <label className="settings-label">Full name
                <input className="settings-input" value={fullName} onChange={e => setFullName(e.target.value)} placeholder="Your name" />
              </label>

              <label className="settings-label">Phone number
                <input className="settings-input" value={phone} onChange={e => setPhone(e.target.value)} placeholder="+1 234 567 8900" type="tel" />
              </label>

              <label className="settings-label">Bio
                <textarea className="settings-input settings-textarea" value={bio} onChange={e => setBio(e.target.value)} placeholder="A short bio…" rows={3} />
              </label>

              <button className="settings-btn-primary" onClick={saveProfile} disabled={saving}>
                {saving ? 'Saving…' : 'Save profile'}
              </button>
            </div>
          </div>
        )}

        {/* ── Account tab ── */}
        {tab === 'account' && (
          <div className="settings-section">
            <h2 className="settings-h2">Account</h2>
            <p className="settings-desc">Manage your email address and security settings.</p>

            <div className="settings-form">
              <label className="settings-label">Email address
                <input className="settings-input" value={newEmail} onChange={e => setNewEmail(e.target.value)} type="email" />
              </label>
              <button className="settings-btn-primary" onClick={saveEmail} disabled={saving || newEmail === user?.email}>
                {saving ? 'Sending…' : 'Update email'}
              </button>

              <div className="settings-divider" />

              <div className="settings-info-row">
                <div>
                  <div className="settings-info-title">Password</div>
                  <div className="settings-info-sub">Send a password reset link to your email.</div>
                </div>
                <button className="settings-btn-outline" onClick={sendPasswordReset} disabled={saving}>
                  Send reset link
                </button>
              </div>

              <div className="settings-divider" />

              <div className="settings-info-row">
                <div>
                  <div className="settings-info-title">Signed in as</div>
                  <div className="settings-info-sub">{user?.email}</div>
                </div>
                <div className="settings-provider-pill">
                  {user?.app_metadata?.provider === 'google' ? '🔵 Google' : user?.app_metadata?.provider === 'github' ? '⚫ GitHub' : '✉️ Email'}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── Danger zone tab ── */}
        {tab === 'danger' && (
          <div className="settings-section">
            <h2 className="settings-h2" style={{ color: 'var(--ai-color)' }}>Danger zone</h2>
            <p className="settings-desc">Irreversible actions — proceed with caution.</p>

            <div className="settings-danger-card">
              <div className="settings-info-row">
                <div>
                  <div className="settings-info-title">Sign out</div>
                  <div className="settings-info-sub">Sign out of your account on this device.</div>
                </div>
                <button className="settings-btn-outline" onClick={signOut}>Sign out</button>
              </div>
              <div className="settings-divider" />
              <div className="settings-info-row">
                <div>
                  <div className="settings-info-title">Delete account</div>
                  <div className="settings-info-sub">Permanently delete your account and all data. This cannot be undone.</div>
                </div>
                <button className="settings-btn-danger" onClick={deleteAccount}>Delete account</button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

function tabLabel(t: Tab) {
  return { profile: 'Profile', account: 'Account', danger: 'Danger zone' }[t];
}
function tabIcon(t: Tab) {
  if (t === 'profile') return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>;
  if (t === 'account') return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>;
  return <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>;
}
const BackIcon = () => <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d="M19 12H5"/><path d="M12 19l-7-7 7-7"/></svg>;
