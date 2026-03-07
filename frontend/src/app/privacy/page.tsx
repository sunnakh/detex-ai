import Link from 'next/link';
import LogoMark from '@/components/LogoMark';

export const metadata = {
  title: 'Privacy Policy — detex.ai',
  description: 'How detex.ai collects, uses, and protects your data.',
};

export default function PrivacyPage() {
  return (
    <div className="legal-shell">
      {/* Nav */}
      <nav className="legal-nav">
        <Link href="/" className="legal-logo">
          <LogoMark size={26} />
          <span>detex<span style={{ color: 'var(--accent)' }}>.ai</span></span>
        </Link>
        <Link href="/" className="legal-back">← Back to home</Link>
      </nav>

      <main className="legal-main">
        <div className="legal-header">
          <p className="legal-label">Legal</p>
          <h1>Privacy Policy</h1>
          <p className="legal-meta">Last updated: February 2026</p>
        </div>

        <div className="legal-body">
          <section>
            <h2>1. Overview</h2>
            <p>
              detex.ai (&quot;we&quot;, &quot;us&quot;, &quot;our&quot;) is built by <strong>neurobrain-ai</strong> and provides AI text detection services.
              This policy explains what data we collect, how we use it, and your rights as a user.
              We believe in minimal data collection — we only collect what&apos;s necessary to provide the service.
            </p>
          </section>

          <section>
            <h2>2. Data We Collect</h2>
            <h3>Account Data</h3>
            <p>When you create an account, we collect:</p>
            <ul>
              <li>Email address (required)</li>
              <li>Username (required)</li>
              <li>Phone number (optional)</li>
            </ul>
            <p>This data is stored securely in <strong>Supabase</strong>, our database provider, and is protected by Row Level Security (RLS) — meaning only you can access your own data.</p>

            <h3>Analysis Data</h3>
            <p>
              Text you submit for analysis is sent to our backend API only to generate a detection result.
              <strong> We do not permanently store the text you analyze.</strong> Only the result metadata
              (score, label, timestamp) is saved to your session history.
            </p>

            <h3>Usage Data</h3>
            <p>We do not use third-party analytics trackers. We may log server-side request counts and error rates for performance monitoring purposes only.</p>
          </section>

          <section>
            <h2>3. How We Use Your Data</h2>
            <ul>
              <li>To provide and maintain the AI detection service</li>
              <li>To authenticate you and keep your session history</li>
              <li>To send account-related emails (e.g. verification, password reset)</li>
              <li>To improve detection accuracy using aggregated, anonymized statistics</li>
            </ul>
            <p>We do not sell, rent, or share your personal data with third parties for marketing purposes.</p>
          </section>

          <section>
            <h2>4. Third-Party Services</h2>
            <p>We use the following trusted third-party providers:</p>
            <ul>
              <li><strong>Supabase</strong> — Authentication and database (GDPR-compliant, data stored in EU/US)</li>
              <li><strong>Resend</strong> — Transactional email delivery</li>
              <li><strong>Hugging Face</strong> — Model weights are downloaded from Hugging Face and run locally on our servers. Your text is never sent to Hugging Face.</li>
            </ul>
          </section>

          <section>
            <h2>5. Your Rights</h2>
            <p>You have the right to:</p>
            <ul>
              <li>Access the data we hold about you</li>
              <li>Request correction of inaccurate data</li>
              <li>Delete your account and all associated data</li>
              <li>Export your data in a portable format</li>
            </ul>
            <p>To exercise any of these rights, contact us at <a href="https://sunnakh.com/contact/" target="_blank" rel="noopener noreferrer">sunnakh.com/contact</a>.</p>
          </section>

          <section>
            <h2>6. Cookies</h2>
            <p>
              We use only essential cookies required for authentication (session tokens managed by Supabase).
              We do not use advertising or tracking cookies.
            </p>
          </section>

          <section>
            <h2>7. Changes to This Policy</h2>
            <p>
              We may update this Privacy Policy from time to time. We will notify you of significant changes
              by updating the &quot;Last updated&quot; date above. Continued use of the service after changes constitutes acceptance.
            </p>
          </section>

          <section>
            <h2>8. Contact</h2>
            <p>
              If you have any questions about this Privacy Policy, please reach out at{' '}
              <a href="https://sunnakh.com/contact/" target="_blank" rel="noopener noreferrer">sunnakh.com/contact</a>.
            </p>
          </section>
        </div>

        <div className="legal-footer-links">
          <Link href="/terms">Terms of Service →</Link>
        </div>
      </main>
    </div>
  );
}
