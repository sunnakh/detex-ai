'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import Sidebar from '@/components/Sidebar';
import Header from '@/components/Header';
import ChatPanel from '@/components/ChatPanel';
import TextInput from '@/components/TextInput';
import { createClient } from '@/lib/supabase/client';
import {
  dbLoadSessions, dbCreateSession, dbUpdateSessionTitle,
  dbSaveMessage, dbLoadMessages,
} from '@/lib/supabase/db';
import type { Session, Theme, QuickPrompt, DetectResult } from '@/types';

let _idCounter = 0;
const nextId = () => ++_idCounter;

const QUICK_PROMPTS: QuickPrompt[] = [
  {
    title: 'Test AI text',
    sub: 'Paste a ChatGPT response to analyze',
    text: 'The implementation of large language models has revolutionized natural language processing by enabling unprecedented capabilities in text generation, summarization, and comprehension. These architectures leverage self-attention mechanisms that allow them to capture long-range dependencies within sequential data.',
  },
  {
    title: 'Test human text',
    sub: 'Paste a blog post or essay excerpt',
    text: "I honestly wasn't sure what to write here. My dog knocked over my coffee this morning before I even got to my desk, and now the whole day feels slightly off. Maybe that's just Tuesday.",
  },
  {
    title: 'News article snippet',
    sub: 'Check if news was AI-generated',
    text: 'Officials confirmed Tuesday that the proposed infrastructure bill would allocate approximately $1.2 trillion toward road repairs, broadband expansion, and public transit upgrades across the country over the next decade.',
  },
  {
    title: 'Academic writing',
    sub: 'Analyze a paragraph from a paper',
    text: 'This study investigates the causal relationship between socioeconomic factors and educational outcomes, utilizing a longitudinal dataset spanning three decades. The methodology employs instrumental variable estimation to address potential endogeneity bias.',
  },
];

function makeLocalSession(title = 'New analysis', dbId?: string): Session {
  return { id: nextId(), dbId, title, messages: [], lastResult: null, loaded: true };
}

export default function AppPage() {
  const router = useRouter();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeId, setActiveId] = useState<number | null>(null);
  const [theme, setTheme] = useState<Theme>('light');
  const [loading, setLoading] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [hydrated, setHydrated] = useState(false);

  // Track pending DB session creation
  const pendingDbId = useRef<Record<number, Promise<string | null>>>({});

  // ── Theme ──────────────────────────────────────────────────────────────────
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
  }, [theme]);

  // ── Hydrate from Supabase on mount ─────────────────────────────────────────
  useEffect(() => {
    (async () => {
      const rows = await dbLoadSessions();
      if (rows.length === 0) {
        // No history — create a blank local session (will persist on first send)
        const blank = makeLocalSession();
        setSessions([blank]);
        setActiveId(blank.id);
      } else {
        const loaded = rows.map(r => ({
          id: nextId(),
          dbId: r.id,
          title: r.title,
          messages: [],
          lastResult: null,
          loaded: false,
        }));
        setSessions(loaded);
        setActiveId(loaded[0].id);
        // Eagerly load messages for the first/most-recent session
        loadMessagesFor(loaded[0]);
      }
      setHydrated(true);
    })();
  }, []);

  // ── Load messages for a session from DB ──────────────────────────────────
  const loadMessagesFor = useCallback(async (s: Session) => {
    if (!s.dbId || s.loaded) return;
    const rows = await dbLoadMessages(s.dbId);
    const messages = rows.map(r => ({
      id: nextId(),
      role: r.role as 'user' | 'result' | 'error',
      text: r.text ?? undefined,
      result: r.result as DetectResult | undefined,
    }));
    const lastResult = [...messages].reverse().find(m => m.role === 'result')?.result ?? null;
    setSessions(prev =>
      prev.map(x => x.id === s.id ? { ...x, messages, lastResult, loaded: true } : x),
    );
  }, []);

  // ── Switch session — lazy-load messages ───────────────────────────────────
  const handleSelect = useCallback((id: number) => {
    setActiveId(id);
    const s = sessions.find(x => x.id === id);
    if (s && !s.loaded) loadMessagesFor(s);
  }, [sessions, loadMessagesFor]);

  // ── Derived active session ────────────────────────────────────────────────
  const activeSession = sessions.find(s => s.id === activeId) ?? sessions[0];

  const updateSession = useCallback(
    (id: number, updater: (s: Session) => Session) =>
      setSessions(prev => prev.map(s => s.id === id ? updater(s) : s)),
    [],
  );

  // ── Add new session ───────────────────────────────────────────────────────
  const addSession = useCallback(async () => {
    const s = makeLocalSession();
    setSessions(prev => [s, ...prev]);
    setActiveId(s.id);
    // Create in Supabase immediately in the background
    const p = dbCreateSession('New analysis').then(dbId => {
      if (dbId) setSessions(prev => prev.map(x => x.id === s.id ? { ...x, dbId } : x));
      return dbId;
    });
    pendingDbId.current[s.id] = p;
  }, []);

  // ── Ensure session has a DB row, return its UUID ──────────────────────────
  const ensureDbId = useCallback(async (localId: number): Promise<string | null> => {
    const s = sessions.find(x => x.id === localId);
    if (s?.dbId) return s.dbId;
    if (localId in pendingDbId.current) return pendingDbId.current[localId];
    const p = dbCreateSession('New analysis').then(dbId => {
      if (dbId) setSessions(prev => prev.map(x => x.id === localId ? { ...x, dbId } : x));
      return dbId;
    });
    pendingDbId.current[localId] = p;
    return p;
  }, [sessions]);

  // ── Handle text send ──────────────────────────────────────────────────────
  const handleSend = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || loading || !activeSession) return;

    const sessionId = activeSession.id;
    setLoading(true);

    const thinkingId = nextId();
    const isFirst = activeSession.messages.length === 0;
    const newTitle = isFirst
      ? trimmed.slice(0, 44) + (trimmed.length > 44 ? '…' : '')
      : activeSession.title;

    updateSession(sessionId, s => ({
      ...s, title: newTitle,
      messages: [
        ...s.messages,
        { id: nextId(), role: 'user', text: trimmed },
        { id: thinkingId, role: 'thinking' },
      ],
    }));

    // Get or create the DB session UUID
    const dbId = await ensureDbId(sessionId);

    // Persist user message + update title (fire-and-forget)
    if (dbId) dbSaveMessage(dbId, 'user', trimmed);
    if (dbId && isFirst) dbUpdateSessionTitle(dbId, newTitle);

    try {
      const res = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: trimmed }),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data: DetectResult = await res.json();

      updateSession(sessionId, s => ({
        ...s, lastResult: data,
        messages: s.messages
          .filter(m => m.id !== thinkingId)
          .concat({ id: nextId(), role: 'result', result: data, inputText: trimmed }),
      }));

      // Persist result
      if (dbId) dbSaveMessage(dbId, 'result', undefined, data);
    } catch (err: unknown) {
      const msg = err instanceof Error && err.message.includes('fetch')
        ? 'Could not reach the API server. Make sure the backend is running on http://localhost:8000'
        : `Error: ${err instanceof Error ? err.message : String(err)}`;

      updateSession(sessionId, s => ({
        ...s,
        messages: s.messages
          .filter(m => m.id !== thinkingId)
          .concat({ id: nextId(), role: 'error', text: msg }),
      }));

      if (dbId) dbSaveMessage(dbId, 'error', msg);

    } finally {
      setLoading(false);
    }
  }, [activeSession, loading, updateSession, ensureDbId]);

  // ── Handle file upload ────────────────────────────────────────────────────
  const handleFileUpload = useCallback(async (file: File) => {
    if (loading || !activeSession) return;

    const sessionId = activeSession.id;
    setLoading(true);

    const thinkingId = nextId();
    const isFirst = activeSession.messages.length === 0;
    const sessionTitle = isFirst ? file.name : activeSession.title;
    const userLabel = `📎 ${file.name}`;

    updateSession(sessionId, s => ({
      ...s,
      title: sessionTitle,
      messages: [
        ...s.messages,
        { id: nextId(), role: 'user', text: userLabel },
        { id: thinkingId, role: 'thinking' },
      ],
    }));

    const dbId = await ensureDbId(sessionId);
    if (dbId) dbSaveMessage(dbId, 'user', userLabel);
    if (dbId && isFirst) dbUpdateSessionTitle(dbId, sessionTitle);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';
      const form = new FormData();
      form.append('file', file);

      const res = await fetch(`${apiUrl}/detect-file`, {
        method: 'POST',
        body: form,
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: `API error ${res.status}` }));
        throw new Error(detail.detail ?? `API error ${res.status}`);
      }

      const data: DetectResult = await res.json();

      updateSession(sessionId, s => ({
        ...s, lastResult: data,
        messages: s.messages
          .filter(m => m.id !== thinkingId)
          .concat({ id: nextId(), role: 'result', result: data }),
      }));

      if (dbId) dbSaveMessage(dbId, 'result', undefined, data);
    } catch (err: unknown) {
      const msg = err instanceof Error
        ? err.message
        : `Error: ${String(err)}`;

      updateSession(sessionId, s => ({
        ...s,
        messages: s.messages
          .filter(m => m.id !== thinkingId)
          .concat({ id: nextId(), role: 'error', text: msg }),
      }));

      if (dbId) dbSaveMessage(dbId, 'error', msg);
    } finally {
      setLoading(false);
    }
  }, [activeSession, loading, updateSession, ensureDbId]);

  const handleSignOut = async () => {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push('/sign-in');
  };

  if (!hydrated) {
    return (
      <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'100vh', color:'var(--text-muted)', fontSize:'14px' }}>
        Loading your history…
      </div>
    );
  }

  return (
    <div className="app-shell">
      <Sidebar
        sessions={sessions}
        activeId={activeId ?? 0}
        onSelect={handleSelect}
        onNew={addSession}
        open={sidebarOpen}
        onToggle={() => setSidebarOpen(o => !o)}
      />
      <div className="main-panel">
        <Header
          theme={theme}
          onToggleTheme={() => setTheme(t => t === 'dark' ? 'light' : 'dark')}
          sidebarOpen={sidebarOpen}
          onToggleSidebar={() => setSidebarOpen(o => !o)}
          onSignOut={handleSignOut}
        />
        <ChatPanel
          session={activeSession ?? sessions[0]}
          quickPrompts={QUICK_PROMPTS}
          onQuickPrompt={handleSend}
        />
        <TextInput onSend={handleSend} onFileUpload={handleFileUpload} disabled={loading} />
      </div>
    </div>
  );
}
