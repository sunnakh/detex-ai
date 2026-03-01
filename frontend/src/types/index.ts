export interface DetectResult {
  label: 'AI-generated' | 'Human-written';
  confidence: number;
  ai_score: number;
  human_score: number;
  word_count: number;
  char_count: number;
  analysis_time_ms: number;
}

export type MessageRole = 'user' | 'thinking' | 'result' | 'error';

export interface Message {
  id: number;
  role: MessageRole;
  text?: string;
  result?: DetectResult;
  inputText?: string;
}

export interface Session {
  id: number;
  dbId?: string;        // Supabase UUID — undefined for sessions not yet persisted
  title: string;
  messages: Message[];
  lastResult: DetectResult | null;
  loaded?: boolean;     // true once messages have been fetched from DB
}

export type Theme = 'light' | 'dark';

export interface QuickPrompt {
  title: string;
  sub: string;
  text: string;
}
