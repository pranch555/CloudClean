import { create } from 'zustand';

/** Accounts (docs/accounts.md): who is signed in. The app shows the sign-in page until someone is. */
export interface Account {
  username: string;
  admin: boolean;
  created?: string;
  last_login?: string | null;
  last_seen?: string | null;
  logins?: number;
  jobs?: number;
}

interface AuthState {
  status: 'checking' | 'signed-out' | 'signed-in';
  /** false when the server runs without accounts (cloudclean serve --no-accounts) */
  enabled: boolean;
  /** no account exists yet: the next one becomes the admin */
  firstAccount: boolean;
  user: Account | null;
  /** why the sign-in page is showing (e.g. the session ran out) */
  notice: string | null;
}

export const useAuth = create<AuthState>(() => ({ status: 'checking', enabled: true, firstAccount: false, user: null, notice: null }));

async function post(url: string, body?: unknown) {
  const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body) });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Something went wrong - try again');
  return data;
}

export async function checkSession() {
  let res: Response;
  try {
    res = await fetch('/api/auth/status');
  } catch {
    // server unreachable: let the app load and show its own "cannot reach" messages
    useAuth.setState({ status: 'signed-in', enabled: false });
    return;
  }
  const s = await res.json().catch(() => ({}));
  if (!res.ok) {
    // e.g. the accounts file is damaged: the sign-in page says so
    const notice = typeof s.detail === 'string' ? s.detail : 'CloudClean could not check who is signed in - try again';
    useAuth.setState({ status: 'signed-out', enabled: true, firstAccount: false, user: null, notice });
    return;
  }
  useAuth.setState({ enabled: s.enabled, firstAccount: !s.users_exist, user: s.user ?? null, status: s.signed_in ? 'signed-in' : 'signed-out' });
}

export async function signIn(username: string, password: string) {
  const { user } = await post('/api/auth/login', { username, password });
  useAuth.setState({ status: 'signed-in', user, notice: null, firstAccount: false });
}

export async function createAccount(username: string, password: string) {
  const { user } = await post('/api/auth/register', { username, password });
  useAuth.setState({ status: 'signed-in', user, notice: null, firstAccount: false });
}

export async function signOut() {
  await post('/api/auth/logout').catch(() => undefined);
  // a clean start: nothing from this session stays in memory for the next person
  window.location.reload();
}

export async function changePassword(current: string, next: string) {
  const { user } = await post('/api/auth/password', { current, new: next });
  useAuth.setState({ user });
}

/** An API call answered 401: the session ended (signed out elsewhere, removed, or ran out). */
export function sessionEnded() {
  const s = useAuth.getState();
  if (s.enabled && s.status === 'signed-in') useAuth.setState({ status: 'signed-out', notice: 'You were signed out. Sign in again to carry on.' });
}
