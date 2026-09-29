import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { LogIn, UserPlus } from 'lucide-react';
import { checkSession, createAccount, signIn, useAuth } from '../lib/auth';
import { Logo } from '../ui/icons';
import { Button } from '../ui/primitives';

/** Shows the app once someone is signed in, the sign-in page until then (docs/accounts.md). */
export function AuthGate({ children }: { children: ReactNode }) {
  const status = useAuth(s => s.status);
  useEffect(() => {
    checkSession();
  }, []);
  if (status === 'checking') return <div className="auth-page" aria-busy="true" />;
  if (status === 'signed-out') return <AuthScreen />;
  return <>{children}</>;
}

function AuthScreen() {
  const { firstAccount, notice } = useAuth();
  const [mode, setMode] = useState<'in' | 'new'>(firstAccount ? 'new' : 'in');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const nameRef = useRef<HTMLInputElement>(null);

  useEffect(() => nameRef.current?.focus(), [mode]);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (mode === 'new' && password !== confirm) {
      setError('The two passwords are not the same');
      return;
    }
    setBusy(true);
    try {
      if (mode === 'in') await signIn(username.trim(), password);
      else await createAccount(username.trim(), password);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  };

  const switchTo = (m: 'in' | 'new') => {
    setMode(m);
    setError(null);
    setPassword('');
    setConfirm('');
  };

  return (
    <main className="auth-page">
      <section className="auth-card" aria-labelledby="auth-title">
        <div className="auth-brand">
          <Logo size={34} />
          <span className="auth-wordmark">CloudClean</span>
        </div>
        <h1 id="auth-title" className="auth-title">{mode === 'in' ? 'Sign in' : 'Create your account'}</h1>
        <p className="auth-sub">
          {mode === 'in'
            ? 'Welcome back. Use your username and password.'
            : firstAccount
              ? 'You are the first here: this account will be the admin, who can see everyone who uses CloudClean.'
              : 'Pick a username and a password. Your projects are shared with everyone who uses this CloudClean.'}
        </p>
        {notice && mode === 'in' && <p className="auth-notice" role="status">{notice}</p>}

        <div className="auth-tabs" role="tablist" aria-label="Sign in or create an account">
          <button type="button" role="tab" aria-selected={mode === 'in'} className={mode === 'in' ? 'is-on' : ''} onClick={() => switchTo('in')}>
            Sign in
          </button>
          <button type="button" role="tab" aria-selected={mode === 'new'} className={mode === 'new' ? 'is-on' : ''} onClick={() => switchTo('new')}>
            Create account
          </button>
        </div>

        <form className="auth-form" onSubmit={submit} noValidate>
          <label className="auth-field">
            <span>Username</span>
            <input ref={nameRef} className="input" name="username" autoComplete="username" autoCapitalize="none" spellCheck={false} required value={username} onChange={e => setUsername(e.target.value)} />
          </label>
          <label className="auth-field">
            <span>Password</span>
            <input className="input" type="password" name="password" autoComplete={mode === 'in' ? 'current-password' : 'new-password'} required value={password} onChange={e => setPassword(e.target.value)} />
            {mode === 'new' && <small>At least 8 characters.</small>}
          </label>
          {mode === 'new' && (
            <label className="auth-field">
              <span>Password again</span>
              <input className="input" type="password" name="confirm" autoComplete="new-password" required value={confirm} onChange={e => setConfirm(e.target.value)} />
            </label>
          )}
          {error && (
            <p className="auth-error" role="alert">
              {error}
            </p>
          )}
          <Button type="submit" variant="primary" size="lg" block loading={busy} disabled={!username.trim() || !password} icon={mode === 'in' ? <LogIn size={18} /> : <UserPlus size={18} />}>
            {mode === 'in' ? 'Sign in' : 'Create account'}
          </Button>
        </form>
        <p className="auth-switch">
          {mode === 'in' ? (
            <>
              New here?{' '}
              <button type="button" className="link" onClick={() => switchTo('new')}>
                Create an account
              </button>
            </>
          ) : (
            <>
              Already have an account?{' '}
              <button type="button" className="link" onClick={() => switchTo('in')}>
                Sign in
              </button>
            </>
          )}
        </p>
      </section>
    </main>
  );
}
