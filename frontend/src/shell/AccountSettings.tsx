import { useEffect, useState, type FormEvent } from 'react';
import { Copy, KeyRound, LogOut, RefreshCw, Trash2 } from 'lucide-react';
import { api } from '../lib/api';
import { changePassword, signOut, useAuth, type Account } from '../lib/auth';
import { fmtAgo } from '../lib/format';
import { useStore } from '../store';
import { Button } from '../ui/primitives';
import { ConfirmDialog } from '../steps/capture/parts';

const when = (iso?: string | null) => (iso ? fmtAgo(iso) : 'never');

/** Settings -> Account: who is signed in, change the password, sign out. */
export function AccountSection() {
  const user = useAuth(s => s.user);
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [again, setAgain] = useState('');
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  if (!user) return null;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setMsg(null);
    if (next !== again) return setMsg({ ok: false, text: 'The two new passwords are not the same' });
    setBusy(true);
    try {
      await changePassword(current, next);
      setCurrent('');
      setNext('');
      setAgain('');
      setMsg({ ok: true, text: 'Password changed. Your other devices were signed out.' });
    } catch (err) {
      setMsg({ ok: false, text: (err as Error).message });
    }
    setBusy(false);
  };

  return (
    <div className="stack">
      <p className="hint-text">
        Signed in as <b>{user.username}</b>
        {user.admin ? ' (admin)' : ''}. Everyone who uses this CloudClean shares the same projects; CloudClean keeps track of who signed in and who started
        each job.
      </p>
      <form className="auth-form" onSubmit={submit} style={{ maxWidth: 360 }}>
        <h3 className="block-title">Change password</h3>
        <label className="auth-field">
          <span>Current password</span>
          <input className="input" type="password" autoComplete="current-password" value={current} onChange={e => setCurrent(e.target.value)} />
        </label>
        <label className="auth-field">
          <span>New password</span>
          <input className="input" type="password" autoComplete="new-password" value={next} onChange={e => setNext(e.target.value)} />
          <small>At least 8 characters.</small>
        </label>
        <label className="auth-field">
          <span>New password again</span>
          <input className="input" type="password" autoComplete="new-password" value={again} onChange={e => setAgain(e.target.value)} />
        </label>
        {msg && (
          <p className={msg.ok ? 'hint-text' : 'auth-error'} role={msg.ok ? 'status' : 'alert'}>
            {msg.text}
          </p>
        )}
        <div className="row">
          <Button type="submit" variant="primary" loading={busy} disabled={!current || !next}>
            Change password
          </Button>
        </div>
      </form>
      <div className="row">
        <Button icon={<LogOut size={16} />} onClick={signOut}>
          Sign out
        </Button>
      </div>
    </div>
  );
}

interface Activity {
  time: string;
  event: string;
  user: string | null;
  detail: string;
}

const EVENT: Record<string, string> = {
  register: 'created an account',
  login: 'signed in',
  logout: 'signed out',
  job: 'started',
  password: 'changed their password',
  removed: 'was removed',
  machine_key: 'made a new bridge key',
};

/** One activity line. A failed sign-in names the account only when it exists (the server never logs other text typed as a name). */
function describe(a: Activity) {
  if (a.event === 'failed_login') {
    const what = a.user ? `wrong password for ${a.user}` : 'sign-in with a username that does not exist';
    return a.detail ? `${what} (from ${a.detail})` : what;
  }
  return `${a.user ?? 'someone'} ${EVENT[a.event] ?? a.event}${a.event === 'job' && a.detail ? `: ${a.detail}` : ''}`;
}

/** Settings -> Users (admin): everyone who uses CloudClean, what they did lately, the bridge key. */
export function UsersSection() {
  const me = useAuth(s => s.user);
  const [users, setUsers] = useState<(Account & { signed_in_on?: number })[]>([]);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [key, setKey] = useState<string | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () =>
    api
      .get<{ users: (Account & { signed_in_on?: number })[]; activity: Activity[] }>('/api/auth/users')
      .then(r => {
        setUsers(r.users);
        setActivity(r.activity);
      })
      .catch(err => setError((err as Error).message));

  useEffect(() => {
    load();
  }, []);

  if (!me?.admin) return <p className="hint-text">Only the admin can see the users.</p>;

  const showKey = async () => setKey((await api.get<{ key: string }>('/api/auth/machine-key')).key);
  const newKey = async () => setKey((await api.post<{ key: string }>('/api/auth/machine-key', {})).key);
  const remove = async (name: string) => {
    setRemoving(null);
    try {
      await api.del(`/api/auth/users/${encodeURIComponent(name)}`);
      load();
    } catch (err) {
      setError((err as Error).message);
    }
  };

  return (
    <div className="stack">
      {error && <p className="auth-error" role="alert">{error}</p>}
      <div className="users-scroll">
        <table className="users-table">
          <thead>
            <tr>
              <th scope="col">User</th>
              <th scope="col">Joined</th>
              <th scope="col">Last seen</th>
              <th scope="col">Sign-ins</th>
              <th scope="col">Jobs</th>
              <th scope="col"><span className="sr-only">Remove</span></th>
            </tr>
          </thead>
          <tbody>
            {users.map(u => (
              <tr key={u.username}>
                <td>
                  <b>{u.username}</b>
                  {u.admin ? ' · admin' : ''}
                  {u.signed_in_on ? <span className="muted"> · signed in on {u.signed_in_on}</span> : null}
                </td>
                <td>{when(u.created)}</td>
                <td>{when(u.last_seen)}</td>
                <td>{u.logins ?? 0}</td>
                <td>{u.jobs ?? 0}</td>
                <td>
                  {u.username.toLowerCase() !== me.username.toLowerCase() && (
                    <button type="button" className="icon-link danger" aria-label={`Remove ${u.username}`} onClick={() => setRemoving(u.username)}>
                      <Trash2 size={15} />
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h3 className="block-title">Recent activity</h3>
      <ul className="activity-list">
        {activity.map((a, i) => (
          <li key={i}>
            <time dateTime={a.time}>{new Date(a.time).toLocaleString()}</time>
            <span>{describe(a)}</span>
          </li>
        ))}
        {!activity.length && <li>Nothing yet.</li>}
      </ul>

      <h3 className="block-title">Key for the Revo Metro bridge and scripts</h3>
      <p className="hint-text">
        The bridge on the scanning PC cannot sign in, so it uses this key: add <code>--key &lt;key&gt;</code> to the bridge command (Scan → Or bring in files shows
        it with the key). Anyone with the key can use CloudClean like a signed-in user - keep it private; make a new one if it leaks.
      </p>
      <div className="row wrap">
        {key ? (
          <>
            <code className="mono" style={{ wordBreak: 'break-all' }}>{key}</code>
            <Button size="sm" icon={<Copy size={14} />} onClick={() => navigator.clipboard.writeText(key).catch(() => undefined)}>
              Copy
            </Button>
          </>
        ) : (
          <Button size="sm" icon={<KeyRound size={14} />} onClick={showKey}>
            Show the key
          </Button>
        )}
        <Button size="sm" icon={<RefreshCw size={14} />} onClick={newKey}>
          Make a new key
        </Button>
      </div>

      {removing && (
        <ConfirmDialog
          title={`Remove ${removing}?`}
          choices={[{ label: 'Remove', variant: 'danger', onPick: () => remove(removing) }]}
          onCancel={() => setRemoving(null)}
        >
          <p className="hint-text">{removing} can no longer sign in. Their projects stay: everyone shares them.</p>
        </ConfirmDialog>
      )}
    </div>
  );
}

/** The signed-in person in the top bar, with Account, Users (admin) and Sign out. */
export function AccountChip() {
  const { enabled, user } = useAuth();
  if (!enabled || !user) return null;
  const open = (id: string) => useStore.getState().set({ settingsOpen: id });
  return (
    <button type="button" className="account-chip" onClick={() => open('account')} aria-label={`Account: ${user.username}`} title="Account and sign out">
      <span className="account-avatar" aria-hidden>{user.username.slice(0, 1)}</span>
      <span className="account-name">{user.username}</span>
    </button>
  );
}
