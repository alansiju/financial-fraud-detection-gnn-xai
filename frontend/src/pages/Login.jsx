import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { Shield, Eye, EyeOff, AlertCircle, Lock, User } from 'lucide-react';

export default function Login() {
  const { login, loading } = useAuth();
  const navigate = useNavigate();
  const [form, setForm]   = useState({ username: '', password: '' });
  const [showPwd, setShowPwd] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    const res = await login(form.username, form.password);
    if (res.success) navigate('/');
    else setError(res.error);
  };

  return (
    <div className="login-page">
      <div className="login-bg-grid"/>
      {/* Blobs */}
      <div className="login-blob" style={{ width: 500, height: 500, background: 'radial-gradient(circle, color-mix(in srgb, var(--brand) 12%, transparent) 0%, transparent 70%)', top: -100, left: -100 }}/>
      <div className="login-blob" style={{ width: 400, height: 400, background: 'radial-gradient(circle, color-mix(in srgb, var(--cat-4) 10%, transparent) 0%, transparent 70%)', bottom: -50, right: -50 }}/>

      {/* Left panel */}
      <div style={{
        flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center',
        padding: '60px 80px', position: 'relative',
        background: 'linear-gradient(135deg, color-mix(in srgb, var(--bg-secondary) 95%, transparent) 0%, color-mix(in srgb, var(--bg-primary) 90%, transparent) 100%)',
        borderRight: '1px solid var(--border)'
      }} className="login-left">
        <div style={{ maxWidth: 480 }}>
          {/* Brand */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginBottom: 60 }}>
            <div style={{
              width: 48, height: 48, borderRadius: 12,
              background: 'linear-gradient(135deg, var(--brand), var(--cat-4))',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              boxShadow: '0 8px 24px color-mix(in srgb, var(--brand) 35%, transparent)'
            }}>
              <Shield size={24} color="var(--on-brand)"/>
            </div>
            <div>
              <div style={{ fontSize: 20, fontWeight: 800, letterSpacing: -0.5 }}>
                Fraud<span style={{ color: 'var(--brand)' }}>Shield</span> AI
              </div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 1 }}>
                GraphSAGE Fraud Detection Platform
              </div>
            </div>
          </div>

          <h1 style={{ fontSize: 36, fontWeight: 900, letterSpacing: -1, lineHeight: 1.1, marginBottom: 12 }}>
            Detect fraud<br/>
            <span style={{ color: 'var(--brand)' }}>before it happens</span>
          </h1>
          <p style={{ fontSize: 15, color: 'var(--text-secondary)', lineHeight: 1.7, marginBottom: 48 }}>
            Graph Neural Network fraud detection on the IEEE-CIS dataset, with explainable predictions and an honest view of what the model can and cannot do.
          </p>

          {/* Stats */}
          <div style={{ display: 'flex', gap: 32 }}>
            {/* Dataset facts, not model claims. The previous row advertised
                "96.8% Accuracy / 0.896 ROC-AUC" from a run that no longer
                exists, to a visitor with no way to verify either. */}
            {[['590,540', 'Transactions'], ['3.5%', 'Fraud rate'], ['2.3M', 'Graph edges']].map(([v, l]) => (
              <div key={l}>
                <div style={{ fontSize: 23, fontWeight: 700, color: 'var(--brand)', fontFamily: 'var(--font-display)' }}>{v}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{l}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right panel — login form */}
      <div style={{
        width: 480, display: 'flex', flexDirection: 'column',
        justifyContent: 'center', alignItems: 'center',
        padding: '60px 48px', position: 'relative'
      }}>
        <div style={{ width: '100%', maxWidth: 360 }}>
          <div style={{ marginBottom: 36 }}>
            <h2 style={{ fontSize: 26, fontWeight: 800, letterSpacing: -0.5, marginBottom: 6 }}>
              Sign in
            </h2>
            <p style={{ fontSize: 13.5, color: 'var(--text-muted)' }}>
              Enter your credentials to access the platform
            </p>
          </div>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
            {/* Username */}
            <div className="form-group">
              <label className="form-label">Username</label>
              <div className="input-group">
                <User className="input-icon" size={15}/>
                <input
                  id="username"
                  className="input"
                  type="text"
                  placeholder="admin"
                  value={form.username}
                  onChange={e => setForm(f => ({ ...f, username: e.target.value }))}
                  required
                  autoFocus
                />
              </div>
            </div>

            {/* Password */}
            <div className="form-group">
              <label className="form-label">Password</label>
              <div style={{ position: 'relative' }}>
                <div className="input-group">
                  <Lock className="input-icon" size={15}/>
                  <input
                    id="password"
                    className="input"
                    type={showPwd ? 'text' : 'password'}
                    placeholder="••••••••"
                    value={form.password}
                    onChange={e => setForm(f => ({ ...f, password: e.target.value }))}
                    required
                    style={{ paddingRight: 40 }}
                  />
                </div>
                <button
                  type="button"
                  onClick={() => setShowPwd(v => !v)}
                  style={{
                    position: 'absolute', right: 12, top: '50%', transform: 'translateY(-50%)',
                    background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)'
                  }}
                >
                  {showPwd ? <EyeOff size={15}/> : <Eye size={15}/>}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '10px 14px', borderRadius: 8,
                background: 'color-mix(in srgb, var(--status-critical) 10%, transparent)', border: '1px solid color-mix(in srgb, var(--status-critical) 20%, transparent)',
                color: 'var(--status-critical)', fontSize: 13
              }}>
                <AlertCircle size={15}/>
                {error}
              </div>
            )}

            <button
              type="submit"
              className="btn btn-primary"
              style={{ width: '100%', height: 44, fontSize: 14, marginTop: 4 }}
              disabled={loading}
            >
              {loading ? <><div className="spinner" style={{ width: 16, height: 16 }}/> Signing in…</> : 'Sign In'}
            </button>
          </form>

          {/* Demo credentials */}
          <div style={{
            marginTop: 28, padding: '14px 18px',
            background: 'color-mix(in srgb, var(--brand) 6%, transparent)',
            border: '1px solid color-mix(in srgb, var(--brand) 15%, transparent)',
            borderRadius: 10
          }}>
            <div style={{ fontSize: 11.5, fontWeight: 700, color: 'var(--brand)', marginBottom: 8, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
              Demo Credentials
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              {[['Admin', 'admin', 'fraud123'], ['Analyst', 'analyst', 'demo456']].map(([role, u, p]) => (
                <button
                  key={u}
                  type="button"
                  onClick={() => setForm({ username: u, password: p })}
                  style={{
                    background: 'none', border: 'none', textAlign: 'left',
                    cursor: 'pointer', padding: '2px 0', color: 'var(--text-secondary)',
                    fontSize: 12.5, borderRadius: 4
                  }}
                >
                  <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{role}:</span>
                  <span className="mono" style={{ marginLeft: 6 }}>{u} / {p}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
