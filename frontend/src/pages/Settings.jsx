import { useEffect, useState } from 'react';
import { Moon, Sun, Save, User, Shield, Bell } from 'lucide-react';
import client from '../api/client';
import { useNotification } from '../context/NotificationContext';
import { useTheme } from '../context/ThemeContext';
import { useAuth } from '../context/AuthContext';

const MODEL_ROW = {
  gnn: 'GraphSAGE (GNN)',
  lightgbm: 'LightGBM',
  ensemble: 'Ensemble (GNN+LGBM)',
};

export default function Settings() {
  const { notify } = useNotification();
  const { theme, toggle, isDark } = useTheme();
  const { user } = useAuth();
  // fraud_threshold starts null and is filled from the API, which seeds it
  // from the model's validation-tuned value. Hardcoding 0.5 here meant the
  // slider showed 50% before the real setting had loaded.
  const [settings, setSettings] = useState({ fraud_threshold: null, theme: 'dark', notifications_enabled: true, model_name: '' });
  const [models, setModels] = useState([]);
  const [activeModel, setActiveModel] = useState('');
  const [testMetrics, setTestMetrics] = useState(null);
  const [blend, setBlend] = useState(null);
  const [switching, setSwitching] = useState(false);
  const [loading,  setLoading]  = useState(true);
  const [saving,   setSaving]   = useState(false);

  useEffect(() => {
    client.get('/api/settings')
      .then(r => setSettings(r.data))
      .catch(() => notify.error('Failed to load settings'))
      .finally(() => setLoading(false));
  }, [notify]);

  const save = async () => {
    setSaving(true);
    try {
      await client.put('/api/settings', settings);
      notify.success('Settings saved successfully');
    } catch { notify.error('Failed to save settings'); }
    finally { setSaving(false); }
  };

  const Section = ({ icon: Icon, title, children }) => (
    <div className="card" style={{ padding: 28 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 24, paddingBottom: 16, borderBottom: '1px solid var(--border)' }}>
        <div style={{ width: 36, height: 36, borderRadius: 8, background: 'color-mix(in srgb, var(--brand) 10%, transparent)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          <Icon size={18} color="var(--brand)"/>
        </div>
        <div style={{ fontWeight: 700, fontSize: 15 }}>{title}</div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
        {children}
      </div>
    </div>
  );

  const Row = ({ label, sub, children }) => (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16 }}>
      <div>
        <div style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)' }}>{label}</div>
        {sub && <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>}
      </div>
      {children}
    </div>
  );

  useEffect(() => {
    client.get('/api/settings/models')
      .then(r => {
        setModels(r.data.models ?? []);
        setActiveModel(r.data.active ?? '');
        setTestMetrics(r.data.test_metrics ?? null);
        setBlend(r.data.blend ?? null);
      })
      .catch(() => { /* selector just stays empty */ });
  }, []);

  const switchModel = async (id) => {
    if (!id || id === activeModel) return;
    setSwitching(true);
    try {
      const { data } = await client.put('/api/settings/models', { model_id: id });
      setActiveModel(data.active);
      // The threshold follows the model unless a human has overridden it.
      setSettings(s => ({ ...s, model_tuned_threshold: data.tuned_threshold,
                          fraud_threshold: s.is_overridden ? s.fraud_threshold : data.threshold }));
      notify.success(`Now serving ${id} at threshold ${data.threshold}`);
    } catch {
      notify.error('Could not switch model');
    } finally {
      setSwitching(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Settings</h1>
          <p className="page-subtitle">Configure detection thresholds, appearance, and notifications</p>
        </div>
        <button className="btn btn-primary" onClick={save} disabled={saving || loading}
          title={loading ? "Loading current settings…" : undefined}>
          {saving ? <><div className="spinner" style={{ width: 15, height: 15 }}/> Saving…</> : <><Save size={15}/> Save Changes</>}
        </button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Detection Settings */}
        <Section icon={Shield} title="Detection Settings">
          <Row label="Fraud Detection Threshold" sub={`Currently: ${((settings.fraud_threshold ?? 0) * 100).toFixed(0)}% — transactions above this are flagged as fraud`}>
            <div style={{ minWidth: 200 }}>
              <input type="range" min={0.01} max={0.99} step={0.01} value={settings.fraud_threshold ?? 0.5}
                onChange={e => setSettings(s => ({ ...s, fraud_threshold: parseFloat(e.target.value) }))}
                style={{ width: '100%', accentColor: 'var(--brand)' }}/>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                <span>Strict (1%)</span>
                <span style={{ fontWeight: 700, color: 'var(--brand)' }}>{((settings.fraud_threshold ?? 0) * 100).toFixed(0)}%</span>
                <span>Lenient (99%)</span>
              </div>
            </div>
          </Row>

          <div style={{ padding: '14px 16px', borderRadius: 10, background: 'color-mix(in srgb, var(--brand) 6%, transparent)', border: '1px solid color-mix(in srgb, var(--brand) 15%, transparent)' }}>
            <div style={{ fontSize: 12.5, color: 'var(--text-secondary)' }}>
              <strong style={{ color: 'var(--brand)' }}>Model-tuned threshold: {settings.model_tuned_threshold != null ? (settings.model_tuned_threshold * 100).toFixed(0) + '%' : '—'}</strong>
              {' '}— chosen on the validation split to maximise F1, and what the model ships with.
              {settings.is_overridden && <span style={{ color: 'var(--status-warning)' }}> You are currently overriding it.</span>}
              {' '}Lowering the threshold increases recall (catches more fraud, more false positives); raising it increases precision.
              Note that 0.50 is <em>not</em> a neutral default for this model — its probabilities are not centred there.
            </div>
          </div>

          {/* A real selector. The old one offered 'GraphSAGE-v2 (future)' and
              'GCN (future)' and switching it changed nothing. */}
          <Row label="Active model" sub="Which model serves predictions">
            <select className="input" value={activeModel}
              onChange={e => switchModel(e.target.value)}
              disabled={switching || models.length === 0}
              style={{ width: 210 }}>
              {models.map(m => (
                <option key={m.id} value={m.id} disabled={!m.available}>
                  {m.name}{m.available ? '' : ' (unavailable)'}
                </option>
              ))}
            </select>
          </Row>

          {models.length > 0 && (
            <div className="table-wrap" style={{ marginTop: 4 }}>
              <table>
                <thead>
                  <tr>{['Model', 'Threshold', 'F1', 'PR-AUC', ''].map(h => <th key={h}>{h}</th>)}</tr>
                </thead>
                <tbody>
                  {models.map(m => {
                    const t = testMetrics?.[MODEL_ROW[m.id]] ?? null;
                    const isActive = m.id === activeModel;
                    return (
                      <tr key={m.id} style={isActive ? { background: 'var(--brand-wash)' } : undefined}>
                        <td>
                          <div style={{ fontWeight: 600, color: isActive ? 'var(--brand-strong)' : 'var(--text-primary)' }}>
                            {m.name}
                          </div>
                          <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>{m.description}</div>
                        </td>
                        <td className="mono">{m.threshold}</td>
                        <td className="mono">{t ? (t.F1 * 100).toFixed(1) + '%' : '—'}</td>
                        <td className="mono">{t ? (t['PR-AUC'] * 100).toFixed(1) + '%' : '—'}</td>
                        <td>
                          {isActive
                            ? <span className="badge badge-low">active</span>
                            : <button className="btn btn-ghost btn-sm" disabled={switching}
                                onClick={() => switchModel(m.id)}>Use</button>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {blend?.verdict && (
                <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: '12px 16px 4px' }}>
                  Blend is {(blend.weight_gnn * 100).toFixed(0)}% GNN +{' '}
                  {(blend.weight_lgbm * 100).toFixed(0)}% LightGBM, weighted by
                  validation PR-AUC — {blend.verdict}. Each model keeps its own
                  threshold; they are calibrated differently, so one shared
                  cut-off would misreport at least two of them.
                </div>
              )}
            </div>
          )}
        </Section>

        {/* Appearance */}
        <Section icon={Sun} title="Appearance">
          <Row label="Theme" sub="Choose your preferred colour scheme">
            <button
              onClick={() => { toggle(); setSettings(s => ({ ...s, theme: isDark ? 'light' : 'dark' })); }}
              style={{
                display: 'flex', alignItems: 'center', gap: 10,
                padding: '10px 18px', borderRadius: 10, cursor: 'pointer',
                border: '1px solid var(--border)',
                background: isDark ? 'color-mix(in srgb, var(--brand) 10%, transparent)' : 'color-mix(in srgb, var(--status-warning) 10%, transparent)',
                color: isDark ? 'var(--brand)' : 'var(--status-warning)', fontWeight: 600, fontSize: 13.5
              }}>
              {isDark ? <Moon size={16}/> : <Sun size={16}/>}
              {isDark ? 'Dark Mode' : 'Light Mode'}
            </button>
          </Row>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            {[['Dark', 'dark', 'var(--bg-primary)', 'var(--bg-card)'], ['Light', 'light', 'var(--bg-card)', 'var(--bg-card)']].map(([l, t, bg, card]) => (
              <div key={t} onClick={() => { if (theme !== t) toggle(); }}
                style={{
                  padding: 14, borderRadius: 10, cursor: 'pointer',
                  border: `2px solid ${theme === t ? 'var(--brand)' : 'var(--border)'}`,
                  background: 'var(--bg-secondary)', transition: 'var(--transition)'
                }}>
                <div style={{ background: bg, borderRadius: 6, padding: 8, marginBottom: 8 }}>
                  <div style={{ width: '60%', height: 6, background: card, borderRadius: 3, marginBottom: 4 }}/>
                  <div style={{ width: '40%', height: 6, background: card, borderRadius: 3, opacity: 0.6 }}/>
                </div>
                <div style={{ fontSize: 12.5, fontWeight: 600, color: 'var(--text-primary)' }}>
                  {l} {theme === t && '✓'}
                </div>
              </div>
            ))}
          </div>
        </Section>

        {/* Notifications */}
        <Section icon={Bell} title="Notifications">
          {[
            ['Enable Notifications',     'Show toast alerts for fraud detections and system events', 'notifications_enabled'],
          ].map(([label, sub, key]) => (
            <Row key={key} label={label} sub={sub}>
              <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
                <div
                  onClick={() => setSettings(s => ({ ...s, [key]: !s[key] }))}
                  style={{
                    width: 44, height: 24, borderRadius: 12,
                    background: settings[key] ? 'var(--brand)' : 'var(--border-light)',
                    position: 'relative', cursor: 'pointer', transition: 'var(--transition)'
                  }}>
                  <div style={{
                    width: 18, height: 18, borderRadius: '50%', background: 'white',
                    position: 'absolute', top: 3,
                    left: settings[key] ? 23 : 3,
                    transition: 'var(--transition)',
                    boxShadow: '0 1px 4px color-mix(in srgb, var(--text-primary) 20%, transparent)'
                  }}/>
                </div>
              </label>
            </Row>
          ))}
        </Section>

        {/* Account */}
        <Section icon={User} title="Account">
          {[['Username', user?.username], ['Email', user?.email], ['Role', user?.role === 'admin' ? '⭐ Administrator' : '👤 Analyst']].map(([l, v]) => (
            <Row key={l} label={l} sub={null}>
              <span style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)' }}>{v}</span>
            </Row>
          ))}
          <div style={{ padding: '12px 16px', borderRadius: 10, background: 'color-mix(in srgb, var(--status-good) 6%, transparent)', border: '1px solid color-mix(in srgb, var(--status-good) 15%, transparent)' }}>
            <div style={{ fontSize: 12.5, color: 'var(--status-good)' }}>
              ✅ Session active · JWT authentication · Token expires in 8 hours
            </div>
          </div>
        </Section>
      </div>
    </div>
  );
}
