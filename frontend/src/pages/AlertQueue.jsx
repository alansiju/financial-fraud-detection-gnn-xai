import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ShieldAlert, Check, X, ArrowUpRight, Undo2, RefreshCw, Keyboard,
} from 'lucide-react';
import client from '../api/client';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import { useNotification } from '../context/NotificationContext';
import { riskClass } from '../lib/risk';

/*
 * A worklist, not a report.
 *
 * The queue is ordered by score rather than by the flag, so changing the
 * threshold never reshuffles the top — an analyst working down from the
 * riskiest item keeps a stable order.
 *
 * Every decision is written back with the score, the threshold and the
 * ground-truth label attached, which is what lets "observed precision" be
 * compared against the precision the test split promised.
 */


export default function AlertQueue() {
  const navigate = useNavigate();
  const { notify } = useNotification();
  const [queue, setQueue] = useState(null);
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [cursor, setCursor] = useState(0);
  const [busy, setBusy] = useState(false);
  const [lastAction, setLastAction] = useState(null);

  const load = useCallback(async () => {
    try {
      const [q, s] = await Promise.all([
        client.get('/api/alerts/queue', { params: { limit: 100 } }),
        client.get('/api/alerts/stats'),
      ]);
      setQueue(q.data);
      setStats(s.data);
      setCursor(c => Math.min(c, Math.max(0, (q.data.items?.length ?? 1) - 1)));
    } catch {
      notify.error('Failed to load the review queue');
    } finally {
      setLoading(false);
    }
  }, [notify]);

  useEffect(() => { load(); }, [load]);

  const items = queue?.items ?? [];
  const active = items[cursor];

  const setItems_ = useCallback(
    (fn) => setQueue(q => (q ? { ...q, items: fn(q.items) } : q)), []);

  const decide = useCallback(async (disposition) => {
    if (!active || busy) return;
    setBusy(true);
    try {
      const { data } = await client.post(
        `/api/alerts/${active.node_id}/disposition`, { disposition });
      setLastAction({ node_id: active.node_id, tx: active.transaction_id });
      // Drop the reviewed row locally so the next item is immediately ready;
      // refetching the whole queue on every keystroke would make the shortcuts
      // feel laggy.
      setItems_(prev => prev.filter(i => i.node_id !== active.node_id));
      setStats(s => s ? {
        ...s,
        total_reviewed: (s.total_reviewed ?? 0) + 1,
        confirmed_fraud: (s.confirmed_fraud ?? 0) + (disposition === 'confirmed_fraud' ? 1 : 0),
        dismissed: (s.dismissed ?? 0) + (disposition === 'dismissed' ? 1 : 0),
        escalated: (s.escalated ?? 0) + (disposition === 'escalated' ? 1 : 0),
      } : s);
      notify.success(
        data.agrees_with_label
          ? `Recorded — matches the labelled outcome`
          : `Recorded — the label says otherwise`);
    } catch {
      notify.error('Could not record that decision');
    } finally {
      setBusy(false);
    }
  }, [active, busy, notify, setItems_]);

  const undo = useCallback(async () => {
    if (!lastAction) return;
    try {
      await client.delete(`/api/alerts/${lastAction.node_id}/disposition`);
      notify.info(`Returned ${lastAction.tx} to the queue`);
      setLastAction(null);
      load();
    } catch {
      notify.error('Could not undo');
    }
  }, [lastAction, load, notify]);

  // Keyboard-first: a triage queue that needs the mouse is a queue nobody works.
  useEffect(() => {
    const onKey = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      const k = e.key.toLowerCase();
      if (k === 'f') { e.preventDefault(); decide('confirmed_fraud'); }
      else if (k === 'd') { e.preventDefault(); decide('dismissed'); }
      else if (k === 'e') { e.preventDefault(); decide('escalated'); }
      else if (k === 'j' || e.key === 'ArrowDown') { e.preventDefault(); setCursor(c => Math.min(c + 1, items.length - 1)); }
      else if (k === 'k' || e.key === 'ArrowUp') { e.preventDefault(); setCursor(c => Math.max(c - 1, 0)); }
      else if (k === 'u') { e.preventDefault(); undo(); }
      else if (e.key === 'Enter' && active) { e.preventDefault(); navigate(`/transactions/${active.node_id}`); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [decide, undo, items.length, active, navigate]);

  if (loading) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header"><h1 className="page-title">Review Queue</h1></div>
      <SkeletonLoader type="kpi"/>
      <SkeletonLoader type="chart"/>
    </div>
  );

  const observed = stats?.observed_precision;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="page-header">
        <div>
          <div className="eyebrow">Triage</div>
          <h1 className="page-title">Review Queue</h1>
          <p className="page-subtitle">
            {queue?.total_flagged?.toLocaleString()} transactions flagged by the{' '}
            <strong>{queue?.model}</strong> model at threshold {queue?.threshold},
            ranked by score. Decisions are recorded against the ground-truth label.
          </p>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={load}>
          <RefreshCw size={14}/> Refresh
        </button>
      </div>

      {/* ── Throughput ────────────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 14 }}>
        {[
          ['Flagged', queue?.total_flagged?.toLocaleString() ?? '—', 'awaiting or reviewed'],
          ['Reviewed', stats?.total_reviewed?.toLocaleString() ?? '0', 'decisions recorded'],
          ['Confirmed fraud', stats?.confirmed_fraud ?? 0, 'analyst agreed'],
          ['Dismissed', stats?.dismissed ?? 0, 'analyst cleared'],
          [
            'Observed precision',
            observed != null ? `${(observed * 100).toFixed(1)}%` : '—',
            observed != null ? 'what analysts actually saw' : 'review some alerts first',
          ],
        ].map(([label, value, sub]) => (
          <div key={label} className="kpi-card" style={{ padding: 18 }}>
            <div className="kpi-value" style={{ fontSize: 24 }}>{value}</div>
            <div className="kpi-label">{label}</div>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 4 }}>{sub}</div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1.25fr 1fr', gap: 20, alignItems: 'start' }}>
        {/* ── The list ────────────────────────────────────────────────── */}
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {['', 'Transaction', 'Amount', 'Score', 'Risk', 'Card', 'Device'].map(h =>
                  <th key={h}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {items.length === 0 && (
                <tr><td colSpan={7} style={{ textAlign: 'center', padding: 36, color: 'var(--text-muted)' }}>
                  Queue empty — every flagged transaction has been reviewed.
                </td></tr>
              )}
              {items.map((it, i) => (
                <tr key={it.node_id}
                  onClick={() => setCursor(i)}
                  style={{
                    cursor: 'pointer',
                    background: i === cursor ? 'var(--brand-wash)' : undefined,
                  }}>
                  <td style={{ width: 8, padding: 0 }}>
                    {i === cursor && <div style={{ width: 3, height: 30, background: 'var(--brand)', borderRadius: 3 }}/>}
                  </td>
                  <td className="mono" style={{ fontWeight: 600 }}>{it.transaction_id}</td>
                  <td className="mono">£{it.amount?.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                  <td className="mono">{(it.fraud_probability * 100).toFixed(1)}%</td>
                  <td><span className={`badge ${riskClass(it.risk_level)}`}>{it.risk_level}</span></td>
                  <td>{it.card_type}</td>
                  <td>{it.device}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* ── The active alert ────────────────────────────────────────── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16, position: 'sticky', top: 88 }}>
          {active ? (
            <div className="card" style={{ padding: 24 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 16 }}>
                <ShieldAlert size={18} color="var(--status-critical)"/>
                <span className={`badge ${riskClass(active.risk_level)}`}>
                  {active.risk_level}
                </span>
                <span style={{ marginLeft: 'auto', fontSize: 12, color: 'var(--text-muted)' }}>
                  {cursor + 1} of {items.length}
                </span>
              </div>

              <div className="stat-hero" style={{ fontSize: 42, color: 'var(--status-critical)' }}>
                {(active.fraud_probability * 100).toFixed(1)}%
              </div>
              <div style={{ fontSize: 12.5, color: 'var(--text-muted)', marginBottom: 18 }}>
                fraud probability · threshold {queue.threshold}
              </div>

              {[
                ['Transaction', active.transaction_id],
                ['Amount', `£${active.amount?.toLocaleString(undefined, { minimumFractionDigits: 2 })}`],
                ['Card', active.card_type],
                ['Device', active.device],
                ['Email domain', active.email_domain],
                ['Time', active.timestamp?.replace('T', ' ').replace('Z', '') ?? '—'],
              ].map(([k, v]) => (
                <div key={k} style={{ display: 'flex', justifyContent: 'space-between', gap: 12, padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
                  <span style={{ fontSize: 12.5, color: 'var(--text-muted)' }}>{k}</span>
                  <span className="mono" style={{ fontSize: 12.5, fontWeight: 600 }}>{v}</span>
                </div>
              ))}

              <div style={{ display: 'flex', gap: 8, marginTop: 20, flexWrap: 'wrap' }}>
                <button className="btn btn-danger" disabled={busy}
                  onClick={() => decide('confirmed_fraud')}>
                  <Check size={14}/> Confirm fraud
                </button>
                <button className="btn btn-success" disabled={busy}
                  onClick={() => decide('dismissed')}>
                  <X size={14}/> Dismiss
                </button>
                <button className="btn btn-ghost" disabled={busy}
                  onClick={() => decide('escalated')}>
                  Escalate
                </button>
              </div>

              <div style={{ display: 'flex', gap: 8, marginTop: 10 }}>
                <button className="btn btn-ghost btn-sm"
                  onClick={() => navigate(`/transactions/${active.node_id}`)}>
                  <ArrowUpRight size={13}/> Full detail
                </button>
                <button className="btn btn-ghost btn-sm"
                  onClick={() => navigate(`/explain/${active.node_id}`)}>
                  Why flagged?
                </button>
                {lastAction && (
                  <button className="btn btn-ghost btn-sm" onClick={undo}>
                    <Undo2 size={13}/> Undo
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="card" style={{ padding: 32, textAlign: 'center', color: 'var(--text-muted)' }}>
              Nothing left to review.
            </div>
          )}

          <div className="card" style={{ padding: 18 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
              <Keyboard size={15} color="var(--text-muted)"/>
              <div style={{ fontSize: 12.5, fontWeight: 700 }}>Shortcuts</div>
            </div>
            {[
              ['F', 'confirm fraud'], ['D', 'dismiss'], ['E', 'escalate'],
              ['J / K', 'move down / up'], ['U', 'undo last'], ['Enter', 'open detail'],
            ].map(([k, v]) => (
              <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', fontSize: 12.5 }}>
                <kbd className="mono" style={{
                  background: 'var(--bg-tint)', border: '1px solid var(--border-light)',
                  borderRadius: 5, padding: '1px 7px', fontSize: 11.5,
                }}>{k}</kbd>
                <span style={{ color: 'var(--text-muted)' }}>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
