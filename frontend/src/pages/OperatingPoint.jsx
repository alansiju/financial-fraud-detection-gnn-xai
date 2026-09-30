import { useEffect, useMemo, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, ReferenceLine,
} from 'recharts';
import { Target, Coins, RotateCcw } from 'lucide-react';
import client from '../api/client';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import { useNotification } from '../context/NotificationContext';

/*
 * The operating point is the decision this whole system reduces to: at what
 * score do we stop the transaction. Everything else — the model, the features,
 * the graph — only matters through where that line sits.
 *
 * Best-F1 is the usual default and it quietly assumes a missed fraud and a
 * false alarm cost the same. They never do. This page lets the real numbers
 * be entered and shows what they imply.
 */

const money = (n) => '£' + Math.round(n).toLocaleString();

export default function OperatingPoint() {
  const [sweep, setSweep] = useState(null);
  const [loading, setLoading] = useState(true);
  const [threshold, setThreshold] = useState(null);
  const [costFN, setCostFN] = useState(150);   // £ lost per missed fraud
  const [costFP, setCostFP] = useState(8);     // £ of review time per false alarm
  const { notify } = useNotification();

  useEffect(() => {
    client.get('/api/insights/threshold-sweep', { params: { steps: 197 } })
      .then(r => { setSweep(r.data); setThreshold(r.data.current_threshold); })
      .catch(() => notify.error('Failed to load the threshold sweep'))
      .finally(() => setLoading(false));
  }, [notify]);

  // Memoised: `?? []` would mint a new array every render and invalidate
  // every useMemo below it.
  const points = useMemo(() => sweep?.points ?? [], [sweep]);

  // Cost at every threshold, so the optimum can be found rather than guessed.
  const costed = useMemo(() => points.map(p => ({
    ...p,
    cost: p.fn * costFN + p.fp * costFP,
  })), [points, costFN, costFP]);

  const cheapest = useMemo(
    () => costed.reduce((a, b) => (b.cost < a.cost ? b : a), costed[0]),
    [costed]);

  const bestF1 = useMemo(
    () => points.reduce((a, b) => (b.f1 > a.f1 ? b : a), points[0]),
    [points]);

  const current = useMemo(() => {
    if (!costed.length || threshold == null) return null;
    return costed.reduce((a, b) =>
      Math.abs(b.threshold - threshold) < Math.abs(a.threshold - threshold) ? b : a);
  }, [costed, threshold]);

  if (loading) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header"><h1 className="page-title">Operating Point</h1></div>
      <SkeletonLoader type="kpi"/>
      <SkeletonLoader type="chart"/>
    </div>
  );

  if (!sweep || !current) return <div className="card">No sweep data available.</div>;

  const savings = current.cost - cheapest.cost;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="page-header">
        <div>
          <div className="eyebrow">Decision</div>
          <h1 className="page-title">Operating Point</h1>
          <p className="page-subtitle">
            Every model decision reduces to one number: the score above which a
            transaction is stopped. {sweep.n_test.toLocaleString()} held-out
            transactions, {sweep.n_fraud.toLocaleString()} of them fraud
            ({(sweep.base_rate * 100).toFixed(2)}%).
          </p>
        </div>
        <button className="btn btn-ghost btn-sm"
          onClick={() => setThreshold(sweep.model_tuned_threshold)}>
          <RotateCcw size={14}/> Reset to tuned ({sweep.model_tuned_threshold})
        </button>
      </div>

      {/* ── Slider + live confusion matrix ─────────────────────────────── */}
      <div className="card" style={{ padding: 26 }}>
        <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 20, flexWrap: 'wrap' }}>
          <div>
            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              Threshold
            </div>
            <div className="stat-hero" style={{ color: 'var(--brand)' }}>
              {current.threshold.toFixed(2)}
            </div>
          </div>
          <div style={{ display: 'flex', gap: 28, flexWrap: 'wrap' }}>
            {[
              ['Precision', current.precision, 'of what we stop, this much is fraud'],
              ['Recall', current.recall, 'of all fraud, we stop this much'],
              ['F1', current.f1, 'balance of the two'],
            ].map(([l, v, sub]) => (
              <div key={l}>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>{l}</div>
                <div className="kpi-value" style={{ fontSize: 26 }}>{(v * 100).toFixed(1)}%</div>
                <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>
              </div>
            ))}
          </div>
        </div>

        <input type="range" min={0.01} max={0.99} step={0.005}
          value={threshold}
          onChange={e => setThreshold(parseFloat(e.target.value))}
          style={{ width: '100%', marginTop: 22, accentColor: 'var(--brand)' }}/>
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11.5, color: 'var(--text-muted)', marginTop: 4 }}>
          <span>0.01 — stop almost everything</span>
          <span>0.99 — stop almost nothing</span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 12, marginTop: 24 }}>
          {[
            ['Caught', current.tp, 'var(--status-good)', 'fraud stopped'],
            ['Missed', current.fn, 'var(--status-critical)', 'fraud that got through'],
            ['False alarms', current.fp, 'var(--status-warning)', 'legitimate, stopped anyway'],
            ['Cleared', current.tn, 'var(--text-muted)', 'legitimate, passed'],
          ].map(([label, v, color, sub]) => (
            <div key={label} style={{
              background: `color-mix(in srgb, ${color} 8%, transparent)`,
              border: `1px solid color-mix(in srgb, ${color} 20%, transparent)`,
              borderRadius: 'var(--radius-md)', padding: '14px 16px',
            }}>
              <div style={{ height: 3, width: 26, borderRadius: 3, background: color, marginBottom: 10 }}/>
              <div className="kpi-value" style={{ fontSize: 24 }}>{v.toLocaleString()}</div>
              <div style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-primary)', marginTop: 3 }}>{label}</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>{sub}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Cost model ─────────────────────────────────────────────────── */}
      <div className="card" style={{ padding: 26 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <Coins size={18} color="var(--brand)"/>
          <div className="chart-title">What it costs</div>
        </div>
        <div className="chart-subtitle" style={{ marginBottom: 20 }}>
          Best-F1 treats a missed fraud and a false alarm as equally bad. Put
          your real numbers in and the optimum usually moves.
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: 20 }}>
          <div className="form-group">
            <label className="form-label">Cost of a missed fraud (£)</label>
            <input className="input" type="number" min={0} value={costFN}
              onChange={e => setCostFN(Math.max(0, parseFloat(e.target.value) || 0))}/>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
              Typically the transaction value plus chargeback handling.
            </div>
          </div>
          <div className="form-group">
            <label className="form-label">Cost of a false alarm (£)</label>
            <input className="input" type="number" min={0} value={costFP}
              onChange={e => setCostFP(Math.max(0, parseFloat(e.target.value) || 0))}/>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
              Analyst review time, plus the cost of annoying a real customer.
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: 14, marginTop: 22 }}>
          <div style={{ background: 'var(--bg-tint)', borderRadius: 'var(--radius-md)', padding: '16px 18px' }}>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>At {current.threshold.toFixed(2)}</div>
            <div className="kpi-value" style={{ fontSize: 26 }}>{money(current.cost)}</div>
          </div>
          <div style={{ background: 'var(--brand-wash)', border: '1px solid var(--brand-wash-alt)', borderRadius: 'var(--radius-md)', padding: '16px 18px' }}>
            <div style={{ fontSize: 11.5, color: 'var(--brand-strong)', textTransform: 'uppercase', letterSpacing: '0.07em', fontWeight: 700 }}>
              Cheapest at {cheapest.threshold.toFixed(2)}
            </div>
            <div className="kpi-value" style={{ fontSize: 26 }}>{money(cheapest.cost)}</div>
            <button className="btn btn-soft btn-sm" style={{ marginTop: 10 }}
              onClick={() => setThreshold(cheapest.threshold)}>
              <Target size={13}/> Move here
            </button>
          </div>
          <div style={{ background: 'var(--bg-tint)', borderRadius: 'var(--radius-md)', padding: '16px 18px' }}>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Difference</div>
            <div className="kpi-value" style={{ fontSize: 26, color: savings > 0 ? 'var(--status-good)' : 'var(--text-muted)' }}>
              {savings > 0 ? '−' + money(savings) : '—'}
            </div>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 2 }}>
              {savings > 0 ? 'saved by moving there' : 'already at the optimum'}
            </div>
          </div>
          <div style={{ background: 'var(--bg-tint)', borderRadius: 'var(--radius-md)', padding: '16px 18px' }}>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em' }}>Best-F1 sits at</div>
            <div className="kpi-value" style={{ fontSize: 26 }}>{bestF1.threshold.toFixed(2)}</div>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 2 }}>
              costs {money(bestF1.fn * costFN + bestF1.fp * costFP)}
            </div>
          </div>
        </div>
      </div>

      {/* ── Curves ─────────────────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div className="chart-container">
          <div className="chart-title">Precision and recall across the range</div>
          <div className="chart-subtitle">
            They trade against each other; the threshold decides the exchange rate
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={points} margin={{ top: 8, right: 10, left: -12, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 5" stroke="var(--border)"/>
              <XAxis dataKey="threshold" type="number" domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]} tick={{ fontSize: 10 }}
                tickFormatter={v => v.toFixed(2)}/>
              <YAxis domain={[0, 1]} tick={{ fontSize: 10 }}
                tickFormatter={v => `${(v * 100).toFixed(0)}%`}/>
              <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: 10 }}
                formatter={(v, n) => [`${(v * 100).toFixed(1)}%`, n]}
                labelFormatter={v => `threshold ${Number(v).toFixed(2)}`}/>
              <Legend wrapperStyle={{ fontSize: 12 }}/>
              <ReferenceLine x={current.threshold} stroke="var(--brand)" strokeWidth={2}/>
              <Line type="monotone" dataKey="precision" name="Precision" stroke="var(--cat-2)" dot={false} strokeWidth={2}/>
              <Line type="monotone" dataKey="recall"    name="Recall"    stroke="var(--cat-3)" dot={false} strokeWidth={2}/>
              <Line type="monotone" dataKey="f1"        name="F1"        stroke="var(--cat-1)" dot={false} strokeWidth={2}/>
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <div className="chart-title">Expected cost</div>
          <div className="chart-subtitle">
            {money(costFN)} per missed fraud · {money(costFP)} per false alarm
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={costed} margin={{ top: 8, right: 10, left: 4, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 5" stroke="var(--border)"/>
              <XAxis dataKey="threshold" type="number" domain={[0, 1]}
                ticks={[0, 0.25, 0.5, 0.75, 1]} tick={{ fontSize: 10 }}
                tickFormatter={v => v.toFixed(2)}/>
              <YAxis tick={{ fontSize: 10 }} width={64}
                tickFormatter={v => v >= 1000 ? `£${(v / 1000).toFixed(0)}k` : `£${v}`}/>
              <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: 10 }}
                formatter={(v) => [money(v), 'Expected cost']}
                labelFormatter={v => `threshold ${Number(v).toFixed(2)}`}/>
              <ReferenceLine x={cheapest.threshold} stroke="var(--status-good)"
                strokeDasharray="4 4"
                label={{ value: 'cheapest', position: 'top', fontSize: 10, fill: 'var(--status-good)' }}/>
              <ReferenceLine x={current.threshold} stroke="var(--brand)" strokeWidth={2}/>
              {/* One series, so the title names it and no legend is needed. */}
              <Line type="monotone" dataKey="cost" stroke="var(--status-critical)" dot={false} strokeWidth={2}/>
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
