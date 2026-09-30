import { useEffect, useState } from 'react';
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  CartesianGrid, Legend, ReferenceLine, Cell,
} from 'recharts';
import { Layers, AlertCircle } from 'lucide-react';
import client from '../api/client';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import { useNotification } from '../context/NotificationContext';

/*
 * Two questions a deployed model has to keep answering:
 *   "is the world still the one I was trained on?"   -> drift
 *   "what have I actually learned?"                  -> global importance
 *
 * GNNExplainer answers "why this transaction". It cannot answer the second
 * question, because a per-node explanation says nothing about the model as a
 * whole. Permutation importance does.
 */

const BAND_COLOR = {
  stable: 'var(--status-good)',
  moderate: 'var(--status-warning)',
  significant: 'var(--status-critical)',
};

export default function ModelHealth() {
  const [drift, setDrift] = useState(null);
  const [imp, setImp] = useState(null);
  const [loading, setLoading] = useState(true);
  const [impLoading, setImpLoading] = useState(true);
  const { notify } = useNotification();

  useEffect(() => {
    client.get('/api/insights/drift')
      .then(r => setDrift(r.data))
      .catch(() => notify.error('Failed to load drift data'))
      .finally(() => setLoading(false));

    // Permutation importance is a few hundred forward passes, so it is fetched
    // separately rather than holding the whole page behind it.
    client.get('/api/insights/global-importance', { params: { top_k: 18 }, timeout: 300000 })
      .then(r => setImp(r.data))
      .catch(() => notify.error('Failed to compute feature importance'))
      .finally(() => setImpLoading(false));
  }, [notify]);

  if (loading) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header"><h1 className="page-title">Model Health</h1></div>
      <SkeletonLoader type="kpi"/>
      <SkeletonLoader type="chart"/>
    </div>
  );

  if (!drift) return <div className="card">No drift data available.</div>;

  // Overlay the three split histograms on one axis.
  // Floor the shares: an empty bin is 0, and log(0) has no place on an axis.
  const floorPct = (v) => Math.max((v ?? 0) * 100, 0.01);
  const bins = drift.splits.train.histogram.map((b, i) => ({
    bin: b.bin,
    train: floorPct(b.share),
    val: floorPct(drift.splits.val.histogram[i]?.share),
    test: floorPct(drift.splits.test.histogram[i]?.share),
  }));

  const psiTest = drift.psi.train_vs_test;
  const psiVal = drift.psi.train_vs_val;
  const anyDrift = psiTest.band !== 'stable' || psiVal.band !== 'stable';

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="page-header">
        <div>
          <div className="eyebrow">Monitoring</div>
          <h1 className="page-title">Model Health</h1>
          <p className="page-subtitle">
            Whether the score distribution still looks like the one the model was
            trained on, and which features are actually driving it.
          </p>
        </div>
      </div>

      {/* ── PSI ───────────────────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: 14 }}>
        {[
          ['Train → validation', psiVal],
          ['Train → test', psiTest],
        ].map(([label, p]) => (
          <div key={label} className="kpi-card" style={{ padding: 20 }}>
            <div style={{ height: 3, width: 30, borderRadius: 3, background: BAND_COLOR[p.band], marginBottom: 12 }}/>
            <div className="kpi-value" style={{ fontSize: 28 }}>{p.value.toFixed(4)}</div>
            <div className="kpi-label">{label} PSI</div>
            <div style={{ fontSize: 12, color: BAND_COLOR[p.band], fontWeight: 600, marginTop: 6, textTransform: 'capitalize' }}>
              {p.band}
            </div>
          </div>
        ))}
        {['train', 'val', 'test'].map(k => (
          <div key={k} className="kpi-card" style={{ padding: 20 }}>
            <div className="kpi-value" style={{ fontSize: 28 }}>
              {(drift.splits[k].fraud_rate * 100).toFixed(2)}%
            </div>
            <div className="kpi-label">{k} fraud rate</div>
            <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 5 }}>
              {drift.splits[k].n.toLocaleString()} rows · mean score{' '}
              {drift.splits[k].mean_score.toFixed(3)}
            </div>
          </div>
        ))}
      </div>

      {/* Say what the numbers mean, including when they say "nothing wrong". */}
      <div className="card" style={{
        padding: '16px 20px', display: 'flex', gap: 12, alignItems: 'flex-start',
        background: anyDrift ? 'var(--status-warning-bg)' : 'var(--status-good-bg)',
        border: `1px solid ${anyDrift ? 'var(--status-warning-br)' : 'var(--status-good-br)'}`,
      }}>
        <AlertCircle size={17} style={{ flexShrink: 0, marginTop: 1 }}
          color={anyDrift ? 'var(--status-warning)' : 'var(--status-good)'}/>
        <div style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.65 }}>
          {anyDrift ? (
            <>The score distribution has shifted between splits. {drift.note}</>
          ) : (
            <>
              <strong style={{ color: 'var(--text-primary)' }}>
                The score distribution is stable across all three splits.
              </strong>{' '}
              That is worth stating plainly, because performance still drops from
              validation to test on this {drift.split_mode} split. Since the
              scores themselves are not drifting, the gap is a change in the
              relationship between features and the label over time, not a shift
              in what the model outputs — so recalibrating the threshold would
              not fix it. {drift.note}
            </>
          )}
        </div>
      </div>

      {/* ── Distributions + timeline ──────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div className="chart-container">
          <div className="chart-title">Score distribution by split</div>
          <div className="chart-subtitle">
            Share of each split falling in each score band, on a log axis.
            Overlapping curves mean the model scores the three periods alike.
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={bins} margin={{ top: 8, right: 10, left: -10, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 5" stroke="var(--border)" vertical={false}/>
              <XAxis dataKey="bin" tick={{ fontSize: 10 }}
                tickFormatter={v => Number(v).toFixed(2)}/>
              {/* Log scale: ~70% of transactions land in the lowest bin, and on a
                  linear axis that flattens the entire tail — which is exactly
                  the region where flagged transactions live. */}
              <YAxis scale="log" domain={[0.01, 100]} allowDataOverflow
                ticks={[0.01, 0.1, 1, 10, 100]} tick={{ fontSize: 10 }}
                tickFormatter={v => v >= 1 ? `${v}%` : `${v}%`} width={46}/>
              <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: 10 }}
                formatter={(v, n) => [`${Number(v).toFixed(2)}%`, n]}
                labelFormatter={v => `score ≈ ${Number(v).toFixed(2)}`}/>
              <Legend wrapperStyle={{ fontSize: 12 }}/>
              <Area type="monotone" dataKey="train" name="Train" stroke="var(--cat-1)" fill="var(--cat-1)" fillOpacity={0.12} strokeWidth={2}/>
              <Area type="monotone" dataKey="val"   name="Validation" stroke="var(--cat-2)" fill="var(--cat-2)" fillOpacity={0.12} strokeWidth={2}/>
              <Area type="monotone" dataKey="test"  name="Test" stroke="var(--cat-3)" fill="var(--cat-3)" fillOpacity={0.12} strokeWidth={2}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <div className="chart-title">Fraud rate over time</div>
          <div className="chart-subtitle">
            Coloured by which split each month landed in
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={drift.timeline} margin={{ top: 8, right: 10, left: -10, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 5" stroke="var(--border)" vertical={false}/>
              <XAxis dataKey="month" tick={{ fontSize: 10 }}/>
              <YAxis tick={{ fontSize: 10 }} tickFormatter={v => `${v}%`} width={42}/>
              <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: 10 }}
                formatter={(v, _n, item) => [
                  `${Number(v).toFixed(2)}%  (${item.payload.n.toLocaleString()} rows, ${item.payload.split})`,
                  'Fraud rate']}/>
              <Bar dataKey="fraud_rate" name="Fraud rate" radius={[5, 5, 0, 0]} maxBarSize={44}>
                {drift.timeline.map((d, i) => (
                  <Cell key={i} fill={
                    d.split === 'train' ? 'var(--cat-1)'
                      : d.split === 'val' ? 'var(--cat-2)' : 'var(--cat-3)'}/>
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="chart-legend">
            {[['Train', 'var(--cat-1)'], ['Validation', 'var(--cat-2)'], ['Test', 'var(--cat-3)']].map(([l, c]) => (
              <span key={l} className="chart-legend-item">
                <span className="chart-legend-swatch" style={{ background: c }}/>{l}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* ── Global importance ─────────────────────────────────────────── */}
      <div className="chart-container">
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <Layers size={17} color="var(--brand)"/>
          <div className="chart-title">What the model has learned</div>
        </div>
        <div className="chart-subtitle">
          {imp
            ? <>Permutation importance on {imp.n_sampled.toLocaleString()} held-out
               transactions: how far average precision falls when one feature is
               shuffled. Baseline PR-AUC {imp.baseline_pr_auc.toFixed(4)}.</>
            : 'Shuffling each feature and re-scoring — this takes a moment…'}
        </div>

        {impLoading ? (
          <div className="skeleton" style={{ height: 420, marginTop: 12 }}/>
        ) : imp ? (
          <>
            <ResponsiveContainer width="100%" height={Math.max(380, imp.features.length * 26)}>
              <BarChart data={imp.features} layout="vertical"
                margin={{ top: 4, right: 40, left: 10, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 5" stroke="var(--border)" horizontal={false}/>
                <XAxis type="number" tick={{ fontSize: 10 }} tickLine={false} axisLine={false}
                  tickFormatter={v => v.toFixed(3)}/>
                <YAxis type="category" dataKey="feature" width={190}
                  tick={{ fontSize: 11 }} tickLine={false} axisLine={false}/>
                <Tooltip contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: 10 }}
                  formatter={(v, _n, item) => [
                    `${Number(v).toFixed(4)}  (${item.payload.importance_pct.toFixed(1)}% of baseline)`,
                    'PR-AUC drop']}
                  cursor={{ fill: 'var(--bg-tint)' }}/>
                <ReferenceLine x={0} stroke="var(--border-light)"/>
                {/* One measure across features; the names are on the axis, so a
                    single hue is correct and a rainbow would encode nothing. */}
                <Bar dataKey="importance" name="PR-AUC drop" fill="var(--cat-1)"
                  radius={[0, 5, 5, 0]} maxBarSize={18}/>
              </BarChart>
            </ResponsiveContainer>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 10 }}>
              {imp.note} A negative bar means shuffling that feature slightly
              <em> helped</em> — noise at this sample size, not a signal.
            </div>
          </>
        ) : (
          <div style={{ padding: 30, textAlign: 'center', color: 'var(--text-muted)' }}>
            Importance unavailable.
          </div>
        )}
      </div>
    </div>
  );
}
