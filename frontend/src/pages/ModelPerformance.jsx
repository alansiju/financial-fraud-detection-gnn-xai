import { useEffect, useState } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, ReferenceLine
} from 'recharts';
import client from '../api/client';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import { useNotification } from '../context/NotificationContext';

// The value is set in ink, not in the metric's colour: hue on small type is
// both a contrast problem (the 16px "%" measured 4.44:1) and against the rule
// that text wears text tokens while a coloured mark beside it carries identity.
const Metric = ({ label, value, color, sub }) => (
  <div className="kpi-card" style={{ '--accent-color': color, paddingTop: 18 }}>
    <div style={{ height: 3, width: 34, borderRadius: 3, background: color, marginBottom: 12 }}/>
    <div style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 6 }}>{label}</div>
    <div className="kpi-value" style={{ fontSize: 30 }}>
      {(value * 100).toFixed(1)}<span style={{ fontSize: 16, color: 'var(--text-muted)' }}>%</span>
    </div>
    {sub && <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 6 }}>{sub}</div>}
  </div>
);

function ConfusionMatrix({ cm }) {
  const { tn, fp, fn, tp } = cm;
  const total = tn + fp + fn + tp;
  const cells = [
    { label: 'True Negative', v: tn, color: 'var(--status-good)', bg: 'color-mix(in srgb, var(--status-good) 10%, transparent)', desc: 'Correctly predicted Legit' },
    { label: 'False Positive', v: fp, color: 'var(--status-warning)', bg: 'color-mix(in srgb, var(--status-warning) 10%, transparent)', desc: 'Legit predicted as Fraud' },
    { label: 'False Negative', v: fn, color: 'var(--status-critical)', bg: 'color-mix(in srgb, var(--status-critical) 10%, transparent)', desc: 'Fraud missed' },
    { label: 'True Positive', v: tp, color: 'var(--brand)', bg: 'color-mix(in srgb, var(--brand) 10%, transparent)', desc: 'Correctly predicted Fraud' },
  ];
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 3, marginBottom: 12 }}>
        {cells.map(c => (
          <div key={c.label} style={{ background: c.bg, border: `1px solid color-mix(in srgb, ${c.color} 20%, transparent)`, borderRadius: 10, padding: '18px 16px', textAlign: 'center' }}>
            <div style={{ fontSize: 26, fontWeight: 800, color: c.color }}>{c.v?.toLocaleString()}</div>
            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginTop: 4 }}>{c.label}</div>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 3 }}>{c.desc}</div>
            <div style={{ fontSize: 11, color: c.color, marginTop: 4, fontWeight: 600 }}>{((c.v / total) * 100).toFixed(2)}%</div>
          </div>
        ))}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
        {[
          ['Sensitivity (TPR)', tp / (tp + fn)],
          ['Specificity (TNR)', tn / (tn + fp)],
          ['FPR',               fp / (fp + tn)],
          ['FNR',               fn / (fn + tp)],
        ].map(([l, v]) => (
          <div key={l} style={{ display: 'flex', justifyContent: 'space-between', padding: '6px 10px', borderRadius: 6, background: 'var(--bg-secondary)' }}>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>{l}</span>
            <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>{(v * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ModelPerformance() {
  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const { notify } = useNotification();

  useEffect(() => {
    client.get('/api/metrics')
      .then(r => setData(r.data))
      .catch(() => notify.error('Failed to load metrics'))
      .finally(() => setLoading(false));
  }, [notify]);

  if (loading) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header"><h1 className="page-title">Model Performance</h1></div>
      <SkeletonLoader type="kpi"/>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>{Array(4).fill(0).map((_, i) => <SkeletonLoader key={i} type="chart"/>)}</div>
    </div>
  );

  const d = data || {};

  // The API returns PR points in descending recall (sklearn's order); sort
  // ascending so the numeric axis reads left-to-right.
  const rocData = [...(d.roc_curve ?? [])].sort((a, b) => a.fpr - b.fpr);
  const prData = [...(d.pr_curve ?? [])].sort((a, b) => a.recall - b.recall);
  // The PR chart's "random" floor is the positive rate, read off the confusion
  // matrix so it always matches the split actually being shown.
  const cm = d.confusion_matrix ?? {};
  const totalTest = (cm.tn ?? 0) + (cm.fp ?? 0) + (cm.fn ?? 0) + (cm.tp ?? 0);
  const baseRate = totalTest ? ((cm.tp ?? 0) + (cm.fn ?? 0)) / totalTest : 0;

  const CustomTT = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: 8, padding: '8px 12px', fontSize: 12 }}>
        {payload.map((p, i) => <div key={i} style={{ color: p.color }}>{p.name}: {p.value?.toFixed(4)}</div>)}
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Model Performance</h1>
          <p className="page-subtitle">
            {d.model_name ?? 'Model'} on the held-out test split{d.split_mode ? ` · ${d.split_mode} split` : ''}{d.threshold != null ? ` · decision threshold ${d.threshold}` : ''}
          </p>
        </div>
      </div>

      {/* Metric cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px,1fr))', gap: 14 }}>
        {/* Ordered by how much each number actually tells you on a 3.5%-positive
            problem. Accuracy is last and deliberately unhighlighted. */}
        <Metric label="PR-AUC"    value={d.pr_auc}    color="var(--brand)" sub={`vs ${(baseRate * 100).toFixed(2)}% if random`}/>
        <Metric label="F1 Score"  value={d.f1}        color="var(--cat-1)" sub="Harmonic mean P·R"/>
        <Metric label="Recall"    value={d.recall}    color="var(--cat-3)" sub="share of fraud caught"/>
        <Metric label="Precision" value={d.precision} color="var(--status-warning)" sub="share of flags that are fraud"/>
        <Metric label="ROC-AUC"   value={d.roc_auc}   color="var(--cat-4)" sub="Area under ROC curve"/>
        <Metric label="Accuracy"  value={d.accuracy}  color="var(--text-muted)" sub={`flagging nothing scores ${((1 - baseRate) * 100).toFixed(1)}%`}/>
      </div>

      {/* ROC + PR curves */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div className="chart-container">
          <div className="chart-title">ROC Curve</div>
          <div className="chart-subtitle">AUC = {d.roc_auc?.toFixed(4)}</div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={rocData} margin={{ top: 8, right: 12, bottom: 12, left: 0 }}>
              <CartesianGrid strokeDasharray="3 5" stroke="var(--border)"/>
              {/* type="number" matters: without it the axis is categorical and
                  the curve's geometry is meaningless. */}
              <XAxis
                dataKey="fpr" type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]}
                tick={{ fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)}
                label={{ value: 'False positive rate', position: 'insideBottom', offset: -6, fontSize: 11, fill: 'var(--text-muted)' }}
              />
              <YAxis
                type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]}
                tick={{ fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)}
                label={{ value: 'True positive rate', angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--text-muted)' }}
              />
              <Tooltip content={<CustomTT/>}/>
              {/* Chance line: a classifier on this diagonal has learned nothing. */}
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                stroke="var(--text-faint)" strokeDasharray="5 5" strokeWidth={1}/>
              <Line type="monotone" dataKey="tpr" stroke="var(--brand)" dot={false} strokeWidth={2.5} name="TPR"/>
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <div className="chart-title">Precision-Recall Curve</div>
          <div className="chart-subtitle">PR-AUC = {d.pr_auc?.toFixed(4)}</div>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={prData} margin={{ top: 8, right: 12, bottom: 12, left: 0 }}>
              <CartesianGrid strokeDasharray="3 5" stroke="var(--border)"/>
              <XAxis
                dataKey="recall" type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]}
                tick={{ fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)}
                label={{ value: 'Recall', position: 'insideBottom', offset: -6, fontSize: 11, fill: 'var(--text-muted)' }}
              />
              <YAxis
                type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]}
                tick={{ fontSize: 10 }} tickFormatter={(v) => v.toFixed(2)}
                label={{ value: 'Precision', angle: -90, position: 'insideLeft', fontSize: 11, fill: 'var(--text-muted)' }}
              />
              <Tooltip content={<CustomTT/>}/>
              {/* A PR curve has no diagonal. Its floor is the positive rate, and
                  without that line a reader cannot judge whether the curve is good. */}
              <ReferenceLine y={baseRate} stroke="var(--text-faint)" strokeDasharray="5 5" strokeWidth={1}
                label={{ value: `random = ${(baseRate * 100).toFixed(1)}%`, position: 'insideTopRight', fontSize: 10, fill: 'var(--text-muted)' }}/>
              <Line type="monotone" dataKey="precision" stroke="var(--brand)" dot={false} strokeWidth={2.5} name="Precision"/>
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Confusion Matrix + Training curves */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div className="chart-container">
          <div className="chart-title">Confusion Matrix</div>
          <div className="chart-subtitle">Test set predictions</div>
          {d.confusion_matrix && <ConfusionMatrix cm={d.confusion_matrix}/>}
        </div>

        <div className="chart-container">
          <div className="chart-title">Training Curves</div>
          <div className="chart-subtitle">
            GraphSAGE training run — {d.training_curves?.length ?? 0} logged epochs{d.active_model && d.active_model !== 'gnn' ? ' (the served model blends this GNN with LightGBM)' : ''}
          </div>
          {/* SINGLE y-axis. This was a dual-axis chart (a second y-scale on the
              right for AUC), which is the most misleading thing a line chart can
              do: it lets two unrelated scales be visually compared and makes any
              crossing point meaningless. All three series here are already in
              0-1, so one axis is both correct and simpler.
              Series use categorical slots, not status colours - status is
              reserved for fraud/legit semantics and must not double as "series 3". */}
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={d.training_curves}>
              <CartesianGrid strokeDasharray="3 5" stroke="var(--border)"/>
              <XAxis dataKey="epoch" tick={{ fontSize: 10 }}/>
              <YAxis domain={[0, 1]} tick={{ fontSize: 10 }}/>
              <Tooltip content={<CustomTT/>}/>
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }}/>
              <Line type="monotone" dataKey="train_loss" stroke="var(--cat-3)" dot={false} strokeWidth={2} name="Train loss"/>
              {/* Validation PR-AUC replaces Val Loss: training now selects on
                  average precision, the metric that tracks fraud detection on a
                  3.5%-positive dataset. */}
              <Line type="monotone" dataKey="val_pr_auc" stroke="var(--cat-5)" dot={false} strokeWidth={2} name="Val PR-AUC"/>
              <Line type="monotone" dataKey="val_auc"    stroke="var(--cat-1)" dot={false} strokeWidth={2} name="Val ROC-AUC"/>
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Baseline comparison */}
      {d.baselines?.length > 0 && (
        <div className="card" style={{ padding: 24 }}>
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 16 }}>Baseline Comparison</div>
          {/* The GNN row used to be printed in success-green with a star, which
              asserts a win the numbers do not support - gradient boosting beats
              it on F1 and PR-AUC. "Ours" is now marked neutrally and the BEST
              value in each column is the thing that gets emphasised, whoever
              it belongs to. */}
          {(() => {
            const rows = [
              ...d.baselines.map(b => ({
                name: b.Model,
                ours: false,
                accuracy: parseFloat(b.Accuracy),
                roc: parseFloat(b['ROC-AUC']),
                f1: parseFloat(b.F1),
                pr: b['PR-AUC'] != null ? parseFloat(b['PR-AUC']) : NaN,
              })),
              // Named after the model actually being served. This row used to say
              // 'GraphSAGE (ours)' whatever was active, so the ensemble's numbers
              // were printed under the GNN's name.
              { name: `${d.model_name ?? 'Served model'} (served)`, ours: true, accuracy: d.accuracy, roc: d.roc_auc, f1: d.f1, pr: d.pr_auc },
            ];
            const best = k => Math.max(...rows.map(r => (Number.isFinite(r[k]) ? r[k] : -Infinity)));
            const bests = { accuracy: best('accuracy'), roc: best('roc'), f1: best('f1'), pr: best('pr') };
            const Cell = ({ v, k }) => {
              const isBest = Number.isFinite(v) && Math.abs(v - bests[k]) < 1e-9;
              return (
                <td className="mono" style={{ color: isBest ? 'var(--text-primary)' : 'var(--text-secondary)', fontWeight: isBest ? 700 : 400 }}>
                  {Number.isFinite(v) ? (v * 100).toFixed(1) + '%' : '—'}
                  {isBest && <span style={{ color: 'var(--brand)', marginLeft: 5 }} title="best in column">●</span>}
                </td>
              );
            };
            return (
              <div className="table-wrap" style={{ border: 'none' }}>
                <table>
                  <thead>
                    <tr>{['Model','Accuracy','ROC-AUC','F1','PR-AUC'].map(h => <th key={h}>{h}</th>)}</tr>
                  </thead>
                  <tbody>
                    {rows.map(r => (
                      <tr key={r.name} style={r.ours ? { background: 'color-mix(in srgb, var(--brand) 7%, transparent)' } : undefined}>
                        <td style={{ fontWeight: 600, color: r.ours ? 'var(--brand-strong)' : 'var(--text-primary)' }}>{r.name}</td>
                        <Cell v={r.accuracy} k="accuracy"/>
                        <Cell v={r.roc} k="roc"/>
                        <Cell v={r.f1} k="f1"/>
                        <Cell v={r.pr} k="pr"/>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', padding: '12px 16px 4px' }}>
                  <span style={{ color: 'var(--brand)' }}>●</span> best in column. Accuracy is a weak
                  signal here — always predicting “legitimate” scores about 96.5% on this split.
                </div>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}
