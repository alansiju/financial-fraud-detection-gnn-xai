import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CreditCard, ShieldAlert, TrendingUp, Target, Brain, Network, BarChart3, ArrowRight, Activity, Zap, AlertTriangle
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, Legend, CartesianGrid
} from 'recharts';
import client from '../api/client';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import { useNotification } from '../context/NotificationContext';

const fmt = (n) => {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n?.toLocaleString() ?? '—';
};
const pct = (v) => (v * 100).toFixed(1) + '%';

export default function Dashboard() {
  const [stats,    setStats]    = useState(null);
  const [analytics,setAnalytics]= useState(null);
  const [loading,  setLoading]  = useState(true);
  const { notify } = useNotification();
  const navigate = useNavigate();

  useEffect(() => {
    Promise.all([
      client.get('/api/dashboard'),
      client.get('/api/analytics/summary'),
    ]).then(([s, a]) => {
      setStats(s.data);
      setAnalytics(a.data);
      notify.success('Dashboard loaded successfully');
    }).catch(() => {
      notify.error('Failed to load dashboard data');
    }).finally(() => setLoading(false));
  }, [notify]);

  // Every figure below is measured. Cards carry a `note` instead of a fake
  // delta; where a comparison is shown it is computed, not asserted.
  const baseRate = stats ? stats.fraud_percentage / 100 : 0;
  const KPI_CARDS = stats ? [
    {
      icon: CreditCard, label: 'Total Transactions', value: fmt(stats.total_transactions),
      color: 'var(--brand)', bg: 'color-mix(in srgb, var(--brand) 10%, transparent)',
      sub: `${stats.graph_nodes?.toLocaleString()} graph nodes`,
      note: `${stats.graph_edges?.toLocaleString() ?? '—'} edges`,
    },
    {
      icon: ShieldAlert, label: 'Known Fraud', value: fmt(stats.fraud_detected),
      color: 'var(--status-critical)', bg: 'color-mix(in srgb, var(--status-critical) 10%, transparent)',
      sub: `${stats.fraud_percentage}% of all transactions`,
      note: 'labelled ground truth',
    },
    {
      icon: Target, label: 'Recall', value: pct(stats.recall),
      color: 'var(--cat-3)', bg: 'color-mix(in srgb, var(--cat-3) 10%, transparent)',
      sub: 'share of fraud the model catches',
      note: 'the number a fraud team is judged on',
    },
    {
      icon: Brain, label: 'Precision', value: pct(stats.precision),
      color: 'var(--status-warning)', bg: 'color-mix(in srgb, var(--status-warning) 10%, transparent)',
      sub: 'share of flagged that really is fraud',
      note: `F1 ${stats.f1_score?.toFixed(3)}`,
    },
    {
      icon: TrendingUp, label: 'ROC-AUC', value: stats.roc_auc?.toFixed(3),
      color: 'var(--cat-4)', bg: 'color-mix(in srgb, var(--cat-4) 10%, transparent)',
      sub: 'ranking quality on the test split',
      note: `best val PR-AUC ${stats.best_val_pr_auc?.toFixed(3) ?? '—'}`,
    },
    {
      icon: Activity, label: 'Accuracy', value: pct(stats.model_accuracy),
      color: 'var(--text-muted)', bg: 'var(--bg-tint)',
      sub: `flagging nothing scores ${pct(1 - baseRate)}`,
      note: 'weak signal at this base rate',
      weak: true,
    },
  ] : [];

  // Fraud RATE, not raw counts. See the note on the chart below.
  const monthly = analytics?.fraud_by_month ?? [];
  const medianVolume = monthly.length
    ? [...monthly.map(m => m.total)].sort((a, b) => a - b)[Math.floor(monthly.length / 2)]
    : 0;
  const trendData = monthly.map(m => ({
    name: m.month,
    rate: m.fraud_rate ?? (m.total ? (m.fraud / m.total) * 100 : 0),
    fraud: m.fraud,
    total: m.total,
    // A month with a fraction of the usual volume is an incomplete month, not a
    // collapse in activity; flag it rather than letting the reader infer a trend.
    partial: medianVolume > 0 && m.total < medianVolume * 0.4,
  }));
  const partialMonths = trendData.filter(d => d.partial);

  const distData = analytics?.class_dist ?? [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 28 }}>
      {/* Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">
            Fraud Detection Dashboard
          </h1>
          <p className="page-subtitle">
            Real-time graph analysis on IEEE-CIS · {stats?.model_name ?? 'model'} · {stats?.split_mode ?? '—'} split · threshold {stats?.threshold ?? '—'}
          </p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/analytics')}>
            <BarChart3 size={14}/> Analytics
          </button>
          <button className="btn btn-primary btn-sm" onClick={() => navigate('/predict')}>
            <Zap size={14}/> Run Prediction
          </button>
        </div>
      </div>

      {/* KPI Grid */}
      {loading ? (
        <SkeletonLoader type="kpi"/>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(270px, 1fr))', gap: 16 }}>
          {KPI_CARDS.map((k, i) => (
            <div key={i} className="kpi-card animate-slide-up"
              style={{ '--accent-color': k.color, animationDelay: `${i * 60}ms` }}>
              <div className="kpi-icon-wrap" style={{ background: k.bg }}>
                <k.icon size={22} color={k.color}/>
              </div>
              <div className="kpi-value animate-count-up">{k.value}</div>
              <div className="kpi-label">{k.label}</div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 4 }}>{k.sub}</div>
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)', marginTop: 10, fontWeight: 500 }}>
                {k.note}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Charts Row 1 */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20 }}>
        {/* Fraud trend */}
        <div className="chart-container">
          <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 16 }}>
            <div>
              <div className="chart-title">Fraud Rate by Month</div>
              <div className="chart-subtitle">
                Share of transactions labelled fraudulent
                {partialMonths.length > 0 && (
                  <> · {partialMonths.map(d => d.name).join(', ')} {partialMonths.length > 1 ? 'are' : 'is'} a
                  partial month ({partialMonths.map(d => d.total.toLocaleString()).join(', ')} transactions)</>
                )}
              </div>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => navigate('/analytics')}>
              View all <ArrowRight size={13}/>
            </button>
          </div>
          {loading ? <div className="skeleton" style={{ height: 260, marginTop: 12 }}/> : (
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={trendData} margin={{ top: 6, right: 8, left: -8, bottom: 0 }}>
              <defs>
                <linearGradient id="gRate" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--status-critical)" stopOpacity={0.26}/>
                  <stop offset="95%" stopColor="var(--status-critical)" stopOpacity={0.02}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 5" stroke="var(--border)" vertical={false}/>
              <XAxis dataKey="name" tick={{ fontSize: 11 }} tickLine={false} axisLine={false}/>
              <YAxis
                tick={{ fontSize: 11 }} tickLine={false} axisLine={false}
                domain={[0, 'auto']} tickFormatter={(v) => `${v}%`} width={44}
              />
              <Tooltip
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: 10 }}
                formatter={(v, _n, item) => [
                  `${Number(v).toFixed(2)}%  (${item.payload.fraud.toLocaleString()} of ${item.payload.total.toLocaleString()})`,
                  'Fraud rate',
                ]}
              />
              {/* One series, so the title names it and no legend box is needed. */}
              <Area
                type="monotone" dataKey="rate"
                stroke="var(--status-critical)" strokeWidth={2}
                fill="url(#gRate)"
                dot={(props) => {
                  const { cx, cy, payload, index } = props;
                  if (!payload.partial) return <g key={index}/>;
                  // Hollow marker calls out the incomplete month.
                  return (
                    <circle key={index} cx={cx} cy={cy} r={4}
                      fill="var(--bg-card)" stroke="var(--status-critical)" strokeWidth={2}/>
                  );
                }}
                activeDot={{ r: 5, strokeWidth: 2, stroke: 'var(--bg-card)' }}
              />
            </AreaChart>
          </ResponsiveContainer>
          )}
        </div>

        {/* Class distribution donut */}
        <div className="chart-container" style={{ display: 'flex', flexDirection: 'column' }}>
          <div className="chart-title">Class Distribution</div>
          <div className="chart-subtitle">Fraud vs legitimate split</div>
          {loading ? <div className="skeleton" style={{ height: 220 }}/> : (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
              <ResponsiveContainer width="100%" height={180}>
                <PieChart>
                  <Pie
                    data={distData}
                    cx="50%" cy="50%"
                    innerRadius={55} outerRadius={80}
                    paddingAngle={3}
                    dataKey="value"
                    nameKey="name"
                  >
                    <Cell fill="var(--status-good)" name="Legitimate"/>
                    <Cell fill="var(--status-critical)" name="Fraud"/>
                  </Pie>
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: 8 }}
                    formatter={(v) => fmt(v)}
                  />
                  <Legend formatter={(v) => <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{v}</span>}/>
                </PieChart>
              </ResponsiveContainer>
              {stats && (
                <div style={{ textAlign: 'center', marginTop: 4 }}>
                  <div style={{ fontSize: 22, fontWeight: 800, color: 'var(--status-critical)' }}>{stats.fraud_percentage}%</div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>Fraud Rate</div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Model performance + Quick actions */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {/* Model metrics */}
        <div className="card" style={{ padding: 24 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 20 }}>
            <div>
              <div className="chart-title">Model Performance</div>
              <div className="chart-subtitle">{stats?.model_name ?? 'Model'} on the held-out test split</div>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={() => navigate('/performance')}>
              Details <ArrowRight size={13}/>
            </button>
          </div>
          {loading ? <SkeletonLoader rows={4}/> : stats && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
              {[
                { label: 'Accuracy',  value: stats.model_accuracy, color: 'var(--cat-2)' },
                { label: 'ROC-AUC',  value: stats.roc_auc,         color: 'var(--cat-4)' },
                { label: 'Precision', value: stats.precision,       color: 'var(--brand)' },
                { label: 'Recall',    value: stats.recall,          color: 'var(--status-warning)' },
                { label: 'F1 Score',  value: stats.f1_score,        color: 'var(--status-good)' },
              ].map(m => (
                <div key={m.label}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                    <span style={{ fontSize: 13, color: 'var(--text-secondary)', fontWeight: 500 }}>{m.label}</span>
                    {/* Ink, not the series colour: hue on 13px text failed contrast in dark
                        mode (4.14:1 indigo, 3.12:1 mulberry). The bar beside it carries
                        the identity. */}
                    <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>{(m.value * 100).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${m.value * 100}%`, background: m.color }}/>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Quick actions */}
        <div className="card" style={{ padding: 24 }}>
          <div className="chart-title" style={{ marginBottom: 8 }}>Quick Actions</div>
          <div className="chart-subtitle" style={{ marginBottom: 20 }}>Common fraud detection tasks</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
            {[
              { icon: Zap,        label: 'Run Fraud Prediction',   sub: 'Manual entry or CSV upload',  to: '/predict',     color: 'var(--brand)', bg: 'color-mix(in srgb, var(--brand) 8%, transparent)' },
              { icon: Brain,      label: 'Explainability Report',  sub: 'Why did the model flag this?', to: '/explain',     color: 'var(--cat-4)', bg: 'color-mix(in srgb, var(--cat-4) 8%, transparent)' },
              { icon: Network,    label: 'Graph Visualization',    sub: 'Interactive transaction network', to: '/graph',    color: 'var(--cat-2)', bg: 'color-mix(in srgb, var(--cat-2) 8%, transparent)' },
              { icon: AlertTriangle, label: 'View Fraud Alerts',   sub: 'High-risk transactions',       to: '/transactions?fraud=fraud', color: 'var(--status-critical)', bg: 'color-mix(in srgb, var(--status-critical) 8%, transparent)' },
            ].map(a => (
              <button
                key={a.label}
                onClick={() => navigate(a.to)}
                style={{
                  display: 'flex', alignItems: 'center', gap: 14,
                  padding: '14px 16px', borderRadius: 10, cursor: 'pointer',
                  background: a.bg, border: `1px solid color-mix(in srgb, ${a.color} 13%, transparent)`,
                  textAlign: 'left', width: '100%', transition: 'var(--transition)'
                }}
                onMouseEnter={e => e.currentTarget.style.transform = 'translateX(4px)'}
                onMouseLeave={e => e.currentTarget.style.transform = 'none'}
              >
                <div style={{ width: 36, height: 36, borderRadius: 8, background: `color-mix(in srgb, ${a.color} 13%, transparent)`, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                  <a.icon size={17} color={a.color}/>
                </div>
                <div>
                  <div style={{ fontSize: 13.5, fontWeight: 600, color: 'var(--text-primary)' }}>{a.label}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>{a.sub}</div>
                </div>
                <ArrowRight size={15} style={{ marginLeft: 'auto', color: 'var(--text-muted)' }}/>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
