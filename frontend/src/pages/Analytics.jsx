import { useEffect, useState } from 'react';
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, Legend, ResponsiveContainer,
  CartesianGrid, ReferenceLine,
} from 'recharts';
import client from '../api/client';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import { useNotification } from '../context/NotificationContext';

/*
 * Every breakdown on this page plots fraud RATE, not fraud COUNT.
 *
 * Counts are not comparable across these categories, because the categories
 * are wildly different sizes: `(missing)` covers 449,730 transactions against
 * 55,645 mobile, and the <$50 amount bucket holds ~190,000 rows against a few
 * thousand above $5k. A bar chart of counts therefore just redraws the volume
 * distribution, and the reader learns which category is common rather than
 * which one is risky.
 *
 * Rate answers the actual question, and the dashed line marks the overall
 * fraud rate so "worse than average" is visible at a glance.
 *
 * Colour: one hue per chart. The category is already named on the axis, so a
 * different hue per bar would encode nothing and invite the reader to look for
 * meaning that is not there.
 */

const tooltipStyle = {
  background: 'var(--bg-card)',
  border: '1px solid var(--border-light)',
  borderRadius: 10,
  fontSize: 12,
};

const rateTooltip = (v, _n, item) => [
  `${Number(v).toFixed(2)}%  (${item.payload.fraud?.toLocaleString()} of ${item.payload.total?.toLocaleString()})`,
  'Fraud rate',
];

/** Horizontal fraud-rate bars for a categorical breakdown. */
function RateBars({ data, labelKey, height = 220, baseRate, labelWidth = 100 }) {
  const rows = [...data]
    .map(d => ({
      ...d,
      label: String(d[labelKey] ?? '—'),
      rate: d.fraud_rate ?? (d.total ? (d.fraud / d.total) * 100 : 0),
    }))
    .sort((a, b) => b.rate - a.rate);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={rows} layout="vertical" margin={{ top: 4, right: 40, left: 0, bottom: 4 }}>
        <CartesianGrid strokeDasharray="3 5" stroke="var(--border)" horizontal={false}/>
        <XAxis type="number" tick={{ fontSize: 11 }} tickLine={false} axisLine={false}
          tickFormatter={v => `${v}%`}/>
        <YAxis type="category" dataKey="label" width={labelWidth}
          tick={{ fontSize: 11 }} tickLine={false} axisLine={false}/>
        <Tooltip contentStyle={tooltipStyle} formatter={rateTooltip} cursor={{ fill: 'var(--bg-tint)' }}/>
        {baseRate > 0 && (
          <ReferenceLine x={baseRate} stroke="var(--text-faint)" strokeDasharray="4 4"
            label={{ value: 'overall', position: 'top', fontSize: 10, fill: 'var(--text-muted)' }}/>
        )}
        <Bar dataKey="rate" name="Fraud rate" fill="var(--cat-1)" radius={[0, 5, 5, 0]} maxBarSize={26}/>
      </BarChart>
    </ResponsiveContainer>
  );
}

export default function Analytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const { notify } = useNotification();

  useEffect(() => {
    client.get('/api/analytics/summary')
      .then(r => setData(r.data))
      .catch(() => notify.error('Failed to load analytics'))
      .finally(() => setLoading(false));
  }, [notify]);

  if (loading) return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
      <div className="page-header"><div><h1 className="page-title">Analytics</h1></div></div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        {Array(6).fill(0).map((_, i) => <SkeletonLoader key={i} type="chart"/>)}
      </div>
    </div>
  );

  const monthly = data?.fraud_by_month ?? [];
  const medianVolume = monthly.length
    ? [...monthly.map(m => m.total)].sort((a, b) => a - b)[Math.floor(monthly.length / 2)]
    : 0;
  const rateData = monthly.map(m => ({
    name: m.month,
    rate: m.fraud_rate ?? (m.total ? (m.fraud / m.total) * 100 : 0),
    fraud: m.fraud,
    total: m.total,
    partial: medianVolume > 0 && m.total < medianVolume * 0.4,
  }));
  const partialMonths = rateData.filter(d => d.partial);

  const amtData = (data?.fraud_by_amount ?? []).map(d => ({
    ...d,
    rate: d.fraud_rate ?? (d.total ? (d.fraud / d.total) * 100 : 0),
  }));
  const deviceData = data?.fraud_by_device ?? [];
  const cardData = data?.fraud_by_card ?? [];
  const emailData = data?.fraud_by_email ?? [];
  const distData = data?.class_dist ?? [];

  const totalAll = distData.reduce((a, b) => a + b.value, 0);
  const fraudSlice = distData.find(d => /fraud/i.test(d.name));
  const overallRate = totalAll ? ((fraudSlice?.value ?? 0) / totalAll) * 100 : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Analytics Dashboard</h1>
          <p className="page-subtitle">
            Fraud rate across each dimension, from the labelled data.
            The dashed line on each chart is the overall rate ({overallRate.toFixed(2)}%).
          </p>
        </div>
      </div>

      {/* Row 1: trend + class split */}
      <div style={{ display: 'grid', gridTemplateColumns: '2fr 1fr', gap: 20 }}>
        <div className="chart-container">
          <div className="chart-title">Fraud Rate by Month</div>
          <div className="chart-subtitle">
            Share of transactions labelled fraudulent
            {partialMonths.length > 0 && (
              <> · {partialMonths.map(d => d.name).join(', ')} {partialMonths.length > 1 ? 'are' : 'is'} a
              partial month ({partialMonths.map(d => d.total.toLocaleString()).join(', ')} transactions)</>
            )}
          </div>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={rateData} margin={{ top: 6, right: 8, left: -10, bottom: 0 }}>
              <defs>
                <linearGradient id="gRateA" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="var(--status-critical)" stopOpacity={0.26}/>
                  <stop offset="95%" stopColor="var(--status-critical)" stopOpacity={0.02}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 5" stroke="var(--border)" vertical={false}/>
              <XAxis dataKey="name" tick={{ fontSize: 11 }} tickLine={false} axisLine={false}/>
              <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false}
                domain={[0, 'auto']} tickFormatter={v => `${v}%`} width={44}/>
              <Tooltip contentStyle={tooltipStyle} formatter={rateTooltip}/>
              <ReferenceLine y={overallRate} stroke="var(--text-faint)" strokeDasharray="4 4"/>
              <Area type="monotone" dataKey="rate" stroke="var(--status-critical)" strokeWidth={2}
                fill="url(#gRateA)"
                dot={(props) => {
                  const { cx, cy, payload, index } = props;
                  if (!payload.partial) return <g key={index}/>;
                  return <circle key={index} cx={cx} cy={cy} r={4}
                    fill="var(--bg-card)" stroke="var(--status-critical)" strokeWidth={2}/>;
                }}
                activeDot={{ r: 5, strokeWidth: 2, stroke: 'var(--bg-card)' }}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <div className="chart-title">Class Distribution</div>
          <div className="chart-subtitle">Overall fraud vs legitimate</div>
          <div style={{ height: 190 }}>
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={distData} cx="50%" cy="50%" innerRadius={52} outerRadius={78}
                  paddingAngle={2} dataKey="value" nameKey="name" isAnimationActive={false}>
                  {distData.map((d, i) => (
                    <Cell key={i}
                      fill={/fraud/i.test(d.name) ? 'var(--status-critical)' : 'var(--status-good)'}/>
                  ))}
                </Pie>
                <Tooltip contentStyle={tooltipStyle}
                  formatter={(v, n) => [`${v.toLocaleString()} (${((v / totalAll) * 100).toFixed(2)}%)`, n]}/>
                <Legend formatter={v => <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{v}</span>}/>
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div style={{ textAlign: 'center', marginTop: 8 }}>
            <div className="stat-hero" style={{ fontSize: 30, color: 'var(--status-critical)' }}>
              {overallRate.toFixed(2)}%
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>overall fraud rate</div>
          </div>
        </div>
      </div>

      {/* Row 2: amount + device */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div className="chart-container">
          <div className="chart-title">Fraud Rate by Transaction Amount</div>
          <div className="chart-subtitle">
            Buckets differ hugely in size, so counts would just retrace volume
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={amtData} margin={{ top: 6, right: 8, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 5" stroke="var(--border)" vertical={false}/>
              <XAxis dataKey="bucket" tick={{ fontSize: 10 }} tickLine={false} axisLine={false}/>
              <YAxis tick={{ fontSize: 11 }} tickLine={false} axisLine={false}
                tickFormatter={v => `${v}%`} width={44}/>
              <Tooltip contentStyle={tooltipStyle} formatter={rateTooltip} cursor={{ fill: 'var(--bg-tint)' }}/>
              <ReferenceLine y={overallRate} stroke="var(--text-faint)" strokeDasharray="4 4"/>
              <Bar dataKey="rate" name="Fraud rate" fill="var(--cat-1)" radius={[5, 5, 0, 0]} maxBarSize={48}/>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-container">
          <div className="chart-title">Fraud Rate by Device Type</div>
          <div className="chart-subtitle">IEEE-CIS records desktop, mobile, or nothing at all</div>
          <RateBars data={deviceData} labelKey="device" baseRate={overallRate} labelWidth={90}/>
        </div>
      </div>

      {/* Row 3: card + email */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <div className="chart-container">
          <div className="chart-title">Fraud Rate by Card Network</div>
          <div className="chart-subtitle">Discover carries roughly double the overall rate</div>
          <RateBars data={cardData} labelKey="card_type" baseRate={overallRate} labelWidth={130}/>
        </div>

        <div className="chart-container">
          <div className="chart-title">Fraud Rate by Email Domain</div>
          <div className="chart-subtitle">Top domains by volume, ranked by their fraud rate</div>
          {/* The old version painted any domain containing "anon" or "proton"
              red regardless of its data. That is a hardcoded prejudice, not a
              finding; the rate now speaks for itself. */}
          <RateBars data={emailData} labelKey="domain" baseRate={overallRate} labelWidth={120}/>
        </div>
      </div>
    </div>
  );
}
