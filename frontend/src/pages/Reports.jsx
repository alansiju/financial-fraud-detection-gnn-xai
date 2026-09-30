import { useState, useEffect } from 'react';
import { FileText, Download, Table, Sheet } from 'lucide-react';
import client from '../api/client';
import { useNotification } from '../context/NotificationContext';

const downloadBlob = (blob, filename) => {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  URL.revokeObjectURL(url);
};

export default function Reports() {
  const { notify } = useNotification();
  const [loading, setLoading] = useState({});
  const [limit,   setLimit]   = useState(1000);
  const [fraudOnly, setFraudOnly] = useState(false);

  const dl = async (type) => {
    setLoading(l => ({ ...l, [type]: true }));
    try {
      const params = { limit, fraud_only: fraudOnly };
      const res = await client.get(`/api/reports/${type}`, { params, responseType: 'blob' });
      const ext = type === 'pdf' ? 'pdf' : type === 'excel' ? 'xlsx' : 'csv';
      const ts  = new Date().toISOString().slice(0,10);
      downloadBlob(res.data, `fraud_report_${ts}.${ext}`);
      notify.success(`${type.toUpperCase()} report downloaded`);
    } catch { notify.error(`Failed to generate ${type} report`); }
    finally { setLoading(l => ({ ...l, [type]: false })); }
  };

  const REPORTS = [
    {
      type: 'pdf', icon: FileText, color: 'var(--status-critical)', bg: 'color-mix(in srgb, var(--status-critical) 8%, transparent)',
      title: 'PDF Summary Report',
      desc: 'Professional summary with model overview, performance metrics, and dataset statistics.',
      features: ['Model architecture overview', 'Performance metrics table', 'Dataset statistics', 'Comparison with baselines'],
    },
    {
      type: 'csv', icon: Table, color: 'var(--status-good)', bg: 'color-mix(in srgb, var(--status-good) 8%, transparent)',
      title: 'CSV Predictions Export',
      desc: 'Full predictions export with fraud probabilities and risk levels for all transactions.',
      features: ['Node ID + Transaction ID', 'Fraud probability', 'Risk level', 'True label'],
    },
    {
      type: 'excel', icon: Sheet, color: 'var(--brand)', bg: 'color-mix(in srgb, var(--brand) 8%, transparent)',
      title: 'Excel Workbook',
      desc: 'Color-coded Excel report with styled predictions and risk categorization.',
      features: ['Colour-coded rows (fraud/legit)', 'Risk level formatting', 'All metrics', 'Professional styling'],
    },
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Reports</h1>
          <p className="page-subtitle">Export fraud detection results and model performance reports</p>
        </div>
      </div>

      {/* Options */}
      <div className="card" style={{ padding: 24 }}>
        <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 16 }}>Export Options</div>
        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <div className="form-group" style={{ flex: 1, minWidth: 180 }}>
            <label className="form-label">Row Limit</label>
            <select className="input" value={limit} onChange={e => setLimit(parseInt(e.target.value))}>
              <option value={500}>500 rows</option>
              <option value={1000}>1,000 rows</option>
              <option value={5000}>5,000 rows</option>
              <option value={10000}>10,000 rows</option>
            </select>
          </div>
          <div className="form-group">
            <label className="form-label">Filter</label>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', height: 42, fontSize: 13.5 }}>
              <input type="checkbox" checked={fraudOnly} onChange={e => setFraudOnly(e.target.checked)}
                style={{ accentColor: 'var(--status-critical)', width: 16, height: 16 }}/>
              Fraud transactions only
            </label>
          </div>
        </div>
      </div>

      {/* Report cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px,1fr))', gap: 20 }}>
        {REPORTS.map(r => (
          <div key={r.type} className="card" style={{ padding: 28, border: `1px solid color-mix(in srgb, ${r.color} 13%, transparent)`, '--accent-color': r.color }}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 16, marginBottom: 20 }}>
              <div style={{ width: 48, height: 48, borderRadius: 12, background: r.bg, display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0 }}>
                <r.icon size={22} color={r.color}/>
              </div>
              <div>
                <div style={{ fontWeight: 700, fontSize: 15, marginBottom: 4 }}>{r.title}</div>
                <div style={{ fontSize: 13, color: 'var(--text-muted)', lineHeight: 1.6 }}>{r.desc}</div>
              </div>
            </div>
            <ul style={{ margin: '0 0 20px 0', padding: 0, listStyle: 'none', display: 'flex', flexDirection: 'column', gap: 6 }}>
              {r.features.map(f => (
                <li key={f} style={{ fontSize: 12.5, color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ color: r.color, fontWeight: 700 }}>✓</span> {f}
                </li>
              ))}
            </ul>
            <button
              className="btn btn-primary"
              style={{ width: '100%', background: r.color, boxShadow: `0 4px 12px color-mix(in srgb, ${r.color} 27%, transparent)` }}
              onClick={() => dl(r.type)}
              disabled={loading[r.type]}
            >
              {loading[r.type]
                ? <><div className="spinner" style={{ width: 15, height: 15 }}/> Generating…</>
                : <><Download size={15}/> Download {r.type.toUpperCase()}</>}
            </button>
          </div>
        ))}
      </div>

      {/* History */}
      <div className="card" style={{ padding: 24 }}>
        <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 4 }}>Recent Predictions Log</div>
        <div style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 16 }}>
          Last 50 predictions made through the platform
        </div>
        <PredictionHistory/>
      </div>
    </div>
  );
}

function PredictionHistory() {
  const [rows,    setRows]    = useState([]);
  const [loading, setLoading] = useState(true);
  const { notify } = useNotification();

  // Was `useState(() => fetch…, [])`: useState's initialiser runs during
  // render (twice under StrictMode) and ignores the deps array. And the
  // empty catch turned a failed request into "No predictions yet".
  useEffect(() => {
    client.get('/api/predict/history', { params: { limit: 50 } })
      .then(r => setRows(r.data))
      .catch(() => notify.error('Failed to load prediction history'))
      .finally(() => setLoading(false));
  }, [notify]);

  if (loading) return <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 20 }}>Loading…</div>;
  if (!rows.length) return <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: 20 }}>No predictions yet — run some predictions first!</div>;

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {['#','Node ID','Fraud Prob','Is Fraud','Confidence','User','Time'].map(h => <th key={h}>{h}</th>)}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.id}>
              <td style={{ color: 'var(--text-muted)', fontSize: 12 }}>{i+1}</td>
              <td className="mono" style={{ fontSize: 12 }}>{r.node_id}</td>
              <td style={{ color: r.is_fraud ? 'var(--status-critical)' : 'var(--status-good)', fontWeight: 600 }}>{(r.fraud_prob * 100).toFixed(1)}%</td>
              <td><span className={`badge ${r.is_fraud ? 'badge-fraud' : 'badge-legit'}`}>{r.is_fraud ? 'Fraud' : 'Legit'}</span></td>
              <td className="mono" style={{ fontSize: 12 }}>{(r.confidence * 100).toFixed(1)}%</td>
              <td style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{r.username}</td>
              <td style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>{new Date(r.created_at).toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
