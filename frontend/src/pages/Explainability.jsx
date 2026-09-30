import { useState } from 'react';
import { Brain } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import client from '../api/client';
import { useNotification } from '../context/NotificationContext';
import { useNavigate } from 'react-router-dom';

export default function Explainability() {
  const { notify } = useNotification();
  const navigate   = useNavigate();
  const [nodeId,   setNodeId]  = useState('');
  const [loading,  setLoading] = useState(false);
  const [result,   setResult]  = useState(null);

  const handleExplain = async (targetId = nodeId) => {
    const idToExplain = String(targetId || nodeId).trim();
    console.log('[XAI UI] Transaction selected:', idToExplain);
    if (!idToExplain) return notify.error('Enter a Transaction ID or Node ID');
    setLoading(true); setResult(null);
    try {
      console.log('[XAI UI] Request sent', `/api/explain/${idToExplain}`);
      const res = await client.get(`/api/explain/${idToExplain}`);
      console.log('[XAI UI] Response status:', res.status);
      console.log('[XAI UI] Explanation received', res.data);
      setResult(res.data);
      notify.success('Explanation generated successfully');
    } catch (e) {
      console.error('[XAI UI] Explanation error:', e);
      notify.error(e.response?.data?.detail || 'Explanation failed');
    } finally {
      setLoading(false);
    }
  };

  const featData = result?.feature_names?.map((name, i) => ({
    name: name.length > 14 ? name.slice(0, 14) + '…' : name,
    score: parseFloat((result.feature_scores[i] * 100).toFixed(2)),
  })) ?? [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Explainable AI</h1>
          <p className="page-subtitle">GNNExplainer — understand why GraphSAGE flagged a transaction</p>
        </div>
      </div>

      {/* Input */}
      <div className="card" style={{ padding: 28 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ flex: 1, minWidth: 200 }}>
            <label className="form-label">Transaction ID or Node ID to Explain</label>
            <input className="input" type="number" placeholder="e.g. 2991340 or 4340"
              value={nodeId} onChange={e => {
                console.log('[XAI UI] Transaction selected:', e.target.value);
                setNodeId(e.target.value);
              }}
              onKeyDown={e => e.key === 'Enter' && handleExplain()}/>
          </div>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {[
              { label: 'Txn 2991340 (Fraud)', val: '2991340' },
              { label: 'Txn 2987000 (Legit)', val: '2987000' },
              { label: 'Node 4340', val: '4340' },
            ].map(item => (
              <button key={item.val} className="btn btn-ghost btn-sm" onClick={() => {
                console.log('[XAI UI] Transaction selected:', item.val);
                setNodeId(item.val);
                handleExplain(item.val);
              }}>{item.label}</button>
            ))}
          </div>
          <button className="btn btn-primary" onClick={() => handleExplain()} disabled={loading} style={{ height: 42 }}>
            {loading ? <><div className="spinner" style={{ width: 16, height: 16 }}/> Generating…</> : <><Brain size={15}/> Explain</>}
          </button>
        </div>
        <div style={{ marginTop: 12, padding: '10px 14px', borderRadius: 8, background: 'color-mix(in srgb, var(--cat-4) 6%, transparent)', border: '1px solid color-mix(in srgb, var(--cat-4) 15%, transparent)' }}>
          <span style={{ fontSize: 12.5, color: 'var(--text-secondary)' }}>
            ⚠️ GNNExplainer runs 100 optimization epochs — first run per node takes ~30 seconds.
            Subsequent runs are instant (cached).
          </span>
        </div>
      </div>

      {loading && (
        <div className="card" style={{ padding: 48, textAlign: 'center' }}>
          <div className="spinner" style={{ width: 48, height: 48, borderWidth: 3, margin: '0 auto 20px' }}/>
          <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 6 }}>Running GNNExplainer</div>
          <div style={{ fontSize: 13, color: 'var(--text-muted)' }}>
            Optimizing node and edge masks on the transaction graph…
          </div>
        </div>
      )}

      {result && !loading && (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Stats row */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14 }}>
            {[
              ['Transaction ID', result.transaction_id || result.node_id, 'var(--status-warning)'],
              ['Node Index', result.node_id, 'var(--brand)'],
              ['Important Edges', result.important_edges?.length, 'var(--cat-2)'],
              ['Neighbour Nodes', result.important_nodes?.length, 'var(--status-good)'],
            ].map(([l, v, c]) => (
              <div key={l} className="card" style={{ padding: '16px 20px', textAlign: 'center' }}>
                <div style={{ fontSize: 22, fontWeight: 800, color: c }}>{v}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>{l}</div>
              </div>
            ))}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
            {/* Feature importance bar chart */}
            <div className="chart-container">
              <div className="chart-title" style={{ marginBottom: 4 }}>Top Risk Features</div>
              <div className="chart-subtitle">Feature importance scores from GNNExplainer</div>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={featData} layout="vertical" margin={{ left: 10 }}>
                  <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={v => v.toFixed(1) + '%'}/>
                  <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 11 }}/>
                  <Tooltip
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--border-light)', borderRadius: 8 }}
                    formatter={v => [v.toFixed(2) + '%', 'Importance']}
                  />
                  <Bar dataKey="score" radius={[0, 4, 4, 0]}>
                    {featData.map((_, i) => (
                      <Cell key={i} fill={i < 3 ? 'var(--status-critical)' : i < 6 ? 'var(--status-warning)' : 'var(--brand)'}/>
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Explanation image */}
            <div className="chart-container" style={{ display: 'flex', flexDirection: 'column' }}>
              <div className="chart-title" style={{ marginBottom: 4 }}>GNNExplainer Visualization</div>
              <div className="chart-subtitle">Transaction neighbourhood + risk features</div>
              {result.image_url ? (
                <img
                  src={`http://localhost:8000${result.image_url}`}
                  alt="GNNExplainer visualization"
                  style={{ width: '100%', borderRadius: 8, marginTop: 8, flex: 1, objectFit: 'contain', background: 'var(--bg-secondary)' }}
                  onError={e => { e.target.style.display = 'none'; }}
                />
              ) : (
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)' }}>
                  No image generated
                </div>
              )}
            </div>
          </div>

          {/* Important edges table */}
          <div className="card" style={{ padding: 24 }}>
            <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 16 }}>Important Graph Edges</div>
            <div className="table-wrap" style={{ maxHeight: 280, overflowY: 'auto' }}>
              <table>
                <thead>
                  <tr>
                    <th>Source Node</th>
                    <th>Target Node</th>
                    <th>Edge Weight</th>
                    <th>Importance</th>
                  </tr>
                </thead>
                <tbody>
                  {result.important_edges?.slice(0, 15).map((e, i) => (
                    <tr key={i}>
                      <td className="mono" style={{ fontSize: 12, color: e.source === result.node_id ? 'var(--status-critical)' : 'var(--text-secondary)' }}>
                        {e.source === result.node_id ? '★ ' : ''}{e.source}
                      </td>
                      <td className="mono" style={{ fontSize: 12, color: e.target === result.node_id ? 'var(--status-critical)' : 'var(--text-secondary)' }}>
                        {e.target === result.node_id ? '★ ' : ''}{e.target}
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div className="progress-bar" style={{ flex: 1, maxWidth: 100 }}>
                            <div className="progress-fill" style={{ width: `${e.weight * 100}%`, background: 'var(--brand)' }}/>
                          </div>
                          <span className="mono" style={{ fontSize: 11, color: 'var(--text-muted)' }}>{e.weight.toFixed(4)}</span>
                        </div>
                      </td>
                      <td>
                        <span className={`badge ${e.weight > 0.7 ? 'badge-critical' : e.weight > 0.4 ? 'badge-high' : 'badge-medium'}`}>
                          {e.weight > 0.7 ? 'HIGH' : e.weight > 0.4 ? 'MEDIUM' : 'LOW'}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={{ display: 'flex', gap: 10 }}>
            <button className="btn btn-ghost" onClick={() => navigate(`/graph/${nodeId}`)}>🌐 View Full Graph</button>
            <button className="btn btn-ghost" onClick={() => navigate(`/transactions/${nodeId}`)}>📄 Transaction Detail</button>
          </div>
        </div>
      )}
    </div>
  );
}
