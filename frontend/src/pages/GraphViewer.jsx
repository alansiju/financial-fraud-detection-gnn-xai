import { useState, useEffect, useRef, useCallback } from 'react';
import { Network, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import client from '../api/client';
import { useNotification } from '../context/NotificationContext';
import { useParams, useNavigate } from 'react-router-dom';

const NODE_COLORS = { target: 'var(--status-warning)', fraud: 'var(--status-critical)', legit: 'var(--status-good)' };

function ForceGraph({ nodes, edges, centerId }) {
  const canvasRef = useRef(null);
  const animRef   = useRef(null);
  const posRef    = useRef({});
  const velRef    = useRef({});
  const [tooltip, setTooltip] = useState(null);
  const [zoom, setZoom]       = useState(1);
  const [pan, setPan]         = useState({ x: 0, y: 0 });
  const dragging = useRef(null);
  const lastMouse = useRef(null);

  // Init positions
  useEffect(() => {
    const pos = {}; const vel = {};
    nodes.forEach((n, i) => {
      const angle = (i / nodes.length) * 2 * Math.PI;
      const r = n.id === centerId ? 0 : 150 + Math.random() * 80;
      pos[n.id] = { x: Math.cos(angle) * r, y: Math.sin(angle) * r };
      vel[n.id] = { x: 0, y: 0 };
    });
    posRef.current = pos;
    velRef.current = vel;
  }, [nodes, centerId]);

  const simulate = useCallback(() => {
    const pos = posRef.current;
    const vel = velRef.current;
    const nodeList = nodes;
    const edgeList = edges;
    const k = 80;

    // Repulsion
    for (let i = 0; i < nodeList.length; i++) {
      for (let j = i + 1; j < nodeList.length; j++) {
        const a = nodeList[i].id, b = nodeList[j].id;
        if (!pos[a] || !pos[b]) continue;
        const dx = pos[b].x - pos[a].x, dy = pos[b].y - pos[a].y;
        const d  = Math.sqrt(dx * dx + dy * dy) || 1;
        const f  = k * k / d;
        vel[a].x -= (dx / d) * f * 0.01;
        vel[a].y -= (dy / d) * f * 0.01;
        vel[b].x += (dx / d) * f * 0.01;
        vel[b].y += (dy / d) * f * 0.01;
      }
    }
    // Attraction
    edgeList.forEach(e => {
      const a = e.source, b = e.target;
      if (!pos[a] || !pos[b]) return;
      const dx = pos[b].x - pos[a].x, dy = pos[b].y - pos[a].y;
      const d  = Math.sqrt(dx * dx + dy * dy) || 1;
      const f  = (d - k) * 0.002;
      vel[a].x += (dx / d) * f;
      vel[a].y += (dy / d) * f;
      vel[b].x -= (dx / d) * f;
      vel[b].y -= (dy / d) * f;
    });
    // Center force for centerId
    if (pos[centerId]) {
      vel[centerId].x -= pos[centerId].x * 0.05;
      vel[centerId].y -= pos[centerId].y * 0.05;
    }
    // Damping + update
    Object.keys(vel).forEach(id => {
      vel[id].x *= 0.85; vel[id].y *= 0.85;
      if (pos[id]) {
        pos[id].x += vel[id].x;
        pos[id].y += vel[id].y;
      }
    });
  }, [nodes, edges, centerId]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx  = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const pos = posRef.current;
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(W / 2 + pan.x, H / 2 + pan.y);
    ctx.scale(zoom, zoom);

    // Edges
    edges.forEach(e => {
      const a = pos[e.source], b = pos[e.target];
      if (!a || !b) return;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = 'color-mix(in srgb, var(--text-muted) 35%, transparent)';
      ctx.lineWidth = 1 / zoom;
      ctx.stroke();
    });

    // Nodes
    nodes.forEach(n => {
      const p = pos[n.id];
      if (!p) return;
      const r = n.id === centerId ? 16 : 9;
      const c = NODE_COLORS[n.type] || 'var(--brand)';
      ctx.beginPath();
      ctx.arc(p.x, p.y, r / zoom, 0, 2 * Math.PI);
      ctx.fillStyle = c + '33';
      ctx.fill();
      ctx.strokeStyle = c;
      ctx.lineWidth = n.id === centerId ? 3 / zoom : 1.5 / zoom;
      ctx.stroke();
      // Label
      if (n.id === centerId || zoom > 1.2) {
        ctx.fillStyle = 'color-mix(in srgb, var(--text-primary) 90%, transparent)';
        ctx.font = `${Math.max(8, 10 / zoom)}px Inter`;
        ctx.textAlign = 'center';
        const displayLabel = n.id === centerId ? `TARGET (${n.label || n.id})` : (n.label || n.id);
        ctx.fillText(displayLabel, p.x, p.y + r / zoom + 10 / zoom);
      }
    });
    ctx.restore();
  }, [nodes, edges, centerId, zoom, pan]);

  useEffect(() => {
    const loop = () => {
      simulate();
      draw();
      animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [simulate, draw]);

  // Mouse events
  const getNodeAt = (mx, my) => {
    const canvas = canvasRef.current;
    const W = canvas.width, H = canvas.height;
    const pos = posRef.current;
    const gx = (mx - W / 2 - pan.x) / zoom;
    const gy = (my - H / 2 - pan.y) / zoom;
    return nodes.find(n => {
      const p = pos[n.id];
      if (!p) return false;
      const r = n.id === centerId ? 16 : 9;
      return Math.hypot(gx - p.x, gy - p.y) < r;
    });
  };

  return (
    <div style={{ position: 'relative' }}>
      <canvas
        ref={canvasRef} width={700} height={480}
        style={{ width: '100%', height: 480, cursor: dragging.current ? 'grabbing' : 'grab', borderRadius: 12, background: 'var(--bg-primary)' }}
        onMouseMove={e => {
          const rect = e.currentTarget.getBoundingClientRect();
          const mx = (e.clientX - rect.left) * (700 / rect.width);
          const my = (e.clientY - rect.top)  * (480 / rect.height);
          if (dragging.current) {
            const dx = mx - lastMouse.current.x, dy = my - lastMouse.current.y;
            setPan(p => ({ x: p.x + dx * (700 / rect.width), y: p.y + dy * (480 / rect.height) }));
            lastMouse.current = { x: mx, y: my };
          } else {
            const n = getNodeAt(mx, my);
            setTooltip(n ? { node: n, x: e.clientX - rect.left, y: e.clientY - rect.top } : null);
          }
        }}
        onMouseDown={e => {
          const rect = e.currentTarget.getBoundingClientRect();
          dragging.current = true;
          lastMouse.current = {
            x: (e.clientX - rect.left) * (700 / rect.width),
            y: (e.clientY - rect.top)  * (480 / rect.height)
          };
        }}
        onMouseUp={() => { dragging.current = false; }}
        onWheel={e => setZoom(z => Math.max(0.3, Math.min(4, z - e.deltaY * 0.001)))}
      />
      {/* Controls */}
      <div style={{ position: 'absolute', top: 12, right: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
        {[
          [ZoomIn, () => setZoom(z => Math.min(4, z + 0.2))],
          [ZoomOut, () => setZoom(z => Math.max(0.3, z - 0.2))],
          [Maximize2, () => { setZoom(1); setPan({ x: 0, y: 0 }); }],
        ].map(([Icon, fn], i) => (
          <button key={i} onClick={fn}
            style={{ width: 30, height: 30, borderRadius: 6, background: 'color-mix(in srgb, var(--bg-card) 90%, transparent)', border: '1px solid var(--border)', color: 'var(--text-secondary)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Icon size={14}/>
          </button>
        ))}
      </div>
      {/* Tooltip */}
      {tooltip && (
        <div style={{
          position: 'absolute', left: tooltip.x + 12, top: tooltip.y - 12,
          background: 'var(--bg-card)', border: '1px solid var(--border-light)',
          borderRadius: 8, padding: '8px 12px', fontSize: 12, pointerEvents: 'none',
          boxShadow: '0 8px 24px color-mix(in srgb, var(--text-primary) 40%, transparent)', zIndex: 10
        }}>
          <div style={{ fontWeight: 700, color: 'var(--text-primary)' }}>
            {tooltip.node.label ? `Txn ${tooltip.node.label}` : `Node ${tooltip.node.id}`}
          </div>
          <div style={{ color: 'var(--text-muted)', marginTop: 2 }}>Node Index: {tooltip.node.id}</div>
          <div style={{ color: tooltip.node.type === 'fraud' ? 'var(--status-critical)' : 'var(--status-good)', marginTop: 2 }}>
            {(tooltip.node.fraud_prob * 100).toFixed(1)}% fraud prob
          </div>
          <div style={{ color: 'var(--text-muted)', marginTop: 2 }}>{tooltip.node.device}</div>
        </div>
      )}
      {/* Legend */}
      <div style={{ position: 'absolute', bottom: 12, left: 12, display: 'flex', gap: 14, background: 'color-mix(in srgb, var(--bg-primary) 85%, transparent)', padding: '6px 12px', borderRadius: 8, border: '1px solid var(--border)' }}>
        {[['🟡 Target','var(--status-warning)'],['🔴 Fraud','var(--status-critical)'],['🟢 Legit','var(--status-good)']].map(([l,c])=>(
          <span key={l} style={{ fontSize: 11, color: c, fontWeight: 600 }}>{l}</span>
        ))}
      </div>
    </div>
  );
}

export default function GraphViewer() {
  const { id: paramId } = useParams();
  const { notify } = useNotification();
  const navigate   = useNavigate();
  const [nodeId,   setNodeId]   = useState(paramId || '');
  const [depth,    setDepth]    = useState(1);
  const [loading,  setLoading]  = useState(false);
  const [graphData,setGraphData]= useState(null);

  const fetchGraph = useCallback(async (nid, d) => {
    const idToLoad = String(nid ?? '').trim();
    console.log('[GRAPH UI] Transaction selected:', idToLoad);
    if (!idToLoad) return notify.error('Enter a Transaction ID or Node ID');
    setLoading(true); setGraphData(null);
    try {
      const res = await client.get(`/api/graph/${idToLoad}`, { params: { depth: d } });
      console.log('[GRAPH UI] API response:', res.data);
      console.log('[GRAPH UI] Nodes:', res.data.nodes?.length);
      console.log('[GRAPH UI] Edges:', res.data.edges?.length);
      console.log('[GRAPH UI] Rendering graph');
      setGraphData(res.data);
      notify.info(`Graph loaded: ${res.data.node_count} nodes, ${res.data.edge_count} edges`);
    } catch (e) {
      console.error('[GRAPH UI] Error loading graph:', e);
      notify.error(e.response?.data?.detail || 'Failed to load graph');
    }
    finally { setLoading(false); }
  }, [notify]);

  // Explicit args: the button passes the typed id; the URL effect passes the
  // route id. Reloads when the route id or the depth selector changes.
  const loadGraph = (nid = nodeId, d = depth) => fetchGraph(nid, d);

  useEffect(() => {
    if (paramId) { setNodeId(paramId); fetchGraph(paramId, depth); }
  }, [paramId, depth, fetchGraph]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Graph Viewer</h1>
          <p className="page-subtitle">Interactive transaction neighbourhood visualization</p>
        </div>
      </div>

      <div className="card" style={{ padding: 24 }}>
        <div style={{ display: 'flex', gap: 12, alignItems: 'flex-end', flexWrap: 'wrap' }}>
          <div className="form-group" style={{ flex: 1, minWidth: 200 }}>
            <label className="form-label">Transaction ID or Node ID</label>
            <input className="input" type="number" placeholder="e.g. 2991340 or 4340" value={nodeId}
              onChange={e => {
                console.log('[GRAPH UI] Transaction selected:', e.target.value);
                setNodeId(e.target.value);
              }} onKeyDown={e => e.key === 'Enter' && loadGraph()}/>
          </div>
          <div className="form-group">
            <label className="form-label">Depth</label>
            <select className="input" value={depth} onChange={e => setDepth(parseInt(e.target.value))} style={{ width: 90 }}>
              <option value={1}>1-hop</option>
              <option value={2}>2-hop</option>
            </select>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            {[
              { label: 'Txn 2991340 (Fraud)', val: '2991340' },
              { label: 'Txn 2987000 (Legit)', val: '2987000' },
              { label: 'Node 4340', val: '4340' },
            ].map(item => (
              <button key={item.val} className="btn btn-ghost btn-sm" onClick={() => {
                console.log('[GRAPH UI] Transaction selected:', item.val);
                setNodeId(item.val);
                loadGraph(item.val);
              }}>{item.label}</button>
            ))}
          </div>
          <button className="btn btn-primary" onClick={() => loadGraph()} disabled={loading} style={{ height: 42 }}>
            {loading ? <><div className="spinner" style={{ width: 16, height: 16 }}/> Loading…</> : <><Network size={15}/> Visualize</>}
          </button>
        </div>
      </div>

      {loading && (
        <div className="card" style={{ padding: 48, textAlign: 'center' }}>
          <div className="spinner" style={{ width: 48, height: 48, borderWidth: 3, margin: '0 auto 20px' }}/>
          <div style={{ color: 'var(--text-muted)' }}>Building ego-subgraph…</div>
        </div>
      )}

      {graphData && !loading && (
        <div className="animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          {/* Stats */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 14 }}>
            {[
              ['Target Transaction', graphData.center_transaction_id || graphData.center_node, 'var(--status-warning)'],
              ['Node Index', graphData.center_node, 'var(--brand)'],
              ['Edges', graphData.edge_count, 'var(--cat-4)'],
              ['Fraud Nodes', graphData.nodes?.filter(n => n.type === 'fraud').length, 'var(--status-critical)'],
            ].map(([l, v, c]) => (
              <div key={l} className="card" style={{ padding: '14px 18px', textAlign: 'center' }}>
                <div style={{ fontSize: 22, fontWeight: 800, color: c }}>{v}</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>{l}</div>
              </div>
            ))}
          </div>

          {/* Canvas */}
          <div className="card" style={{ padding: 16 }}>
            <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 12 }}>
              Transaction Network — {graphData.center_transaction_id ? `Txn ${graphData.center_transaction_id}` : `Node ${graphData.center_node}`}
              <span style={{ fontSize: 12, color: 'var(--text-muted)', fontWeight: 400, marginLeft: 8 }}>
                (scroll to zoom · drag to pan)
              </span>
            </div>
            <ForceGraph nodes={graphData.nodes} edges={graphData.edges} centerId={graphData.center_node}/>
          </div>

          {/* Node list */}
          <div className="card" style={{ padding: 0 }}>
            <div style={{ padding: '16px 20px', fontWeight: 700, fontSize: 14, borderBottom: '1px solid var(--border)' }}>
              Neighbour Nodes
            </div>
            <div style={{ maxHeight: 280, overflowY: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr style={{ background: 'var(--bg-secondary)' }}>
                    {['Node ID','Transaction ID','Type','Fraud Prob','Device'].map(h => (
                      <th key={h} style={{ padding: '10px 16px', textAlign: 'left', fontSize: 11, color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {graphData.nodes?.map(n => (
                    <tr key={n.id} style={{ borderBottom: '1px solid var(--border)', cursor: 'pointer' }}
                      onClick={() => navigate(`/transactions/${n.id}`)}>
                      <td style={{ padding: '10px 16px' }}>
                        <span className="mono" style={{ fontSize: 12, color: n.id === graphData.center_node ? 'var(--status-warning)' : 'var(--text-secondary)' }}>
                          {n.id === graphData.center_node ? '★ ' : ''}{n.id}
                        </span>
                      </td>
                      <td style={{ padding: '10px 16px' }}><span className="mono" style={{ fontSize: 12 }}>{n.transaction_id || n.label || n.id}</span></td>
                      <td style={{ padding: '10px 16px' }}>
                        <span className={`badge ${n.type === 'fraud' ? 'badge-fraud' : n.type === 'target' ? 'badge-high' : 'badge-legit'}`}>
                          {n.type.toUpperCase()}
                        </span>
                      </td>
                      <td style={{ padding: '10px 16px', color: n.fraud_prob >= 0.5 ? 'var(--status-critical)' : 'var(--status-good)', fontWeight: 600, fontSize: 13 }}>
                        {(n.fraud_prob * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '10px 16px', fontSize: 12.5, color: 'var(--text-secondary)' }}>{n.device}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
