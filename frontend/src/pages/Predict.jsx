import { useState, useRef } from 'react';
import { Zap, Upload, AlertTriangle, CheckCircle, Eye } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import client from '../api/client';
import { useNotification } from '../context/NotificationContext';
import { riskClass, riskColor } from '../lib/risk';

function RiskGauge({ prob, level }) {
  const angle = prob * 180;
  const cx = 100, cy = 100;
  const rad = (a) => (a - 90) * Math.PI / 180;
  // Semicircle gauge
  // Colour from the risk level the backend assigned, not fixed 0.8/0.6/0.4
  // cut-offs — those ignored the decision threshold, so the gauge could
  // disagree with the label printed beside it.
  const color = riskColor(level);
  return (
    <div style={{textAlign:'center'}}>
      <svg width="200" height="120" viewBox="0 0 200 120">
        {/* Background arc */}
        <path d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="var(--border)" strokeWidth="12" strokeLinecap="round"/>
        {/* Colored arc */}
        <path
          d={`M 20 100 A 80 80 0 ${angle > 90 ? 1 : 0} 1 ${20 + (160*(prob))} ${100 - 80*Math.sin((prob)*Math.PI)}`}
          fill="none" stroke={color} strokeWidth="12" strokeLinecap="round"
          style={{transition:'all 600ms cubic-bezier(0.4,0,0.2,1)'}}
        />
        {/* Needle */}
        <line
          x1={cx} y1={cy}
          x2={cx + 70*Math.cos(rad(prob*180 - 90))}
          y2={cy + 70*Math.sin(rad(prob*180 - 90))}
          stroke={color} strokeWidth="2.5" strokeLinecap="round"
          style={{transition:'all 600ms cubic-bezier(0.4,0,0.2,1)'}}
        />
        <circle cx={cx} cy={cy} r="6" fill={color}/>
        {/* Labels */}
        <text x="18" y="116" fill="var(--text-muted)" fontSize="9" fontFamily="Inter">0%</text>
        <text x="88" y="25" fill="var(--text-muted)" fontSize="9" fontFamily="Inter">50%</text>
        <text x="172" y="116" fill="var(--text-muted)" fontSize="9" fontFamily="Inter">100%</text>
      </svg>
      <div style={{fontSize:36,fontWeight:900,color,letterSpacing:-1,marginTop:-16}}>
        {(prob*100).toFixed(1)}<span style={{fontSize:18}}>%</span>
      </div>
      <div style={{fontSize:13,color:'var(--text-muted)',marginTop:4}}>Fraud Probability</div>
    </div>
  );
}

export default function Predict() {
  const { notify } = useNotification();
  const navigate   = useNavigate();
  const fileRef    = useRef();
  const [tab,      setTab]    = useState('manual'); // manual | csv
  const [nodeId,   setNodeId] = useState('');
  const [threshold,setThres]  = useState(0.5);
  const [loading,  setLoading]= useState(false);
  const [result,   setResult] = useState(null);
  const [csvResult,setCsvResult]=useState(null);
  const [file,     setFile]   = useState(null);
  const [drag,     setDrag]   = useState(false);

  const handlePredict = async () => {
    if (!nodeId) return notify.error('Please enter a Transaction ID or Node ID');
    setLoading(true); setResult(null);
    try {
      const res = await client.post('/api/predict/node', { node_id: parseInt(nodeId), threshold });
      setResult(res.data);
      const txDisplay = res.data.transaction_id ? `Txn ${res.data.transaction_id}` : `Node ${res.data.node_id}`;
      if (res.data.is_fraud) notify.warning(`🚨 Fraud detected! ${txDisplay} — ${(res.data.fraud_probability*100).toFixed(1)}% probability`);
      else notify.success(`✅ Legitimate transaction — ${(res.data.confidence*100).toFixed(1)}% confidence`);
    } catch (e) { notify.error(e.response?.data?.detail || 'Prediction failed'); }
    finally { setLoading(false); }
  };

  const handleCSV = async () => {
    if (!file) return notify.error('Please upload a CSV file');
    setLoading(true); setCsvResult(null);
    try {
      const form = new FormData();
      form.append('file', file);
      const res = await client.post(`/api/predict/upload?threshold=${threshold}`, form);
      setCsvResult(res.data);
      notify.success(`CSV processed: ${res.data.fraud_count} fraud detected out of ${res.data.total}`);
    } catch (e) { notify.error(e.response?.data?.detail || 'CSV upload failed'); }
    finally { setLoading(false); }
  };

  const onDrop = (e) => {
    e.preventDefault(); setDrag(false);
    const f = e.dataTransfer.files[0];
    if (f?.name.endsWith('.csv')) setFile(f);
    else notify.error('Only CSV files accepted');
  };

  return (
    <div style={{display:'flex',flexDirection:'column',gap:24}}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Fraud Prediction</h1>
          <p className="page-subtitle">Run GraphSAGE inference on individual transactions or upload a CSV batch</p>
        </div>
      </div>

      {/* Tabs */}
      <div style={{display:'flex',gap:0,borderRadius:10,overflow:'hidden',border:'1px solid var(--border)',width:'fit-content'}}>
        {[['manual','🎯 Manual Entry'],['csv','📂 CSV Upload']].map(([t,l])=>(
          <button key={t} onClick={()=>setTab(t)}
            style={{padding:'10px 24px',fontSize:13.5,fontWeight:600,cursor:'pointer',border:'none',
              background:tab===t?'var(--accent-blue)':'transparent',
              color:tab===t?'var(--on-brand)':'var(--text-muted)'}}>
            {l}
          </button>
        ))}
      </div>

      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:24}}>
        {/* Left: input */}
        <div style={{display:'flex',flexDirection:'column',gap:16}}>
          <div className="card" style={{padding:28}}>
            <div style={{fontWeight:700,fontSize:15,marginBottom:20}}>
              {tab === 'manual' ? '🎯 Enter Transaction or Node ID' : '📂 Upload CSV File'}
            </div>

            {tab === 'manual' ? (
              <div style={{display:'flex',flexDirection:'column',gap:16}}>
                <div className="form-group">
                  <label className="form-label">Transaction ID or Node Index</label>
                  <input className="input" type="number" placeholder="e.g. 2991340 or 4340" value={nodeId}
                    onChange={e=>setNodeId(e.target.value)}
                    onKeyDown={e=>e.key==='Enter'&&handlePredict()}/>
                  <span style={{fontSize:11.5,color:'var(--text-muted)',marginTop:4}}>
                    Supports IEEE-CIS TransactionID (e.g. 2991340) or graph node index (0 – 590,539)
                  </span>
                </div>

                {/* Quick samples */}
                <div>
                  <div style={{fontSize:12,color:'var(--text-muted)',marginBottom:8}}>Quick samples:</div>
                  <div style={{display:'flex',gap:6,flexWrap:'wrap'}}>
                    {[
                      ['Txn 2991340', '2991340'],
                      ['Txn 2987000', '2987000'],
                      ['Node 4340', '4340'],
                      ['Node 7006', '7006'],
                      ['Node 1024', '1024'],
                    ].map(([lbl, val])=>(
                      <button key={val} className="btn btn-ghost btn-sm" onClick={()=>setNodeId(val)}>{lbl}</button>
                    ))}
                  </div>
                </div>

                <div className="form-group">
                  <label className="form-label">Detection Threshold: {(threshold*100).toFixed(0)}%</label>
                  <input type="range" min={0.01} max={0.99} step={0.01} value={threshold}
                    onChange={e=>setThres(parseFloat(e.target.value))}
                    style={{width:'100%',accentColor:'var(--brand)'}}/>
                  <div style={{display:'flex',justifyContent:'space-between',fontSize:11,color:'var(--text-muted)'}}>
                    <span>1% (strict)</span><span>50% (default)</span><span>99% (lenient)</span>
                  </div>
                </div>

                <button className="btn btn-primary" onClick={handlePredict} disabled={loading} style={{height:44}}>
                  {loading ? <><div className="spinner" style={{width:16,height:16}}/> Analyzing…</> : <><Zap size={15}/> Run Prediction</>}
                </button>
              </div>
            ) : (
              <div style={{display:'flex',flexDirection:'column',gap:16}}>
                <div
                  className={`upload-zone ${drag?'dragover':''}`}
                  onClick={()=>fileRef.current.click()}
                  onDragOver={e=>{e.preventDefault();setDrag(true);}}
                  onDragLeave={()=>setDrag(false)}
                  onDrop={onDrop}
                >
                  <Upload size={36} style={{color:'var(--text-muted)',margin:'0 auto 12px'}}/>
                  <div style={{fontWeight:600,marginBottom:6}}>Drop CSV here or click to browse</div>
                  <div style={{fontSize:12.5,color:'var(--text-muted)'}}>Must contain a <code style={{background:'var(--bg-card)',padding:'1px 5px',borderRadius:4}}>TransactionID</code> or <code style={{background:'var(--bg-card)',padding:'1px 5px',borderRadius:4}}>node_id</code> column</div>
                  {file && <div style={{marginTop:12,color:'var(--status-good)',fontWeight:600,fontSize:13}}>✓ {file.name}</div>}
                </div>
                <input ref={fileRef} type="file" accept=".csv" style={{display:'none'}} onChange={e=>setFile(e.target.files[0])}/>

                <div className="form-group">
                  <label className="form-label">Threshold: {(threshold*100).toFixed(0)}%</label>
                  <input type="range" min={0.01} max={0.99} step={0.01} value={threshold}
                    onChange={e=>setThres(parseFloat(e.target.value))} style={{width:'100%',accentColor:'var(--brand)'}}/>
                </div>

                <div style={{
                  padding:'14px 16px',borderRadius:10,
                  background:'color-mix(in srgb, var(--brand) 6%, transparent)',border:'1px solid color-mix(in srgb, var(--brand) 15%, transparent)',
                  fontSize:12.5,color:'var(--text-secondary)'
                }}>
                  <div style={{fontWeight:600,color:'var(--brand)',marginBottom:4}}>Expected CSV format:</div>
                  <code className="mono" style={{display:'block',padding:'8px 10px',background:'var(--bg-primary)',borderRadius:6}}>
                    TransactionID{'\n'}2991340{'\n'}2987000{'\n'}# or node_id{'\n'}4340{'\n'}...
                  </code>
                </div>

                <button className="btn btn-primary" onClick={handleCSV} disabled={loading||!file} style={{height:44}}>
                  {loading ? <><div className="spinner" style={{width:16,height:16}}/> Processing…</> : <><Upload size={15}/> Upload & Predict</>}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Right: results */}
        <div style={{display:'flex',flexDirection:'column',gap:16}}>
          {/* Manual result */}
          {tab === 'manual' && (
            <div className="card" style={{padding:28}}>
              {!result && !loading && (
                <div style={{textAlign:'center',padding:'40px 0',color:'var(--text-muted)'}}>
                  <Zap size={40} style={{opacity:0.2,margin:'0 auto 12px'}}/>
                  <div style={{fontSize:14}}>Enter a Transaction ID or node ID and run prediction</div>
                </div>
              )}
              {loading && (
                <div style={{textAlign:'center',padding:'40px 0'}}>
                  <div className="spinner" style={{width:40,height:40,borderWidth:3,margin:'0 auto 16px'}}/>
                  <div style={{color:'var(--text-muted)',fontSize:14}}>Running GraphSAGE inference…</div>
                </div>
              )}
              {result && !loading && (
                <div className="animate-fade-in" style={{display:'flex',flexDirection:'column',gap:20}}>
                  <RiskGauge prob={result.fraud_probability} level={result.risk_level}/>
                  <div style={{
                    padding:'16px 20px',borderRadius:12,
                    background: result.is_fraud ? 'color-mix(in srgb, var(--status-critical) 8%, transparent)' : 'color-mix(in srgb, var(--status-good) 8%, transparent)',
                    border: `1px solid ${result.is_fraud ? 'color-mix(in srgb, var(--status-critical) 20%, transparent)' : 'color-mix(in srgb, var(--status-good) 20%, transparent)'}`,
                    display:'flex',alignItems:'center',gap:12
                  }}>
                    {result.is_fraud
                      ? <AlertTriangle size={20} color="var(--status-critical)"/>
                      : <CheckCircle size={20} color="var(--status-good)"/>}
                    <div>
                      <div style={{fontWeight:700,fontSize:15,color:result.is_fraud?'var(--status-critical)':'var(--status-good)'}}>
                        {result.is_fraud ? '🚨 FRAUD DETECTED' : '✅ LEGITIMATE'}
                      </div>
                      <div style={{fontSize:12.5,color:'var(--text-muted)',marginTop:2}}>
                        {result.transaction_id ? `Txn ${result.transaction_id} (Node ${result.node_id})` : `Node ${result.node_id}`} · Risk: {result.risk_level} · Confidence: {(result.confidence*100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  <div style={{display:'flex',gap:8}}>
                    <button className="btn btn-ghost btn-sm" onClick={()=>navigate(`/explain/${result.transaction_id || result.node_id}`)}>
                      🧠 Explain
                    </button>
                    <button className="btn btn-ghost btn-sm" onClick={()=>navigate(`/graph/${result.transaction_id || result.node_id}`)}>
                      🌐 Graph
                    </button>
                    <button className="btn btn-ghost btn-sm" onClick={()=>navigate(`/transactions/${result.transaction_id || result.node_id}`)}>
                      <Eye size={13}/> Details
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* CSV result */}
          {tab === 'csv' && (
            <div className="card" style={{padding:28}}>
              {!csvResult && !loading && (
                <div style={{textAlign:'center',padding:'40px 0',color:'var(--text-muted)'}}>
                  <Upload size={40} style={{opacity:0.2,margin:'0 auto 12px'}}/>
                  <div style={{fontSize:14}}>Upload a CSV to batch predict</div>
                </div>
              )}
              {loading && (
                <div style={{textAlign:'center',padding:'40px 0'}}>
                  <div className="spinner" style={{width:40,height:40,borderWidth:3,margin:'0 auto 16px'}}/>
                  <div style={{color:'var(--text-muted)',fontSize:14}}>Processing CSV batch…</div>
                </div>
              )}
              {csvResult && !loading && (
                <div className="animate-fade-in" style={{display:'flex',flexDirection:'column',gap:16}}>
                  <div style={{fontWeight:700,fontSize:15}}>Batch Results</div>
                  <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:12}}>
                    {[
                      ['Total','total','var(--brand)'],
                      ['Fraud','fraud_count','var(--status-critical)'],
                      ['Legit','legit_count','var(--status-good)'],
                    ].map(([l,k,c])=>(
                      <div key={l} style={{textAlign:'center',padding:'16px 12px',borderRadius:10,background:`color-mix(in srgb, ${c} 7%, transparent)`,border:`1px solid color-mix(in srgb, ${c} 13%, transparent)`}}>
                        <div style={{fontSize:24,fontWeight:800,color:c}}>{csvResult[k]}</div>
                        <div style={{fontSize:11,color:'var(--text-muted)',marginTop:4}}>{l}</div>
                      </div>
                    ))}
                  </div>
                  <div style={{maxHeight:240,overflowY:'auto'}}>
                    <table style={{width:'100%',borderCollapse:'collapse'}}>
                      <thead>
                        <tr style={{background:'var(--bg-secondary)'}}>
                          <th style={{padding:'8px 12px',fontSize:11,textAlign:'left',color:'var(--text-muted)'}}>Node</th>
                          <th style={{padding:'8px 12px',fontSize:11,textAlign:'left',color:'var(--text-muted)'}}>Probability</th>
                          <th style={{padding:'8px 12px',fontSize:11,textAlign:'left',color:'var(--text-muted)'}}>Risk</th>
                        </tr>
                      </thead>
                      <tbody>
                        {csvResult.results?.slice(0,20).map(r=>(
                          <tr key={r.node_id} style={{borderBottom:'1px solid var(--border)'}}>
                            <td style={{padding:'8px 12px',fontSize:12}}>{r.node_id}</td>
                            <td style={{padding:'8px 12px',fontSize:12,color:r.is_fraud?'var(--status-critical)':'var(--status-good)',fontWeight:600}}>{(r.fraud_probability*100).toFixed(1)}%</td>
                            <td style={{padding:'8px 12px'}}><span className={`badge ${riskClass(r.risk_level)}`}>{r.risk_level}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
