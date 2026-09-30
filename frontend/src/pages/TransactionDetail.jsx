import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Brain, Network, ShieldAlert, ShieldCheck, CreditCard, Cpu, Globe } from 'lucide-react';
import client from '../api/client';
import SkeletonLoader from '../components/UI/SkeletonLoader';
import { useNotification } from '../context/NotificationContext';
import { riskColor as riskColorOf } from '../lib/risk';


export default function TransactionDetail() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { notify } = useNotification();
  const [tx, setTx] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    client.get(`/api/transactions/${id}`)
      .then(r => setTx(r.data))
      .catch(() => notify.error('Transaction not found'))
      .finally(() => setLoading(false));
  }, [id, notify]);

  if (loading) return <div style={{padding:40}}><SkeletonLoader rows={8}/></div>;
  if (!tx) return <div style={{padding:40,textAlign:'center',color:'var(--text-muted)'}}>Transaction not found</div>;

  const isFraud = tx.is_fraud;
  const riskColor = riskColorOf(tx.risk_level);

  return (
    <div style={{display:'flex',flexDirection:'column',gap:24}}>
      {/* Header */}
      <div style={{display:'flex',alignItems:'center',gap:16}}>
        <button className="btn btn-ghost btn-sm" onClick={()=>navigate(-1)}><ArrowLeft size={14}/> Back</button>
        <div>
          <h1 className="page-title">{tx.transaction_id}</h1>
          <p className="page-subtitle">Node #{tx.node_id} · {tx.timestamp?.slice(0,10)}</p>
        </div>
        <span className={`badge ${isFraud?'badge-fraud':'badge-legit'}`} style={{marginLeft:'auto',padding:'6px 14px',fontSize:13}}>
          <span className={`status-dot ${isFraud?'dot-fraud':'dot-legit'}`}/>
          {isFraud ? 'FRAUD DETECTED' : 'LEGITIMATE'}
        </span>
      </div>

      {/* Hero risk card */}
      <div style={{
        background: isFraud ? 'linear-gradient(135deg,color-mix(in srgb, var(--status-critical) 12%, transparent),color-mix(in srgb, var(--status-critical) 4%, transparent))' : 'linear-gradient(135deg,color-mix(in srgb, var(--status-good) 12%, transparent),color-mix(in srgb, var(--status-good) 4%, transparent))',
        border: `1px solid ${isFraud?'color-mix(in srgb, var(--status-critical) 25%, transparent)':'color-mix(in srgb, var(--status-good) 25%, transparent)'}`,
        borderRadius:'var(--radius-xl)',
        padding: '32px 36px',
        display:'flex', alignItems:'center', gap:32
      }}>
        <div style={{
          width:80,height:80,borderRadius:'50%',
          background: isFraud?'color-mix(in srgb, var(--status-critical) 15%, transparent)':'color-mix(in srgb, var(--status-good) 15%, transparent)',
          display:'flex',alignItems:'center',justifyContent:'center',flexShrink:0
        }}>
          {isFraud ? <ShieldAlert size={36} color="var(--status-critical)"/> : <ShieldCheck size={36} color="var(--status-good)"/>}
        </div>
        <div style={{flex:1}}>
          <div style={{fontSize:13,color:'var(--text-muted)',marginBottom:4,textTransform:'uppercase',letterSpacing:'0.06em'}}>Fraud Probability</div>
          <div style={{fontSize:48,fontWeight:900,letterSpacing:-2,color:isFraud?'var(--status-critical)':'var(--status-good)',lineHeight:1}}>
            {(tx.fraud_probability*100).toFixed(1)}<span style={{fontSize:24}}>%</span>
          </div>
          <div style={{marginTop:12}}>
            <div className="progress-bar" style={{height:8}}>
              <div className="progress-fill" style={{width:`${tx.fraud_probability*100}%`,background:isFraud?'linear-gradient(90deg,var(--status-critical),var(--status-critical))':'linear-gradient(90deg,var(--status-good),var(--status-good))'}}/>
            </div>
          </div>
        </div>
        <div style={{display:'flex',flexDirection:'column',gap:12}}>
          <div style={{textAlign:'right'}}>
            <div style={{fontSize:11,color:'var(--text-muted)',textTransform:'uppercase',letterSpacing:'0.06em'}}>Risk Level</div>
            <div style={{fontSize:18,fontWeight:800,color:riskColor}}>{tx.risk_level}</div>
          </div>
          <div style={{textAlign:'right'}}>
            <div style={{fontSize:11,color:'var(--text-muted)',textTransform:'uppercase',letterSpacing:'0.06em'}}>Confidence</div>
            <div style={{fontSize:18,fontWeight:800,color:'var(--text-primary)'}}>{(tx.confidence*100).toFixed(1)}%</div>
          </div>
          <div style={{textAlign:'right'}}>
            <div style={{fontSize:11,color:'var(--text-muted)',textTransform:'uppercase',letterSpacing:'0.06em'}}>Threshold</div>
            <div style={{fontSize:18,fontWeight:800,color:'var(--text-secondary)'}}>{(tx.threshold*100).toFixed(0)}%</div>
          </div>
        </div>
      </div>

      {/* Details grid */}
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:20}}>
        {/* Transaction info */}
        <div className="card" style={{padding:24}}>
          <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:20}}>
            <CreditCard size={18} color="var(--brand)"/>
            <div style={{fontWeight:700,fontSize:14}}>Transaction Details</div>
          </div>
          <div style={{display:'flex',flexDirection:'column',gap:14}}>
            {[
              ['Transaction ID', tx.transaction_id],
              ['Amount',         `$${tx.amount?.toFixed(2)}`],
              ['Device',         tx.device],
              ['Email Domain',   tx.email_domain],
              ['Card Type',      tx.card_type],
              ['Product Category', tx.product_cat],
              ['Graph Degree',   `${tx.degree} neighbours`],
            ].map(([k,v]) => (
              <div key={k} style={{display:'flex',justifyContent:'space-between',paddingBottom:12,borderBottom:'1px solid var(--border)'}}>
                <span style={{fontSize:12.5,color:'var(--text-muted)'}}>{k}</span>
                <span style={{fontSize:13,fontWeight:600,color:'var(--text-primary)'}}>{v}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Feature vector */}
        <div className="card" style={{padding:24}}>
          <div style={{display:'flex',alignItems:'center',gap:10,marginBottom:20}}>
            <Cpu size={18} color="var(--cat-4)"/>
            <div style={{fontWeight:700,fontSize:14}}>Feature Vector (top 20)</div>
          </div>
          <div style={{display:'flex',flexDirection:'column',gap:8,maxHeight:320,overflowY:'auto'}}>
            {tx.feature_vector?.map((v,i) => (
              <div key={i} style={{display:'flex',alignItems:'center',gap:10}}>
                <span className="mono" style={{fontSize:11,color:'var(--text-muted)',width:24,flexShrink:0}}>f{i}</span>
                <div style={{flex:1}}>
                  <div className="progress-bar" style={{height:5}}>
                    <div className="progress-fill" style={{width:`${Math.min(100,Math.abs(v)*50+5)}%`,background:'var(--brand)'}}/>
                  </div>
                </div>
                <span className="mono" style={{fontSize:11,color:'var(--text-secondary)',width:60,textAlign:'right'}}>{v.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Action buttons */}
      <div style={{display:'flex',gap:12}}>
        <button className="btn btn-primary" onClick={()=>navigate(`/explain/${tx.node_id}`)}>
          <Brain size={15}/> Explain with GNNExplainer
        </button>
        <button className="btn btn-ghost" onClick={()=>navigate(`/graph/${tx.node_id}`)}>
          <Network size={15}/> View Transaction Graph
        </button>
        {tx.neighbours?.length > 0 && (
          <div style={{marginLeft:'auto',display:'flex',alignItems:'center',gap:8}}>
            <Globe size={15} style={{color:'var(--text-muted)'}}/>
            <span style={{fontSize:12.5,color:'var(--text-muted)'}}>{tx.neighbours.length} connected nodes</span>
          </div>
        )}
      </div>
    </div>
  );
}
