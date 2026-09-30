import { useEffect, useState, useCallback } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { Search, ArrowUpDown, ChevronLeft, ChevronRight, Eye, RefreshCw } from 'lucide-react';
import client from '../api/client';
import { useNotification } from '../context/NotificationContext';
import { riskClass } from '../lib/risk';


export default function Transactions() {
  const navigate = useNavigate();
  const [sp] = useSearchParams();
  const { notify } = useNotification();

  const [data,    setData]    = useState(null);
  const [loading, setLoading] = useState(true);
  const [page,    setPage]    = useState(1);
  const [search,  setSearch]  = useState('');
  const [fraud,   setFraud]   = useState(sp.get('fraud') || 'all');
  const [sort,    setSort]    = useState('id');
  const [order,   setOrder]   = useState('asc');
  const [device,  setDevice]  = useState('');
  const [minAmt,  setMinAmt]  = useState('');
  const [maxAmt,  setMaxAmt]  = useState('');
  // Filter options come from the data now. They used to be a hardcoded list
  // ('Mobile','Tablet','POS Terminal','ATM','Web Browser') of categories that
  // do not exist in IEEE-CIS, so every one of them matched zero rows.
  const [facets,  setFacets]  = useState({ devices: [] });

  const LIMIT = 25;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const params = { page, limit: LIMIT, sort, order };
      if (fraud !== 'all') params.fraud = fraud;
      if (search) params.search = search;
      if (device) params.device = device;
      if (minAmt) params.min_amt = minAmt;
      if (maxAmt) params.max_amt = maxAmt;
      const res = await client.get('/api/transactions', { params });
      setData(res.data);
    } catch { notify.error('Failed to load transactions'); }
    finally { setLoading(false); }
  }, [page, search, fraud, sort, order, device, minAmt, maxAmt, notify]);

  useEffect(() => { load(); }, [load]);

  useEffect(() => {
    client.get('/api/transactions/facets')
      .then(res => setFacets(res.data))
      .catch(() => { /* filters degrade to "All"; the table still works */ });
  }, []);

  const toggleSort = (col) => {
    if (sort === col) setOrder(o => o === 'asc' ? 'desc' : 'asc');
    else { setSort(col); setOrder('asc'); }
    setPage(1);
  };

  const SortIcon = ({ col }) => (
    <ArrowUpDown size={12} style={{ opacity: sort === col ? 1 : 0.35, marginLeft: 4 }}/>
  );

  const Pagination = () => {
    if (!data) return null;
    const pages = data.total_pages;
    const buttons = [];
    const start = Math.max(1, page - 2);
    const end   = Math.min(pages, page + 2);
    for (let i = start; i <= end; i++) buttons.push(i);
    return (
      <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'16px 20px', borderTop:'1px solid var(--border)' }}>
        <div style={{ fontSize:13, color:'var(--text-muted)' }}>
          Showing <b style={{color:'var(--text-primary)'}}>{(page-1)*LIMIT+1}–{Math.min(page*LIMIT, data.total)}</b> of <b style={{color:'var(--text-primary)'}}>{data.total?.toLocaleString()}</b> results
        </div>
        <div className="pagination">
          <button className="page-btn" onClick={()=>setPage(p=>p-1)} disabled={page<=1}><ChevronLeft size={14}/></button>
          {start > 1 && <><button className="page-btn" onClick={()=>setPage(1)}>1</button><span style={{color:'var(--text-muted)'}}>…</span></>}
          {buttons.map(b=><button key={b} className={`page-btn${b===page?' active':''}`} onClick={()=>setPage(b)}>{b}</button>)}
          {end < pages && <><span style={{color:'var(--text-muted)'}}>…</span><button className="page-btn" onClick={()=>setPage(pages)}>{pages}</button></>}
          <button className="page-btn" onClick={()=>setPage(p=>p+1)} disabled={page>=pages}><ChevronRight size={14}/></button>
        </div>
      </div>
    );
  };

  return (
    <div style={{ display:'flex', flexDirection:'column', gap:20 }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Transactions</h1>
          <p className="page-subtitle">{data ? `${data.total?.toLocaleString()} transactions` : 'Loading…'} · IEEE-CIS dataset</p>
        </div>
        <button className="btn btn-ghost btn-sm" onClick={load}><RefreshCw size={14}/> Refresh</button>
      </div>

      {/* Filter bar */}
      <div className="filter-bar">
        <div className="input-group" style={{width:220}}>
          <Search className="input-icon" size={14}/>
          <input className="input" placeholder="Search transaction ID…" value={search}
            onChange={e=>{setSearch(e.target.value);setPage(1);}} style={{height:36,paddingTop:0,paddingBottom:0}}/>
        </div>

        {/* Fraud filter tabs */}
        <div style={{display:'flex', borderRadius:8, overflow:'hidden', border:'1px solid var(--border)'}}>
          {['all','fraud','legit'].map(f=>(
            <button key={f} onClick={()=>{setFraud(f);setPage(1);}}
              style={{padding:'7px 14px', fontSize:12.5, fontWeight:600, cursor:'pointer', border:'none',
                background: fraud===f ? (f==='fraud'?'color-mix(in srgb, var(--status-critical) 15%, transparent)':f==='legit'?'color-mix(in srgb, var(--status-good) 15%, transparent)':'var(--accent-blue)') : 'transparent',
                color: fraud===f ? (f==='fraud'?'var(--status-critical)':f==='legit'?'var(--status-good)':'var(--on-brand)') : 'var(--text-muted)',
                textTransform:'capitalize'}}>
              {f === 'all' ? 'All' : f === 'fraud' ? '🔴 Fraud' : '🟢 Legit'}
            </button>
          ))}
        </div>

        <select className="input" value={device} onChange={e=>{setDevice(e.target.value);setPage(1);}} style={{height:36,width:150,paddingTop:0,paddingBottom:0}}>
          <option value="">All Devices</option>
          {(facets.devices ?? []).map(d=>(
            <option key={d.value} value={d.value}>{d.value} ({d.count.toLocaleString()})</option>
          ))}
        </select>
        <input className="input" type="number" placeholder="Min $" value={minAmt} onChange={e=>setMinAmt(e.target.value)} style={{height:36,width:90,paddingTop:0,paddingBottom:0}}/>
        <input className="input" type="number" placeholder="Max $" value={maxAmt} onChange={e=>setMaxAmt(e.target.value)} style={{height:36,width:90,paddingTop:0,paddingBottom:0}}/>
        <button className="btn btn-ghost btn-sm" onClick={()=>{setSearch('');setFraud('all');setDevice('');setMinAmt('');setMaxAmt('');setPage(1);}}>Clear</button>
      </div>

      {/* Table */}
      <div className="card" style={{padding:0}}>
        <div className="table-wrap" style={{border:'none',borderRadius:'inherit'}}>
          <table>
            <thead>
              <tr>
                <th onClick={()=>toggleSort('id')}>Node ID <SortIcon col="id"/></th>
                <th>Transaction ID</th>
                <th onClick={()=>toggleSort('amount')}>Amount <SortIcon col="amount"/></th>
                <th onClick={()=>toggleSort('fraud_probability')}>Fraud Prob <SortIcon col="fraud_probability"/></th>
                <th>Risk Level</th>
                <th>Status</th>
                <th>Device</th>
                <th>Card</th>
                <th>Timestamp</th>
                <th style={{textAlign:'center'}}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                Array(LIMIT).fill(0).map((_, i) => (
                  <tr key={i}>
                    {Array(10).fill(0).map((_, j) => (
                      <td key={j}><div className="skeleton" style={{height:12,borderRadius:6}}/></td>
                    ))}
                  </tr>
                ))
              ) : data?.data?.map(tx => (
                <tr key={tx.id} className={tx.is_fraud ? 'tr-fraud' : ''}>
                  <td><span className="mono" style={{fontSize:12,color:'var(--text-muted)'}}>{tx.id}</span></td>
                  <td><span className="mono" style={{fontWeight:600,color:'var(--text-primary)'}}>{tx.transaction_id}</span></td>
                  <td><span style={{fontWeight:600}}>${tx.amount?.toFixed(2)}</span></td>
                  <td>
                    <div style={{display:'flex',alignItems:'center',gap:8}}>
                      <div style={{flex:1,maxWidth:80}}>
                        <div className="progress-bar"><div className="progress-fill" style={{width:`${tx.fraud_probability*100}%`,background:tx.fraud_probability>=0.5?'var(--status-critical)':'var(--status-good)'}}/></div>
                      </div>
                      <span className="mono" style={{fontSize:12,fontWeight:600,color:tx.fraud_probability>=0.5?'var(--status-critical)':'var(--status-good)'}}>
                        {(tx.fraud_probability*100).toFixed(1)}%
                      </span>
                    </div>
                  </td>
                  <td><span className={`badge ${riskClass(tx.risk_level)}`}>{tx.risk_level}</span></td>
                  <td>
                    <span className={`badge ${tx.is_fraud?'badge-fraud':'badge-legit'}`}>
                      <span className={`status-dot ${tx.is_fraud?'dot-fraud':'dot-legit'}`}/>
                      {tx.is_fraud?'Fraud':'Legitimate'}
                    </span>
                  </td>
                  <td><span style={{fontSize:12,color:'var(--text-secondary)'}}>{tx.device}</span></td>
                  <td><span style={{fontSize:12,color:'var(--text-secondary)'}}>{tx.card_type}</span></td>
                  <td><span style={{fontSize:11.5,color:'var(--text-muted)'}}>{tx.timestamp?.slice(0,10)}</span></td>
                  <td style={{textAlign:'center'}}>
                    <button className="btn btn-ghost btn-sm" onClick={()=>navigate(`/transactions/${tx.id}`)} title="View details">
                      <Eye size={13}/>
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Pagination/>
      </div>
    </div>
  );
}
