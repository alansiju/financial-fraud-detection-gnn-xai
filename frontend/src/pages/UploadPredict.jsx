import React, { useState, useRef } from 'react';
import { Upload, FileText, AlertTriangle, CheckCircle, Activity, Download, ChevronDown, ChevronUp, Network } from 'lucide-react';
import api from '../api/client';
import { useNotification } from '../context/NotificationContext';

export default function UploadPredict() {
  const [file, setFile] = useState(null);
  const [threshold, setThreshold] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState('');
  const [result, setResult] = useState(null);
  const [expandedRow, setExpandedRow] = useState(null);
  const [page, setPage] = useState(1);
  const ROWS_PER_PAGE = 50;
  
  const fileInputRef = useRef(null);
  const { notify } = useNotification();  // notify.success / notify.error

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile && droppedFile.name.endsWith('.csv')) {
      console.log('[SCANNER UI] File selected');
      console.log('[SCANNER UI] File name:', droppedFile.name);
      console.log('[SCANNER UI] Threshold:', threshold);
      setFile(droppedFile);
      setResult(null);
    } else {
      notify.error('Only CSV files are accepted');
    }
  };

  const handleFileSelect = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      console.log('[SCANNER UI] File selected');
      console.log('[SCANNER UI] File name:', selectedFile.name);
      console.log('[SCANNER UI] Threshold:', threshold);
      setFile(selectedFile);
      setResult(null);
    }
  };

  const handleScan = async () => {
    console.log('[SCANNER UI] Start Full Scan clicked');
    console.log('[SCANNER UI] File name:', file?.name);
    console.log('[SCANNER UI] Threshold:', threshold);

    if (!file) {
      console.warn('[SCANNER UI] No file selected — aborting.');
      return;
    }

    // Guard against locale-comma threshold (e.g. "0,46" → NaN)
    const safeThreshold = isNaN(threshold) || threshold <= 0 || threshold >= 1 ? 0.5 : threshold;

    setLoading(true);
    setStatusText('Uploading & Processing Data...');
    setResult(null);
    setPage(1);

    const formData = new FormData();
    formData.append('file', file);

    // Add threshold to URL as query param
    try {
      console.log('[SCANNER UI] Sending request');
      const res = await api.post(
        `/api/upload-predict?threshold=${safeThreshold}`,
        formData,
        {
          timeout: 300000,   // 5 minutes — large CSV scans take 60-120 s
        }
      );

      console.log('[SCANNER UI] Response status:', res.status);
      console.log('[SCANNER UI] Response received', res.data);
      console.log('[SCANNER UI] Result parsed', res.data);
      setResult(res.data);
      notify.success('Scan completed successfully!');
    } catch (err) {
      const errorMsg =
        err.response?.data?.detail?.message ||
        err.response?.data?.detail ||
        err.message;
      console.error('[SCANNER UI] Scan error:', err);
      notify.error(typeof errorMsg === 'string' ? errorMsg : 'Scan failed');

      if (err.response?.data?.detail?.errors) {
        setResult({
          isError: true,
          errors: err.response.data.detail.errors,
          warnings: err.response.data.detail.warnings,
        });
      }
    } finally {
      setLoading(false);
    }
  };

  const downloadSample = async () => {
    try {
      const res = await api.get('/api/upload-predict/sample', { responseType: 'blob' });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'sample_transactions.csv');
      document.body.appendChild(link);
      link.click();
      link.parentNode.removeChild(link);
    } catch {
      notify.error('Failed to download sample file');
    }
  };

  const downloadResults = () => {
    if (!result || !result.results) return;
    
    // Convert results array to CSV
    const headers = ['Row', 'TransactionID', 'Probability', 'Prediction', 'RiskLevel', 'TopFeatures'];
    const csvRows = [headers.join(',')];
    
    for (const r of result.results) {
      csvRows.push([
        r.row_index,
        r.transaction_id,
        r.fraud_probability,
        r.is_fraud ? 'FRAUD' : 'LEGIT',
        r.risk_level,
        `"${r.top_features.join(', ')}"`
      ].join(','));
    }
    
    const csvData = csvRows.join('\n');
    const blob = new Blob([csvData], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `predictions_${file.name}`);
    document.body.appendChild(link);
    link.click();
    link.parentNode.removeChild(link);
  };

  return (
    <div className="animate-fade-in" style={{ maxWidth: 1200, margin: '0 auto' }}>
      <div className="page-header">
        <div>
          <h1 className="page-title">Dataset Scanner</h1>
          <p className="page-subtitle">Upload raw transaction datasets for automatic preprocessing, graph construction, and GraphSAGE inference.</p>
        </div>
        <div style={{ display: 'flex', gap: 10 }}>
          <button type="button" className="btn btn-ghost" onClick={downloadSample}>
            <Download size={16} /> Sample CSV
          </button>
          {result && result.success && (
            <button type="button" className="btn btn-primary" onClick={downloadResults}>
              <Download size={16} /> Export Results
            </button>
          )}
        </div>
      </div>

      {!result && !loading && (
        <div className="card" style={{ padding: 32 }}>
          <div 
            className="upload-zone"
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept=".csv"
              onChange={handleFileSelect}
            />
            
            <Upload size={48} color="var(--accent-blue)" style={{ marginBottom: 16 }} />
            
            {file ? (
              <div>
                <h3 style={{ fontSize: 18, marginBottom: 8 }}>{file.name}</h3>
                <p className="text-muted">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                <div style={{ marginTop: 24 }} onClick={e => e.stopPropagation()}>
                  <div className="form-group" style={{ maxWidth: 200, margin: '0 auto 24px', textAlign: 'left' }}>
                    <label className="form-label">Fraud Threshold</label>
                    <input 
                      type="number" 
                      className="input" 
                      min="0.01" max="0.99" step="0.05"
                      value={threshold} 
                      onChange={e => {
                        // Replace any locale comma with a dot before parsing
                        const raw = e.target.value.replace(',', '.');
                        const parsed = parseFloat(raw);
                        if (!isNaN(parsed)) setThreshold(parsed);
                      }}
                    />
                  </div>
                  <button type="button" className="btn btn-primary btn-lg" onClick={handleScan} style={{ width: '100%', maxWidth: 240 }}>
                    <Activity size={18} /> Start Full Scan
                  </button>
                </div>
              </div>
            ) : (
              <div>
                <h3 style={{ fontSize: 18, marginBottom: 8 }}>Drag & drop transaction CSV here</h3>
                <p className="text-muted">or click to browse from your computer</p>
                <div style={{ marginTop: 24, padding: '16px', background: 'var(--bg-primary)', borderRadius: 8, display: 'inline-block', textAlign: 'left' }}>
                  <p style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)', marginBottom: 8 }}>REQUIRED FORMAT:</p>
                  <ul style={{ fontSize: 12, color: 'var(--text-muted)', paddingLeft: 16, display: 'flex', flexDirection: 'column', gap: 4 }}>
                    <li>Standard CSV file with header row</li>
                    <li>IEEE-CIS column names (TransactionAmt, card1, etc.)</li>
                    <li>Max 10,000 rows per file</li>
                    <li>Missing columns will be automatically imputed</li>
                  </ul>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {loading && (
        <div className="card" style={{ padding: 64, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <div className="spinner" style={{ width: 48, height: 48, borderTopColor: 'var(--accent-blue)', borderWidth: 4, marginBottom: 24 }}></div>
          <h2 style={{ fontSize: 20, marginBottom: 8 }}>{statusText}</h2>
          <p className="text-muted">Preprocessing features, building graph, and running GraphSAGE inference...</p>
        </div>
      )}

      {result?.isError && (
        <div className="card" style={{ padding: 32 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, color: 'var(--accent-red)', marginBottom: 20 }}>
            <AlertTriangle size={32} />
            <h2 style={{ fontSize: 20 }}>Validation Failed</h2>
          </div>
          <button type="button" className="btn btn-ghost" onClick={() => { setResult(null); setFile(null); }} style={{ marginBottom: 24 }}>
            Try Another File
          </button>
          
          <div style={{ display: 'grid', gap: 24 }}>
            {result.errors?.length > 0 && (
              <div>
                <h3 style={{ fontSize: 14, color: 'var(--accent-red)', marginBottom: 12, textTransform: 'uppercase' }}>Errors</h3>
                <ul style={{ paddingLeft: 20, color: 'var(--text-primary)', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {result.errors.map((err, i) => <li key={i}>{err}</li>)}
                </ul>
              </div>
            )}
            
            {result.warnings?.length > 0 && (
              <div>
                <h3 style={{ fontSize: 14, color: 'var(--accent-orange)', marginBottom: 12, textTransform: 'uppercase' }}>Warnings</h3>
                <ul style={{ paddingLeft: 20, color: 'var(--text-secondary)', display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {result.warnings.map((warn, i) => <li key={i}>{warn}</li>)}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {result?.success && (
        <div className="animate-slide-up">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <CheckCircle size={28} color="var(--accent-green)" />
              <div>
                <h2 style={{ fontSize: 18 }}>Scan Complete</h2>
                <p className="text-muted" style={{ fontSize: 13 }}>Processed {result.total_rows} rows in {result.processing_time_ms}ms</p>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 12 }}>
              <button type="button" className="btn btn-primary" onClick={() => setResult(null)}>
                Adjust Threshold & Re-scan
              </button>
              <button type="button" className="btn btn-ghost" onClick={() => { setResult(null); setFile(null); }}>
                Scan New File
              </button>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 20, marginBottom: 24 }}>
            <div className="kpi-card" style={{ '--accent-color': 'var(--text-secondary)' }}>
              <div className="kpi-label">Total Transactions</div>
              <div className="kpi-value">{result.total_rows.toLocaleString()}</div>
              <div className="kpi-change"><FileText size={14}/> Graph Nodes: {result.graph_stats.nodes}</div>
            </div>
            <div className="kpi-card" style={{ '--accent-color': 'var(--accent-red)' }}>
              <div className="kpi-label">Fraud Detected</div>
              <div className="kpi-value text-fraud">{result.fraud_count.toLocaleString()}</div>
              <div className="kpi-change up">{(result.fraud_rate).toFixed(1)}% fraud rate</div>
            </div>
            <div className="kpi-card" style={{ '--accent-color': 'var(--accent-green)' }}>
              <div className="kpi-label">Legitimate</div>
              <div className="kpi-value text-legit">{result.legit_count.toLocaleString()}</div>
              <div className="kpi-change">Threshold: {result.threshold}</div>
            </div>
            <div className="kpi-card" style={{ '--accent-color': 'var(--accent-purple)' }}>
              <div className="kpi-label">Graph Edges</div>
              <div className="kpi-value">{(result.graph_stats.cross_edges || 0).toLocaleString()}</div>
              <div className="kpi-change">
                <Network size={14}/>
                {result.graph_stats.mode === 'attached' 
                  ? ` Cross: ${(result.graph_stats.cross_edges || 0).toLocaleString()} · Intra: ${(result.graph_stats.intra_edges || 0).toLocaleString()}`
                  : ` ${result.graph_stats.edges.toLocaleString()} intra-batch`
                }
              </div>
            </div>
          </div>

          {result.graph_attachment && (
            <div style={{ background: 'color-mix(in srgb, var(--cat-2) 8%, transparent)', border: '1px solid color-mix(in srgb, var(--cat-2) 20%, transparent)', padding: '12px 16px', borderRadius: 8, marginBottom: 16, fontSize: 13, color: 'var(--text-secondary)' }}>
              <strong style={{ color: 'var(--accent-purple)' }}>Graph Attachment Active:</strong>{' '}
              {result.graph_attachment.nodes_with_training_neighbors} of {result.total_rows} transactions connected to training graph
              {' · '}{result.graph_attachment.cross_edges.toLocaleString()} cross-edges
              {' · '}{result.graph_attachment.nodes_without_training_neighbors} isolated
            </div>
          )}

          {result.validation?.warnings?.length > 0 && (
            <div style={{ background: 'color-mix(in srgb, var(--status-warning) 10%, transparent)', border: '1px solid color-mix(in srgb, var(--status-warning) 20%, transparent)', padding: '12px 16px', borderRadius: 8, marginBottom: 24, fontSize: 13, color: 'var(--status-warning)' }}>
              <strong>Note:</strong> {result.validation.warnings[0]} {result.validation.warnings.length > 1 ? `(+${result.validation.warnings.length - 1} more warnings)` : ''}
            </div>
          )}

          <div className="card">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Row</th>
                    <th>Transaction ID</th>
                    <th>Probability</th>
                    <th>Risk Level</th>
                    <th>Prediction</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.slice((page - 1) * ROWS_PER_PAGE, page * ROWS_PER_PAGE).map((row) => (
                    <React.Fragment key={row.row_index}>
                      <tr 
                        className={row.is_fraud ? 'tr-fraud' : ''}
                        style={{ cursor: 'pointer' }}
                        onClick={() => setExpandedRow(expandedRow === row.row_index ? null : row.row_index)}
                      >
                        <td className="text-muted">#{row.row_index + 1}</td>
                        <td className="mono">{row.transaction_id}</td>
                        <td className="mono">{(row.fraud_probability * 100).toFixed(2)}%</td>
                        <td>
                          <span className={`badge badge-${row.risk_level.toLowerCase()}`}>
                            {row.risk_level}
                          </span>
                        </td>
                        <td>
                          <span className={row.is_fraud ? 'text-fraud font-bold' : 'text-legit font-bold'} style={{ fontWeight: 600 }}>
                            {row.is_fraud ? 'FRAUD' : 'LEGITIMATE'}
                          </span>
                        </td>
                        <td style={{ textAlign: 'right', color: 'var(--text-muted)' }}>
                          {expandedRow === row.row_index ? <ChevronUp size={16}/> : <ChevronDown size={16}/>}
                        </td>
                      </tr>
                      {expandedRow === row.row_index && (
                        <tr style={{ background: 'var(--bg-secondary)' }}>
                          <td colSpan={6} style={{ padding: '20px 24px' }}>
                            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 32 }}>
                              <div>
                                <h4 style={{ fontSize: 12, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 12 }}>Model Insights</h4>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <span className="text-secondary">Confidence</span>
                                    <span className="mono">{(row.confidence * 100).toFixed(1)}%</span>
                                  </div>
                                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <span className="text-secondary">Threshold Used</span>
                                    <span className="mono">{result.threshold}</span>
                                  </div>
                                  <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                    <span className="text-secondary">Training Neighbors</span>
                                    <span className="mono" style={{ color: (row.training_neighbors || 0) > 0 ? 'var(--accent-green)' : 'var(--text-muted)' }}>
                                      {row.training_neighbors != null ? row.training_neighbors : 'N/A'}
                                    </span>
                                  </div>
                                </div>
                              </div>
                              <div>
                                <h4 style={{ fontSize: 12, textTransform: 'uppercase', color: 'var(--text-muted)', marginBottom: 12 }}>Feature Deviation Explanation</h4>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                                  {row.top_features.map((feat, i) => (
                                    <span key={i} style={{ 
                                      background: 'var(--bg-card)', border: '1px solid var(--border)', 
                                      padding: '4px 10px', borderRadius: 4, fontSize: 12, fontFamily: 'monospace' 
                                    }}>
                                      {feat}
                                    </span>
                                  ))}
                                </div>
                                <p style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 8 }}>
                                  Feature Deviation Explanation: Shows highest statistical absolute deviation from training mean (Z-Score). Not GNNExplainer.
                                </p>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
            
            {result.results.length > ROWS_PER_PAGE && (
              <div className="pagination" style={{ padding: '20px', borderTop: '1px solid var(--border)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="text-muted" style={{ fontSize: 13 }}>
                  Showing {(page - 1) * ROWS_PER_PAGE + 1} to {Math.min(page * ROWS_PER_PAGE, result.results.length)} of {result.results.length} rows
                </span>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button type="button"
                    className="btn btn-ghost btn-sm" 
                    disabled={page === 1}
                    onClick={() => setPage(p => p - 1)}
                  >
                    Previous
                  </button>
                  <button type="button"
                    className="btn btn-ghost btn-sm" 
                    disabled={page >= Math.ceil(result.results.length / ROWS_PER_PAGE)}
                    onClick={() => setPage(p => p + 1)}
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
