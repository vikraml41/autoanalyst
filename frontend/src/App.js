import React, { useState, useEffect } from 'react';
import { Database, TrendingUp, Activity, BarChart3, ChevronRight, RefreshCw } from 'lucide-react';

const API_URL = window.location.hostname === 'localhost' 
  ? 'http://localhost:8000'
  : 'https://autoanalyst-dz11.onrender.com';

function App() {
  // State management
  const [currentTime, setCurrentTime] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState('CHECKING...');
  const [marketConditions, setMarketConditions] = useState({
    regime: 'Loading...',
    fedStance: 'Loading...',
    vix: 'Loading...',
    recessionRisk: 'Loading...'
  });
  const [dataReady, setDataReady] = useState(false);
  const [sectors, setSectors] = useState([]);
  const [subIndustries, setSubIndustries] = useState([]);
  const [analysisType, setAnalysisType] = useState('sector');
  const [selectedTarget, setSelectedTarget] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState('CHECKING...');

  // Clock update
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
      updateMarketStatus();
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const updateMarketStatus = () => {
    const now = new Date();
    const hours = now.getUTCHours() - 5;
    const day = now.getDay();
    
    if (day === 0 || day === 6) {
      setMarketStatus('WEEKEND');
    } else if (hours < 9 || hours === 9) {
      setMarketStatus('PRE-MARKET');
    } else if (hours >= 16) {
      setMarketStatus('AFTER-HOURS');
    } else {
      setMarketStatus('OPEN');
    }
  };

  // Wake backend
  const wakeUpBackend = async () => {
    setBackendStatus('Waking...');
    try {
      const response = await fetch(`${API_URL}/api/health`);
      if (response.ok) {
        setBackendStatus('Online');
        return true;
      }
    } catch (error) {
      setBackendStatus('Offline');
    }
    return false;
  };

  // Fetch market conditions
  const fetchMarketConditions = async () => {
    try {
      const response = await fetch(`${API_URL}/api/market-conditions`);
      if (response.ok) {
        const data = await response.json();
        setMarketConditions({
          regime: data.regime || 'Unknown',
          fedStance: data.fed_stance || 'Neutral',
          vix: data.vix ? data.vix.toFixed(2) : 'N/A',
          recessionRisk: data.recession_risk || 'Unknown'
        });
      }
    } catch (error) {
      console.error('Error fetching market conditions:', error);
    }
  };

  // Check data status
  const checkDataStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/stocks/list`);
      if (response.ok) {
        const data = await response.json();
        if (data.total_stocks > 0) {
          setDataReady(true);
          setSectors(data.sectors || []);
          setSubIndustries(data.sub_industries || []);
        }
      }
    } catch (error) {
      console.error('Error checking data:', error);
    }
  };

  // Initialize
  useEffect(() => {
    const initialize = async () => {
      const isAwake = await wakeUpBackend();
      if (isAwake) {
        await checkDataStatus();
        await fetchMarketConditions();
        setInterval(fetchMarketConditions, 30000);
      }
    };
    initialize();
  }, []);

  // Execute analysis
  const executeAnalysis = async () => {
    if (!selectedTarget) {
      setError('Please select a target');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setError(null);
    setResults(null);

    const progressInterval = setInterval(() => {
      setAnalysisProgress(prev => Math.min(prev + 10, 90));
    }, 200);

    try {
      const response = await fetch(`${API_URL}/api/analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysis_type: analysisType,
          target: selectedTarget
        })
      });

      if (response.ok) {
        const data = await response.json();
        setResults(data.results);
        setAnalysisProgress(100);
      }
    } catch (error) {
      setError(`Analysis failed: ${error.message}`);
    } finally {
      clearInterval(progressInterval);
      setTimeout(() => {
        setIsAnalyzing(false);
        setAnalysisProgress(0);
      }, 1000);
    }
  };

  // Styles
  const styles = `
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    @keyframes holographic {
      0% { background-position: 0% 50%; }
      50% { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }

    @keyframes glow {
      0% { box-shadow: 0 0 20px rgba(255, 255, 255, 0.1); }
      50% { box-shadow: 0 0 30px rgba(255, 255, 255, 0.2), 0 0 60px rgba(255, 255, 255, 0.1); }
      100% { box-shadow: 0 0 20px rgba(255, 255, 255, 0.1); }
    }

    .glass-card {
      background: rgba(255, 255, 255, 0.03);
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 20px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
      transition: all 0.3s ease;
    }

    .glass-card:hover {
      background: rgba(255, 255, 255, 0.05);
      border-color: rgba(255, 255, 255, 0.2);
      transform: translateY(-2px);
      box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }

    .holographic-text {
      background: linear-gradient(135deg, #ffffff 0%, #e0e0e0 25%, #ffffff 50%, #f0f0f0 75%, #ffffff 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      background-size: 200% 200%;
      animation: holographic 3s ease infinite;
    }

    .glass-button {
      background: rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 12px;
      color: white;
      padding: 14px 28px;
      font-weight: 500;
      transition: all 0.3s ease;
      cursor: pointer;
    }

    .glass-button:hover:not(:disabled) {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.3);
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(255, 255, 255, 0.1);
    }

    .glass-button:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .glass-select {
      background: rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(10px);
      border: 1px solid rgba(255, 255, 255, 0.2);
      border-radius: 10px;
      color: white;
      padding: 12px;
      width: 100%;
      outline: none;
      transition: all 0.3s ease;
    }

    .glass-select:focus {
      background: rgba(255, 255, 255, 0.08);
      border-color: rgba(255, 255, 255, 0.3);
    }

    .glass-select option {
      background: #1a1a1a;
      color: white;
    }

    .progress-bar {
      background: rgba(255, 255, 255, 0.1);
      border-radius: 10px;
      overflow: hidden;
      height: 6px;
    }

    .progress-fill {
      height: 100%;
      background: linear-gradient(90deg, rgba(255,255,255,0.3), rgba(255,255,255,0.6));
      border-radius: 10px;
      transition: width 0.3s ease;
    }
  `;

  return (
    <>
      <style>{styles}</style>
      <div style={{
        minHeight: '100vh',
        background: 'radial-gradient(ellipse at top, #1a1a1a 0%, #000000 100%)',
        color: 'white',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Background gradient orbs */}
        <div style={{
          position: 'absolute',
          width: '600px',
          height: '600px',
          background: 'radial-gradient(circle, rgba(255,255,255,0.02) 0%, transparent 70%)',
          borderRadius: '50%',
          top: '-300px',
          left: '-300px',
          pointerEvents: 'none'
        }} />
        <div style={{
          position: 'absolute',
          width: '800px',
          height: '800px',
          background: 'radial-gradient(circle, rgba(255,255,255,0.01) 0%, transparent 70%)',
          borderRadius: '50%',
          bottom: '-400px',
          right: '-400px',
          pointerEvents: 'none'
        }} />

        {/* Header */}
        <header className="glass-card" style={{
          margin: '20px',
          padding: '24px 32px',
          borderRadius: '20px'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <h1 className="holographic-text" style={{
              fontSize: '32px',
              fontWeight: '700',
              letterSpacing: '-1px',
              margin: 0
            }}>
              doDiligence
            </h1>
            <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Database size={18} color={dataReady ? '#4ade80' : '#ef4444'} />
                <span style={{ fontSize: '13px', opacity: 0.8 }}>
                  {backendStatus}
                </span>
              </div>
              <div style={{ fontSize: '13px', opacity: 0.8 }}>
                Market {marketStatus}
              </div>
              <div style={{ fontSize: '13px', opacity: 0.6 }}>
                {currentTime.toLocaleTimeString()}
              </div>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main style={{ padding: '0 20px 20px', maxWidth: '1400px', margin: '0 auto' }}>
          {/* Market Overview */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
            gap: '20px',
            marginBottom: '30px'
          }}>
            {[
              { label: 'Market Regime', value: marketConditions.regime, icon: <TrendingUp size={18} /> },
              { label: 'Fed Stance', value: marketConditions.fedStance, icon: <Activity size={18} /> },
              { label: 'VIX Index', value: marketConditions.vix, icon: <BarChart3 size={18} /> },
              { label: 'Recession Risk', value: marketConditions.recessionRisk, icon: <Activity size={18} /> }
            ].map((item, idx) => (
              <div key={idx} className="glass-card" style={{ padding: '24px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
                  <div style={{ opacity: 0.6 }}>{item.icon}</div>
                  <div style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px', opacity: 0.6 }}>
                    {item.label}
                  </div>
                </div>
                <div className="holographic-text" style={{ fontSize: '24px', fontWeight: '600' }}>
                  {item.value}
                </div>
              </div>
            ))}
          </div>

          {/* Analysis Section */}
          <div style={{ display: 'grid', gridTemplateColumns: '400px 1fr', gap: '20px' }}>
            {/* Controls */}
            <div className="glass-card" style={{ padding: '32px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '24px', opacity: 0.9 }}>
                Analysis Parameters
              </h2>
              
              <div style={{ marginBottom: '24px' }}>
                <label style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px', opacity: 0.6 }}>
                  Analysis Type
                </label>
                <select 
                  className="glass-select"
                  value={analysisType}
                  onChange={(e) => setAnalysisType(e.target.value)}
                  style={{ marginTop: '8px' }}
                >
                  <option value="sector">Sector Analysis</option>
                  <option value="sub_industry">Sub-Industry Analysis</option>
                </select>
              </div>

              <div style={{ marginBottom: '32px' }}>
                <label style={{ fontSize: '12px', textTransform: 'uppercase', letterSpacing: '1px', opacity: 0.6 }}>
                  Target Selection
                </label>
                <select 
                  className="glass-select"
                  value={selectedTarget}
                  onChange={(e) => setSelectedTarget(e.target.value)}
                  style={{ marginTop: '8px' }}
                >
                  <option value="">Select Target</option>
                  {analysisType === 'sector' 
                    ? sectors.map(s => <option key={s} value={s}>{s}</option>)
                    : subIndustries.map(s => <option key={s} value={s}>{s}</option>)
                  }
                </select>
              </div>

              <button
                className="glass-button"
                onClick={executeAnalysis}
                disabled={isAnalyzing || !selectedTarget}
                style={{ width: '100%' }}
              >
                {isAnalyzing ? 'Analyzing...' : 'Execute Analysis'}
              </button>

              {isAnalyzing && (
                <div style={{ marginTop: '20px' }}>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${analysisProgress}%` }} />
                  </div>
                </div>
              )}
            </div>

            {/* Results */}
            <div className="glass-card" style={{ padding: '32px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: '600', marginBottom: '24px', opacity: 0.9 }}>
                Analysis Results
              </h2>
              
              {results && results.top_stocks ? (
                <div style={{ display: 'grid', gap: '16px' }}>
                  {results.top_stocks.map((stock, idx) => (
                    <div key={stock.symbol} className="glass-card" style={{ padding: '20px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                        <div>
                          <div className="holographic-text" style={{ fontSize: '20px', fontWeight: '600' }}>
                            {stock.symbol}
                          </div>
                          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', marginTop: '12px' }}>
                            <div>
                              <div style={{ fontSize: '11px', opacity: 0.6 }}>Current Price</div>
                              <div style={{ fontSize: '16px', fontWeight: '500' }}>${stock.metrics.current_price}</div>
                            </div>
                            <div>
                              <div style={{ fontSize: '11px', opacity: 0.6 }}>Target Price</div>
                              <div style={{ fontSize: '16px', fontWeight: '500' }}>${stock.metrics.target_price}</div>
                            </div>
                            <div>
                              <div style={{ fontSize: '11px', opacity: 0.6 }}>Upside</div>
                              <div style={{ fontSize: '16px', fontWeight: '500', color: stock.metrics.upside_potential > 0 ? '#4ade80' : '#ef4444' }}>
                                {stock.metrics.upside_potential}%
                              </div>
                            </div>
                            <div>
                              <div style={{ fontSize: '11px', opacity: 0.6 }}>Confidence</div>
                              <div style={{ fontSize: '16px', fontWeight: '500' }}>{stock.metrics.confidence_score}%</div>
                            </div>
                          </div>
                        </div>
                        <div style={{ fontSize: '24px', opacity: 0.3 }}>
                          #{idx + 1}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '60px', opacity: 0.4 }}>
                  {error || 'No analysis executed'}
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </>
  );
}

export default App;