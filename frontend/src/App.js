import React, { useState, useEffect } from 'react';
import { Database, TrendingUp, AlertCircle, Activity, DollarSign, Target, Brain, ChevronRight, Clock, BarChart3, FileText, Download, RefreshCw } from 'lucide-react';

// IMPORTANT: Update this with your actual backend URL
const API_URL = window.location.hostname === 'localhost' 
  ? 'http://localhost:8000'
  : 'https://autoanalyst-docker.onrender.com';  // Your backend URL

function App() {
  // State management
  const [currentTime, setCurrentTime] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState('CHECKING...');
  const [marketConditions, setMarketConditions] = useState({
    regime: 'LOADING...',
    fedStance: 'LOADING...',
    vix: 'LOADING...',
    recessionRisk: 'LOADING...'
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

  // Update clock
  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
      updateMarketStatus();
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Update market status based on time
  const updateMarketStatus = () => {
    const now = new Date();
    const hours = now.getUTCHours() - 5; // Convert to EST
    const minutes = now.getMinutes();
    const day = now.getDay();
    
    if (day === 0 || day === 6) {
      setMarketStatus('WEEKEND');
    } else if (hours < 9 || (hours === 9 && minutes < 30)) {
      setMarketStatus('PRE-MARKET');
    } else if (hours >= 16) {
      setMarketStatus('AFTER-HOURS');
    } else if (hours >= 9 && hours < 16) {
      setMarketStatus('OPEN');
    } else {
      setMarketStatus('CLOSED');
    }
  };

  // Fetch market conditions
  const fetchMarketConditions = async () => {
    try {
      const response = await fetch(`${API_URL}/api/market-conditions`);
      if (!response.ok) throw new Error('Failed to fetch market conditions');
      const data = await response.json();
      
      setMarketConditions({
        regime: data.regime || 'UNKNOWN',
        fedStance: data.fed_stance || 'NEUTRAL',
        vix: data.vix ? data.vix.toFixed(2) : 'N/A',
        recessionRisk: data.recession_risk || 'UNKNOWN'
      });
    } catch (error) {
      console.error('Error fetching market conditions:', error);
      setMarketConditions({
        regime: 'ERROR',
        fedStance: 'ERROR',
        vix: 'ERROR',
        recessionRisk: 'ERROR'
      });
    }
  };

  // Check data status and load sectors/industries
  const checkDataStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/stocks/list`);
      if (!response.ok) throw new Error('Failed to fetch stocks list');
      const data = await response.json();
      
      console.log('Stocks data:', data); // Debug log
      
      if (data.total_stocks > 0) {
        setDataReady(true);
        setSectors(data.sectors || []);
        setSubIndustries(data.sub_industries || []);
      } else {
        setDataReady(false);
        setError('No stocks loaded in backend');
      }
    } catch (error) {
      console.error('Error checking data status:', error);
      setDataReady(false);
      setError('Failed to connect to backend');
    }
  };

  // Initialize on mount
  useEffect(() => {
    // Wake up backend if sleeping
    fetch(`${API_URL}/api/health`)
      .then(() => console.log('Backend is awake'))
      .catch(() => console.log('Waking up backend...'));
    
    // Load data
    checkDataStatus();
    fetchMarketConditions();
    
    // Refresh market conditions every 30 seconds
    const interval = setInterval(fetchMarketConditions, 30000);
    return () => clearInterval(interval);
  }, []);

  // Execute analysis
  const executeAnalysis = async () => {
    if (!selectedTarget) {
      setError('Please select a target for analysis');
      return;
    }

    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setError(null);
    setResults(null);

    // Simulate progress
    const progressInterval = setInterval(() => {
      setAnalysisProgress(prev => Math.min(prev + 10, 90));
    }, 200);

    try {
      const response = await fetch(`${API_URL}/api/analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          analysis_type: analysisType,
          target: selectedTarget
        })
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data.results);
      setAnalysisProgress(100);
    } catch (error) {
      console.error('Analysis error:', error);
      setError(`Analysis failed: ${error.message}`);
    } finally {
      clearInterval(progressInterval);
      setTimeout(() => {
        setIsAnalyzing(false);
        setAnalysisProgress(0);
      }, 1000);
    }
  };

  // Get color for market indicators
  const getIndicatorColor = (type, value) => {
    switch(type) {
      case 'regime':
        return value.includes('Bull') ? '#00ff00' : value.includes('Bear') ? '#ff0000' : '#ffff00';
      case 'vix':
        const vixNum = parseFloat(value);
        if (isNaN(vixNum)) return '#808080';
        return vixNum < 20 ? '#00ff00' : vixNum > 30 ? '#ff0000' : '#ffff00';
      case 'fedStance':
        return value.includes('Dovish') ? '#00ff00' : value.includes('Hawkish') ? '#ff0000' : '#ffff00';
      case 'recessionRisk':
        return value.includes('Low') ? '#00ff00' : value.includes('High') ? '#ff0000' : '#ffff00';
      default:
        return '#cccc00';
    }
  };

  const containerStyle = {
    minHeight: '100vh',
    width: '100vw',
    backgroundColor: '#000000',
    color: '#cccc00',
    fontFamily: 'monospace',
    fontSize: '13px',
    overflow: 'auto',
    position: 'relative'
  };

  const headerStyle = {
    borderBottom: '2px solid #cccc00',
    padding: '15px 30px',
    backgroundColor: '#0a0a0a',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center'
  };

  const titleStyle = {
    fontSize: '28px',
    fontWeight: 'bold',
    letterSpacing: '2px',
    fontFamily: "'Instrument Serif', serif",
    color: '#cccc00'
  };

  const mainContentStyle = {
    padding: '30px',
    maxWidth: '1400px',
    margin: '0 auto'
  };

  const marketOverviewStyle = {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '20px',
    marginBottom: '30px'
  };

  const marketBoxStyle = {
    border: '1px solid #cccc00',
    padding: '20px',
    backgroundColor: '#0a0a0a',
    minHeight: '100px'
  };

  const selectionPanelStyle = {
    display: 'grid',
    gridTemplateColumns: '1fr 2fr',
    gap: '30px',
    marginBottom: '30px'
  };

  const controlsStyle = {
    border: '1px solid #cccc00',
    padding: '25px',
    backgroundColor: '#0a0a0a'
  };

  const selectStyle = {
    width: '100%',
    padding: '10px',
    backgroundColor: '#000',
    color: '#cccc00',
    border: '1px solid #cccc00',
    fontSize: '13px',
    fontFamily: 'monospace',
    marginTop: '10px',
    cursor: 'pointer'
  };

  const buttonStyle = {
    width: '100%',
    padding: '12px',
    backgroundColor: '#0a0a0a',
    color: '#cccc00',
    border: '2px solid #cccc00',
    fontSize: '14px',
    fontWeight: 'bold',
    cursor: 'pointer',
    marginTop: '20px',
    transition: 'all 0.3s ease',
    letterSpacing: '1px'
  };

  const resultsStyle = {
    border: '1px solid #cccc00',
    padding: '25px',
    backgroundColor: '#0a0a0a'
  };

  const stockCardStyle = {
    border: '1px solid #cccc00',
    padding: '20px',
    marginBottom: '15px',
    backgroundColor: '#050505',
    transition: 'all 0.3s ease'
  };

  return (
    <div style={containerStyle}>
      {/* Background Chart Pattern */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        opacity: 0.05,
        pointerEvents: 'none',
        backgroundImage: `repeating-linear-gradient(90deg, transparent, transparent 50px, #cccc00 50px, #cccc00 51px)`,
        zIndex: 0
      }} />

      {/* Header */}
      <header style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <h1 style={titleStyle}>AutoAnalyst</h1>
          <span style={{ color: '#808080', fontSize: '11px' }}>QUANT.ML.V3</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '30px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Database size={16} color={dataReady ? '#00ff00' : '#ff0000'} />
            <span style={{ color: dataReady ? '#00ff00' : '#ff0000' }}>
              {dataReady ? 'DATA.READY' : 'NO.DATA'}
            </span>
          </div>
          <div style={{ color: marketStatus === 'OPEN' ? '#00ff00' : '#ff0000' }}>
            MARKET.{marketStatus}
          </div>
          <div>
            {currentTime.toLocaleTimeString('en-US', { hour12: false })}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main style={mainContentStyle}>
        {/* Market Overview */}
        <div style={marketOverviewStyle}>
          <div style={marketBoxStyle}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              MARKET.REGIME
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: getIndicatorColor('regime', marketConditions.regime) }}>
              {marketConditions.regime}
            </div>
          </div>
          
          <div style={marketBoxStyle}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              FED.STANCE
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: getIndicatorColor('fedStance', marketConditions.fedStance) }}>
              {marketConditions.fedStance}
            </div>
          </div>
          
          <div style={marketBoxStyle}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              VIX.INDEX
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: getIndicatorColor('vix', marketConditions.vix) }}>
              {marketConditions.vix}
            </div>
          </div>
          
          <div style={marketBoxStyle}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              RECESSION.RISK
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: getIndicatorColor('recessionRisk', marketConditions.recessionRisk) }}>
              {marketConditions.recessionRisk}
            </div>
          </div>
        </div>

        {/* Selection Panel */}
        <div style={selectionPanelStyle}>
          <div style={controlsStyle}>
            <h2 style={{ fontSize: '16px', marginBottom: '20px', color: '#cccc00' }}>
              ANALYSIS.PARAMETERS
            </h2>
            
            <div style={{ marginBottom: '20px' }}>
              <label style={{ fontSize: '11px', color: '#808080' }}>
                ANALYSIS.TYPE
              </label>
              <select 
                style={selectStyle}
                value={analysisType}
                onChange={(e) => {
                  setAnalysisType(e.target.value);
                  setSelectedTarget('');
                }}
              >
                <option value="sector">SECTOR.ANALYSIS</option>
                <option value="sub_industry">SUB.INDUSTRY.ANALYSIS</option>
              </select>
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label style={{ fontSize: '11px', color: '#808080' }}>
                TARGET.SELECTION
              </label>
              <select 
                style={selectStyle}
                value={selectedTarget}
                onChange={(e) => setSelectedTarget(e.target.value)}
              >
                <option value="">-- SELECT.TARGET --</option>
                {analysisType === 'sector' 
                  ? sectors.map(sector => (
                      <option key={sector} value={sector}>{sector.toUpperCase()}</option>
                    ))
                  : subIndustries.map(industry => (
                      <option key={industry} value={industry}>{industry.toUpperCase()}</option>
                    ))
                }
              </select>
            </div>

            <button
              style={{
                ...buttonStyle,
                backgroundColor: isAnalyzing ? '#1a1a00' : buttonStyle.backgroundColor,
                borderColor: isAnalyzing ? '#808080' : buttonStyle.border,
                cursor: isAnalyzing ? 'not-allowed' : 'pointer'
              }}
              onClick={executeAnalysis}
              disabled={isAnalyzing || !dataReady || !selectedTarget}
              onMouseEnter={(e) => {
                if (!isAnalyzing) {
                  e.target.style.backgroundColor = '#1a1a00';
                  e.target.style.boxShadow = '0 0 10px #cccc00';
                }
              }}
              onMouseLeave={(e) => {
                e.target.style.backgroundColor = '#0a0a0a';
                e.target.style.boxShadow = 'none';
              }}
            >
              {isAnalyzing ? 'ANALYZING...' : 'EXECUTE.ANALYSIS'}
            </button>

            {isAnalyzing && (
              <div style={{ marginTop: '20px' }}>
                <div style={{ fontSize: '11px', color: '#808080', marginBottom: '5px' }}>
                  PROGRESS: {analysisProgress}%
                </div>
                <div style={{ width: '100%', height: '4px', backgroundColor: '#1a1a1a', border: '1px solid #cccc00' }}>
                  <div style={{
                    width: `${analysisProgress}%`,
                    height: '100%',
                    backgroundColor: '#cccc00',
                    transition: 'width 0.3s ease'
                  }} />
                </div>
              </div>
            )}
          </div>

          <div style={resultsStyle}>
            <h2 style={{ fontSize: '16px', marginBottom: '20px', color: '#cccc00' }}>
              ANALYSIS.RESULTS
            </h2>
            
            {error && (
              <div style={{ color: '#ff0000', marginBottom: '20px' }}>
                ERROR: {error}
              </div>
            )}

            {results && results.top_stocks ? (
              <div>
                {results.top_stocks.map((stock, index) => (
                  <div 
                    key={stock.symbol} 
                    style={stockCardStyle}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.borderColor = '#ffff00';
                      e.currentTarget.style.boxShadow = '0 0 15px rgba(204, 204, 0, 0.3)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.borderColor = '#cccc00';
                      e.currentTarget.style.boxShadow = 'none';
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                      <div>
                        <div style={{ fontSize: '18px', fontWeight: 'bold', marginBottom: '10px', color: '#ffff00' }}>
                          #{index + 1} {stock.symbol}
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '10px', fontSize: '12px' }}>
                          <div>
                            <span style={{ color: '#808080' }}>PRICE: </span>
                            <span style={{ color: '#00ff00' }}>${stock.metrics.current_price.toFixed(2)}</span>
                          </div>
                          <div>
                            <span style={{ color: '#808080' }}>TARGET: </span>
                            <span style={{ color: '#00ff00' }}>${stock.metrics.target_price.toFixed(2)}</span>
                          </div>
                          <div>
                            <span style={{ color: '#808080' }}>UPSIDE: </span>
                            <span style={{ color: stock.metrics.upside_potential > 0 ? '#00ff00' : '#ff0000' }}>
                              {stock.metrics.upside_potential.toFixed(1)}%
                            </span>
                          </div>
                          <div>
                            <span style={{ color: '#808080' }}>CONFIDENCE: </span>
                            <span style={{ color: '#cccc00' }}>{stock.metrics.confidence_score}%</span>
                          </div>
                        </div>
                      </div>
                      <div style={{ fontSize: '24px', color: '#00ff00' }}>
                        ↗
                      </div>
                    </div>
                    
                    <div style={{ marginTop: '15px', paddingTop: '15px', borderTop: '1px solid #333' }}>
                      <div style={{ fontSize: '11px', color: '#808080', marginBottom: '5px' }}>
                        SENTIMENT.SCORE
                      </div>
                      <div style={{ width: '100%', height: '6px', backgroundColor: '#1a1a1a', border: '1px solid #333' }}>
                        <div style={{
                          width: `${stock.metrics.sentiment_score * 100}%`,
                          height: '100%',
                          backgroundColor: stock.metrics.sentiment_score > 0.7 ? '#00ff00' : 
                                         stock.metrics.sentiment_score > 0.4 ? '#ffff00' : '#ff0000'
                        }} />
                      </div>
                    </div>
                  </div>
                ))}
                
                <div style={{ marginTop: '20px', paddingTop: '20px', borderTop: '1px solid #cccc00' }}>
                  <button
                    style={{
                      ...buttonStyle,
                      width: 'auto',
                      padding: '10px 20px'
                    }}
                    onClick={() => console.log('Export report')}
                  >
                    <Download size={14} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                    EXPORT.REPORT
                  </button>
                </div>
              </div>
            ) : (
              <div style={{ color: '#808080', textAlign: 'center', padding: '40px' }}>
                {dataReady ? 'NO.ANALYSIS.EXECUTED' : 'WAITING.FOR.DATA...'}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        borderTop: '1px solid #cccc00',
        padding: '15px 30px',
        backgroundColor: '#0a0a0a',
        color: '#808080',
        fontSize: '11px',
        display: 'flex',
        justifyContent: 'space-between'
      }}>
        <div>AUTOANALYST.QUANTUM.FINANCE.SYSTEM</div>
        <div>LATENCY: 12MS | COMPUTE: 0.3s</div>
      </footer>
    </div>
  );
}

export default App;