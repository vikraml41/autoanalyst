import React, { useState, useEffect } from 'react';
import { Database, TrendingUp, AlertCircle, Activity, DollarSign, Target, Brain, ChevronRight, Clock, BarChart3, FileText, Download, RefreshCw } from 'lucide-react';

// Your backend URL
const API_URL = 'https://autoanalyst-docker.onrender.com';

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
  const [debugInfo, setDebugInfo] = useState({});
  const [backendStatus, setBackendStatus] = useState('CHECKING...');

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

  // Wake up backend with longer timeout and better retries
  const wakeUpBackend = async () => {
    console.log('Waking up backend...');
    setBackendStatus('WAKING...');
    setError(null);
    
    let attempts = 0;
    const maxAttempts = 10; // More attempts for sleepy backend
    
    while (attempts < maxAttempts) {
      try {
        console.log(`Attempt ${attempts + 1} to wake backend...`);
        setBackendStatus(`WAKING... (attempt ${attempts + 1}/${maxAttempts})`);
        
        // Longer timeout for sleepy backend
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
        
        const response = await fetch(`${API_URL}/api/health`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
          },
          signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
          const data = await response.json();
          console.log('Backend is awake!', data);
          setBackendStatus('ONLINE');
          setDebugInfo(prev => ({...prev, health: data}));
          return true;
        }
      } catch (error) {
        console.error(`Wake attempt ${attempts + 1} failed:`, error);
      }
      
      attempts++;
      if (attempts < maxAttempts) {
        // Wait longer between attempts (5 seconds)
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }
    
    setBackendStatus('OFFLINE - Click RETRY');
    setError('Backend took too long to wake up. Click RETRY to try again.');
    return false;
  };

  // Fetch market conditions with error handling
  const fetchMarketConditions = async () => {
    try {
      console.log('Fetching market conditions...');
      
      const response = await fetch(`${API_URL}/api/market-conditions`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Market conditions:', data);
      
      setMarketConditions({
        regime: data.regime || 'UNKNOWN',
        fedStance: data.fed_stance || 'NEUTRAL',
        vix: data.vix ? data.vix.toFixed(2) : 'N/A',
        recessionRisk: data.recession_risk || 'UNKNOWN'
      });
      
      setDebugInfo(prev => ({...prev, marketData: data}));
    } catch (error) {
      console.error('Error fetching market conditions:', error);
      // Don't overwrite with ERROR if still loading
      if (marketConditions.regime !== 'LOADING...') {
        setMarketConditions({
          regime: 'OFFLINE',
          fedStance: 'OFFLINE',
          vix: 'N/A',
          recessionRisk: 'OFFLINE'
        });
      }
    }
  };

  // Check data status and load sectors/industries
  const checkDataStatus = async () => {
    try {
      console.log('Checking data status...');
      
      const response = await fetch(`${API_URL}/api/stocks/list`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      
      const data = await response.json();
      console.log('Stocks data:', data);
      
      if (data.total_stocks > 0) {
        setDataReady(true);
        setSectors(data.sectors || []);
        setSubIndustries(data.sub_industries || []);
        setError(null);
      } else {
        setDataReady(false);
      }
      
      setDebugInfo(prev => ({...prev, stocksData: data}));
    } catch (error) {
      console.error('Error checking data status:', error);
      setDataReady(false);
    }
  };

  // Initialize on mount with proper sequencing
  useEffect(() => {
    const initialize = async () => {
      // First wake up backend
      const isAwake = await wakeUpBackend();
      
      if (isAwake) {
        // Then load data
        await checkDataStatus();
        await fetchMarketConditions();
        
        // Set up refresh interval for market conditions
        const interval = setInterval(() => {
          fetchMarketConditions();
        }, 30000);
        
        return () => clearInterval(interval);
      }
    };
    
    initialize();
  }, []);

  // Manual refresh function
  const manualRefresh = async () => {
    console.log('Manual refresh triggered');
    setError(null);
    setBackendStatus('REFRESHING...');
    const isAwake = await wakeUpBackend();
    if (isAwake) {
      await checkDataStatus();
      await fetchMarketConditions();
    }
  };

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

    const progressInterval = setInterval(() => {
      setAnalysisProgress(prev => Math.min(prev + 10, 90));
    }, 200);

    try {
      const response = await fetch(`${API_URL}/api/analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify({
          analysis_type: analysisType,
          target: selectedTarget
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Analysis failed: ${errorText}`);
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
    if (value === 'LOADING...' || value === 'OFFLINE') return '#808080';
    
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
    overflow: 'auto'
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

  const marketBoxStyle = {
    border: '1px solid #cccc00',
    padding: '20px',
    backgroundColor: '#0a0a0a',
    minHeight: '100px'
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
    padding: '10px 20px',
    backgroundColor: '#0a0a0a',
    color: '#cccc00',
    border: '2px solid #cccc00',
    fontSize: '14px',
    fontWeight: 'bold',
    cursor: 'pointer',
    transition: 'all 0.3s ease'
  };

  return (
    <div style={containerStyle}>
      {/* Header */}
      <header style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <h1 style={titleStyle}>AutoAnalyst</h1>
          <span style={{ 
            color: backendStatus === 'ONLINE' ? '#00ff00' : '#ff0000', 
            fontSize: '11px' 
          }}>
            BACKEND: {backendStatus}
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '30px' }}>
          <button 
            onClick={manualRefresh}
            style={{
              padding: '5px 10px',
              backgroundColor: '#0a0a0a',
              color: '#cccc00',
              border: '1px solid #cccc00',
              cursor: 'pointer',
              fontSize: '11px'
            }}
          >
            RETRY CONNECTION
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Database size={16} color={dataReady ? '#00ff00' : '#ff0000'} />
            <span style={{ color: dataReady ? '#00ff00' : '#ff0000' }}>
              {dataReady ? `DATA.READY` : 'NO.DATA'}
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
        {/* Debug Info Box */}
        <div style={{
          border: '1px solid #ffff00',
          padding: '15px',
          marginBottom: '20px',
          backgroundColor: '#0a0a0a',
          fontSize: '11px'
        }}>
          <div style={{ color: '#ffff00', marginBottom: '10px' }}>DEBUG.INFO</div>
          <div>API_URL: {API_URL}</div>
          <div>Backend Status: {backendStatus}</div>
          <div>Data Ready: {dataReady ? 'YES' : 'NO'}</div>
          <div>Sectors Loaded: {sectors.length}</div>
          <div>Sub-Industries Loaded: {subIndustries.length}</div>
          {error && <div style={{ color: '#ff0000', marginTop: '10px' }}>ERROR: {error}</div>}
        </div>

        {/* Wake up message */}
        {(backendStatus.includes('WAKING') || backendStatus.includes('OFFLINE')) && (
          <div style={{
            border: '1px solid #ffff00',
            padding: '15px',
            marginBottom: '20px',
            backgroundColor: '#1a1a00',
            textAlign: 'center'
          }}>
            <div style={{ color: '#ffff00', marginBottom: '10px' }}>
              {backendStatus.includes('WAKING') 
                ? 'BACKEND IS WAKING UP FROM SLEEP MODE' 
                : 'BACKEND IS OFFLINE'}
            </div>
            <div style={{ fontSize: '11px', color: '#cccc00' }}>
              Free tier sleeps after 15 minutes. This can take 30-60 seconds...
            </div>
            <button
              onClick={manualRefresh}
              style={{
                marginTop: '10px',
                padding: '8px 16px',
                backgroundColor: '#000',
                color: '#ffff00',
                border: '1px solid #ffff00',
                cursor: 'pointer'
              }}
            >
              RETRY CONNECTION
            </button>
          </div>
        )}

        {/* Market Overview */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '20px',
          marginBottom: '30px'
        }}>
          <div style={marketBoxStyle}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              MARKET.REGIME
            </div>
            <div style={{ 
              fontSize: '20px', 
              fontWeight: 'bold', 
              color: getIndicatorColor('regime', marketConditions.regime) 
            }}>
              {marketConditions.regime}
            </div>
          </div>
          
          <div style={marketBoxStyle}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              FED.STANCE
            </div>
            <div style={{ 
              fontSize: '20px', 
              fontWeight: 'bold', 
              color: getIndicatorColor('fedStance', marketConditions.fedStance) 
            }}>
              {marketConditions.fedStance}
            </div>
          </div>
          
          <div style={marketBoxStyle}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              VIX.INDEX
            </div>
            <div style={{ 
              fontSize: '20px', 
              fontWeight: 'bold', 
              color: getIndicatorColor('vix', marketConditions.vix) 
            }}>
              {marketConditions.vix}
            </div>
          </div>
          
          <div style={marketBoxStyle}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              RECESSION.RISK
            </div>
            <div style={{ 
              fontSize: '20px', 
              fontWeight: 'bold', 
              color: getIndicatorColor('recessionRisk', marketConditions.recessionRisk) 
            }}>
              {marketConditions.recessionRisk}
            </div>
          </div>
        </div>

        {/* Analysis Controls */}
        {dataReady && (
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 2fr',
            gap: '30px',
            marginBottom: '30px'
          }}>
            <div style={{
              border: '1px solid #cccc00',
              padding: '25px',
              backgroundColor: '#0a0a0a'
            }}>
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
                  width: '100%',
                  marginTop: '20px'
                }}
                onClick={executeAnalysis}
                disabled={isAnalyzing || !selectedTarget}
              >
                {isAnalyzing ? 'ANALYZING...' : 'EXECUTE.ANALYSIS'}
              </button>

              {isAnalyzing && (
                <div style={{ marginTop: '20px' }}>
                  <div style={{ fontSize: '11px', color: '#808080', marginBottom: '5px' }}>
                    PROGRESS: {analysisProgress}%
                  </div>
                  <div style={{ 
                    width: '100%', 
                    height: '4px', 
                    backgroundColor: '#1a1a1a', 
                    border: '1px solid #cccc00' 
                  }}>
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

            <div style={{
              border: '1px solid #cccc00',
              padding: '25px',
              backgroundColor: '#0a0a0a'
            }}>
              <h2 style={{ fontSize: '16px', marginBottom: '20px', color: '#cccc00' }}>
                ANALYSIS.RESULTS
              </h2>
              
              {results && results.top_stocks ? (
                <div>
                  {results.top_stocks.map((stock, index) => (
                    <div key={stock.symbol} style={{
                      border: '1px solid #cccc00',
                      padding: '15px',
                      marginBottom: '10px',
                      backgroundColor: '#050505'
                    }}>
                      <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#ffff00' }}>
                        #{index + 1} {stock.symbol}
                      </div>
                      <div style={{ marginTop: '10px', fontSize: '12px' }}>
                        <div>PRICE: ${stock.metrics.current_price.toFixed(2)}</div>
                        <div>TARGET: ${stock.metrics.target_price.toFixed(2)}</div>
                        <div>UPSIDE: {stock.metrics.upside_potential.toFixed(1)}%</div>
                        <div>CONFIDENCE: {stock.metrics.confidence_score}%</div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: '#808080', textAlign: 'center', padding: '40px' }}>
                  NO.ANALYSIS.EXECUTED
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;