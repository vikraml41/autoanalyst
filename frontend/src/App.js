import React, { useState, useEffect } from 'react';
import { Database, TrendingUp, AlertCircle, Activity, DollarSign, Target, Brain, ChevronRight, Clock, BarChart3, FileText, Download, RefreshCw } from 'lucide-react';

// IMPORTANT: Your actual backend URL
const API_URL = 'https://autoanalyst-docker.onrender.com';  // NO trailing slash

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

  // Wake up backend with retries
  const wakeUpBackend = async () => {
    console.log('Waking up backend...');
    setBackendStatus('WAKING...');
    
    let attempts = 0;
    const maxAttempts = 5;
    
    while (attempts < maxAttempts) {
      try {
        console.log(`Attempt ${attempts + 1} to wake backend...`);
        const response = await fetch(`${API_URL}/api/health`, {
          method: 'GET',
          headers: {
            'Accept': 'application/json',
          }
        });
        
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
        await new Promise(resolve => setTimeout(resolve, 3000)); // Wait 3 seconds between attempts
      }
    }
    
    setBackendStatus('OFFLINE');
    return false;
  };

  // Fetch market conditions with error handling
  const fetchMarketConditions = async () => {
    try {
      console.log('Fetching market conditions from:', `${API_URL}/api/market-conditions`);
      
      const response = await fetch(`${API_URL}/api/market-conditions`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      });
      
      console.log('Market conditions response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('Market conditions data:', data);
      
      setMarketConditions({
        regime: data.regime || 'UNKNOWN',
        fedStance: data.fed_stance || 'NEUTRAL',
        vix: data.vix ? data.vix.toFixed(2) : 'N/A',
        recessionRisk: data.recession_risk || 'UNKNOWN'
      });
      
      setDebugInfo(prev => ({...prev, marketData: data}));
    } catch (error) {
      console.error('Error fetching market conditions:', error);
      setError(`Market data error: ${error.message}`);
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
      console.log('Checking data status from:', `${API_URL}/api/stocks/list`);
      
      const response = await fetch(`${API_URL}/api/stocks/list`, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        }
      });
      
      console.log('Stocks list response status:', response.status);
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
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
        setError('No stocks loaded in backend');
      }
      
      setDebugInfo(prev => ({...prev, stocksData: data}));
    } catch (error) {
      console.error('Error checking data status:', error);
      setDataReady(false);
      setError(`Data loading error: ${error.message}`);
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
        
        // Set up refresh interval
        const interval = setInterval(() => {
          fetchMarketConditions();
        }, 30000);
        
        return () => clearInterval(interval);
      } else {
        setError('Backend is not responding. It may be sleeping. Please refresh the page in 30 seconds.');
      }
    };
    
    initialize();
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

    const progressInterval = setInterval(() => {
      setAnalysisProgress(prev => Math.min(prev + 10, 90));
    }, 200);

    try {
      console.log('Executing analysis:', { analysisType, target: selectedTarget });
      
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

      console.log('Analysis response status:', response.status);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Analysis failed: ${errorText}`);
      }

      const data = await response.json();
      console.log('Analysis results:', data);
      
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
    if (value === 'ERROR' || value === 'LOADING...') return '#808080';
    
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

  // Manual refresh function
  const manualRefresh = async () => {
    console.log('Manual refresh triggered');
    setError(null);
    await wakeUpBackend();
    await checkDataStatus();
    await fetchMarketConditions();
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

  return (
    <div style={containerStyle}>
      {/* Header */}
      <header style={headerStyle}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <h1 style={titleStyle}>AutoAnalyst</h1>
          <span style={{ color: '#808080', fontSize: '11px' }}>BACKEND: {backendStatus}</span>
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
            REFRESH
          </button>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Database size={16} color={dataReady ? '#00ff00' : '#ff0000'} />
            <span style={{ color: dataReady ? '#00ff00' : '#ff0000' }}>
              {dataReady ? `DATA.READY (${sectors.length} sectors)` : 'NO.DATA'}
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

        {/* Market Overview */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '20px',
          marginBottom: '30px'
        }}>
          <div style={{ border: '1px solid #cccc00', padding: '20px', backgroundColor: '#0a0a0a' }}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              MARKET.REGIME
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: getIndicatorColor('regime', marketConditions.regime) }}>
              {marketConditions.regime}
            </div>
          </div>
          
          <div style={{ border: '1px solid #cccc00', padding: '20px', backgroundColor: '#0a0a0a' }}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              FED.STANCE
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: getIndicatorColor('fedStance', marketConditions.fedStance) }}>
              {marketConditions.fedStance}
            </div>
          </div>
          
          <div style={{ border: '1px solid #cccc00', padding: '20px', backgroundColor: '#0a0a0a' }}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              VIX.INDEX
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: getIndicatorColor('vix', marketConditions.vix) }}>
              {marketConditions.vix}
            </div>
          </div>
          
          <div style={{ border: '1px solid #cccc00', padding: '20px', backgroundColor: '#0a0a0a' }}>
            <div style={{ fontSize: '11px', color: '#808080', marginBottom: '10px' }}>
              RECESSION.RISK
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: getIndicatorColor('recessionRisk', marketConditions.recessionRisk) }}>
              {marketConditions.recessionRisk}
            </div>
          </div>
        </div>

        {/* Rest of your UI continues here... */}
        
      </main>
    </div>
  );
}

export default App;