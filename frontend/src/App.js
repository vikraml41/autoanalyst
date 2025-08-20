import React, { useState, useEffect } from 'react';
import { Terminal, Database, Download, Loader2, Cpu, Signal } from 'lucide-react';

const API_URL = 'http://localhost:8000';

export default function App() {
  const [dataLoaded, setDataLoaded] = useState(false);
  const [sectors, setSectors] = useState([]);
  const [subIndustries, setSubIndustries] = useState([]);
  const [selectionType, setSelectionType] = useState('sector');
  const [selectedValue, setSelectedValue] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [analysisStatus, setAnalysisStatus] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [marketConditions, setMarketConditions] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState('');

  // Update time and market status
  useEffect(() => {
    const updateTimeAndMarket = () => {
      const now = new Date();
      setCurrentTime(now);
      
      const estOffset = -5;
      const utc = now.getTime() + (now.getTimezoneOffset() * 60000);
      const estTime = new Date(utc + (3600000 * estOffset));
      
      const hours = estTime.getHours();
      const minutes = estTime.getMinutes();
      const day = estTime.getDay();
      
      const isWeekday = day > 0 && day < 6;
      const marketOpen = (hours === 9 && minutes >= 30) || (hours > 9 && hours < 16);
      
      if (!isWeekday) {
        setMarketStatus('WEEKEND');
      } else if (hours < 9 || (hours === 9 && minutes < 30)) {
        setMarketStatus('PRE-MARKET');
      } else if (marketOpen) {
        setMarketStatus('OPEN');
      } else if (hours === 16 && minutes < 30) {
        setMarketStatus('AFTER-HOURS');
      } else {
        setMarketStatus('CLOSED');
      }
    };

    updateTimeAndMarket();
    const timer = setInterval(updateTimeAndMarket, 1000);
    return () => clearInterval(timer);
  }, []);

  // Check data status on mount
  useEffect(() => {
    checkDataStatus();
    fetchMarketConditions();
  }, []);

  const checkDataStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/data-status`);
      const data = await response.json();
      
      if (data.loaded) {
        setDataLoaded(true);
        fetchSectors();
      } else {
        setError('NO DATA LOADED. ADD CSV FILES TO BACKEND/DATA/ DIRECTORY');
      }
    } catch (err) {
      setError('FAILED TO CONNECT TO BACKEND. ENSURE SERVER IS RUNNING.');
    }
  };

  const fetchSectors = async () => {
    try {
      const response = await fetch(`${API_URL}/api/sectors`);
      const data = await response.json();
      setSectors(data.sectors || []);
      setSubIndustries(data.sub_industries || []);
    } catch (err) {
      console.error('Failed to fetch sectors:', err);
    }
  };

  const fetchMarketConditions = async () => {
    try {
      const response = await fetch(`${API_URL}/api/market-conditions`);
      const data = await response.json();
      setMarketConditions(data);
    } catch (err) {
      console.error('Failed to fetch market conditions:', err);
    }
  };

  const startAnalysis = async () => {
    if (!selectedValue) {
      setError('SELECT A SECTOR OR SUB-INDUSTRY');
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setAnalysisResults(null);

    try {
      const response = await fetch(`${API_URL}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selection_type: selectionType,
          selection_value: selectedValue,
        }),
      });

      if (!response.ok) throw new Error('Analysis failed');
      
      const data = await response.json();
      setSessionId(data.session_id);
      pollAnalysisStatus(data.session_id);
    } catch (err) {
      setError('ANALYSIS FAILED');
      setIsAnalyzing(false);
    }
  };

  const pollAnalysisStatus = (sessionId) => {
    const interval = setInterval(async () => {
      try {
        const response = await fetch(`${API_URL}/api/status/${sessionId}`);
        const data = await response.json();
        
        setAnalysisStatus(data);
        
        if (data.status === 'completed') {
          clearInterval(interval);
          setAnalysisResults(data.results);
          setIsAnalyzing(false);
          fetchMarketConditions();
        } else if (data.status === 'error') {
          clearInterval(interval);
          setError('ANALYSIS ERROR');
          setIsAnalyzing(false);
        }
      } catch (err) {
        clearInterval(interval);
        setError('STATUS CHECK FAILED');
        setIsAnalyzing(false);
      }
    }, 1000);
  };

  const downloadReport = () => {
    if (sessionId) {
      window.open(`${API_URL}/api/download-report/${sessionId}`, '_blank');
    }
  };

  // Helper functions for colors
  const getMarketStatusColor = () => {
    if (marketStatus === 'OPEN') return '#00ff00';
    if (marketStatus === 'CLOSED') return '#ff0000';
    if (marketStatus === 'PRE-MARKET' || marketStatus === 'AFTER-HOURS') return '#cccc00';
    return '#808080';
  };

  const getMarketRegimeColor = (regime) => {
    if (!regime) return '#808080';
    const regimeStr = String(regime);
    if (regimeStr.includes('Bull')) return '#00ff00';
    if (regimeStr.includes('Bear')) return '#ff0000';
    return '#ffff00';
  };

  const getFedStanceColor = (stance) => {
    if (!stance) return '#808080';
    const stanceStr = String(stance);
    if (stanceStr.includes('Dovish') || stanceStr.includes('Easing')) return '#00ff00';
    if (stanceStr.includes('Hawkish') || stanceStr.includes('Tightening')) return '#ff0000';
    return '#ffff00';
  };

  const getVixColor = (vix) => {
    if (!vix) return '#808080';
    const vixNum = Number(vix);
    if (vixNum < 20) return '#00ff00';
    if (vixNum > 30) return '#ff0000';
    return '#ffff00';
  };

  const getRecessionRiskColor = (risk) => {
    if (!risk) return '#808080';
    const riskStr = String(risk);
    if (riskStr === 'Low') return '#00ff00';
    if (riskStr === 'High') return '#ff0000';
    return '#ffff00';
  };

  const formatNumber = (num) => {
    if (!num) return '0';
    const n = Number(num);
    if (n >= 1e12) return (n / 1e12).toFixed(2) + 'T';
    if (n >= 1e9) return (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    return n.toFixed(2);
  };

  const styles = {
    container: {
      minHeight: '100vh',
      width: '100vw',
      backgroundColor: '#000',
      color: '#cccc00',
      fontFamily: 'monospace',
      padding: 0,
      margin: 0,
      overflow: 'auto'
    },
    header: {
      borderBottom: '2px solid #cccc00',
      padding: '15px 30px',
      backgroundColor: '#000',
      width: '100%',
      boxSizing: 'border-box'
    },
    headerContent: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      width: '100%',
      maxWidth: '1400px',
      margin: '0 auto'
    },
    logo: {
      display: 'flex',
      alignItems: 'center',
      gap: '15px'
    },
    title: {
      fontSize: '28px',
      fontWeight: 'bold',
      color: '#cccc00',
      fontFamily: 'Instrument Serif, serif',
      margin: 0
    },
    subtitle: {
      fontSize: '10px',
      color: '#808080',
      letterSpacing: '2px',
      margin: 0
    },
    main: {
      padding: '30px',
      width: '100%',
      maxWidth: '1400px',
      margin: '0 auto',
      backgroundColor: '#000',
      boxSizing: 'border-box'
    },
    marketGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(4, 1fr)',
      gap: '20px',
      marginBottom: '30px',
      width: '100%'
    },
    marketBox: {
      border: '2px solid #cccc00',
      padding: '20px',
      backgroundColor: '#000',
      minHeight: '100px',
      display: 'flex',
      flexDirection: 'column',
      justifyContent: 'center'
    },
    analysisGrid: {
      display: 'grid',
      gridTemplateColumns: '350px 1fr',
      gap: '30px',
      width: '100%'
    },
    panel: {
      border: '2px solid #cccc00',
      padding: '25px',
      backgroundColor: '#000',
      width: '100%',
      boxSizing: 'border-box'
    },
    button: {
      width: '100%',
      padding: '12px',
      fontSize: '14px',
      fontWeight: 'bold',
      border: '2px solid #cccc00',
      backgroundColor: '#cccc00',
      color: '#000',
      cursor: 'pointer'
    },
    buttonDisabled: {
      backgroundColor: '#000',
      color: '#606060',
      borderColor: '#606060',
      cursor: 'not-allowed'
    },
    select: {
      width: '100%',
      padding: '10px',
      fontSize: '13px',
      backgroundColor: '#000',
      color: '#cccc00',
      border: '2px solid #cccc00',
      outline: 'none',
      boxSizing: 'border-box'
    },
    stockCard: {
      border: '2px solid #cccc00',
      padding: '20px',
      backgroundColor: '#000',
      marginBottom: '15px',
      width: '100%',
      boxSizing: 'border-box'
    },
    errorBox: {
      margin: '15px 30px',
      padding: '12px 20px',
      border: '2px solid #ff0000',
      backgroundColor: '#000',
      color: '#ff0000',
      fontSize: '13px',
      maxWidth: '1340px',
      margin: '15px auto',
      boxSizing: 'border-box'
    }
  };

  return (
    <div style={styles.container}>
      <link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&display=swap" rel="stylesheet" />
      
      <style>{`
        body {
          margin: 0;
          padding: 0;
          background-color: #000;
          overflow-x: hidden;
        }
        * {
          box-sizing: border-box;
        }
      `}</style>
      
      {/* Header */}
      <header style={styles.header}>
        <div style={styles.headerContent}>
          <div style={styles.logo}>
            <svg width="36" height="36" viewBox="0 0 32 32" fill="none">
              <rect x="4" y="20" width="4" height="8" fill="#cccc00" />
              <rect x="10" y="16" width="4" height="12" fill="#cccc00" />
              <rect x="16" y="12" width="4" height="16" fill="#cccc00" />
              <rect x="22" y="8" width="4" height="20" fill="#cccc00" />
            </svg>
            <div>
              <h1 style={styles.title}>AutoAnalyst</h1>
              <p style={styles.subtitle}>QUANTITATIVE.FINANCIAL.INTELLIGENCE</p>
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '30px' }}>
            {dataLoaded && (
              <div style={{ display: 'flex', alignItems: 'center', color: '#cccc00', fontSize: '13px' }}>
                <Database size={16} style={{ marginRight: '8px' }} />
                DATA.READY
              </div>
            )}
            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '11px', color: '#808080' }}>SYSTEM.TIME</div>
              <div style={{ fontSize: '16px', color: '#cccc00' }}>
                {currentTime.toLocaleTimeString('en-US', { hour12: false })}
              </div>
              <div style={{ fontSize: '11px', fontWeight: 'bold', color: getMarketStatusColor() }}>
                MKT.{marketStatus}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Error Display */}
      {error && (
        <div style={styles.errorBox}>
          [ERROR] {error}
          <button onClick={() => setError(null)} style={{ float: 'right', background: 'none', border: 'none', color: '#ff0000', cursor: 'pointer', fontSize: '16px' }}>
            [X]
          </button>
        </div>
      )}

      {/* Main Content */}
      <main style={styles.main}>
        {/* Market Overview */}
        <div style={styles.marketGrid}>
          <div style={styles.marketBox}>
            <div style={{ fontSize: '12px', color: '#808080', marginBottom: '8px' }}>MARKET.REGIME</div>
            <div style={{ fontSize: '18px', fontWeight: 'bold', color: getMarketRegimeColor(marketConditions?.market_regime), lineHeight: '1.2' }}>
              {marketConditions?.market_regime || 'LOADING...'}
            </div>
          </div>
          
          <div style={styles.marketBox}>
            <div style={{ fontSize: '12px', color: '#808080', marginBottom: '8px' }}>FED.STANCE</div>
            <div style={{ fontSize: '18px', fontWeight: 'bold', color: getFedStanceColor(marketConditions?.fed_stance?.stance), lineHeight: '1.2' }}>
              {marketConditions?.fed_stance?.stance || 'LOADING...'}
            </div>
          </div>
          
          <div style={styles.marketBox}>
            <div style={{ fontSize: '12px', color: '#808080', marginBottom: '8px' }}>VIX.INDEX</div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: getVixColor(marketConditions?.economic_indicators?.['Volatility Index']?.current) }}>
              {marketConditions?.economic_indicators?.['Volatility Index']?.current?.toFixed(2) || 'N/A'}
            </div>
          </div>
          
          <div style={styles.marketBox}>
            <div style={{ fontSize: '12px', color: '#808080', marginBottom: '8px' }}>RECESSION.RISK</div>
            <div style={{ fontSize: '20px', fontWeight: 'bold', color: getRecessionRiskColor(marketConditions?.yield_curve?.recession_risk) }}>
              {marketConditions?.yield_curve?.recession_risk || 'LOADING...'}
            </div>
          </div>
        </div>

        {/* Analysis Interface */}
        <div style={styles.analysisGrid}>
          {/* Selection Panel */}
          <div>
            <div style={{ ...styles.panel, minHeight: '400px' }}>
              <h2 style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '25px', color: '#cccc00' }}>
                ANALYSIS.PARAMETERS
              </h2>

              <div style={{ marginBottom: '25px' }}>
                <label style={{ fontSize: '11px', color: '#808080', display: 'block', marginBottom: '10px' }}>
                  SELECTION.TYPE
                </label>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <button
                    onClick={() => setSelectionType('sector')}
                    style={{
                      ...styles.button,
                      padding: '10px',
                      fontSize: '13px',
                      ...(selectionType !== 'sector' ? styles.buttonDisabled : {})
                    }}
                  >
                    SECTOR
                  </button>
                  <button
                    onClick={() => setSelectionType('sub_industry')}
                    style={{
                      ...styles.button,
                      padding: '10px',
                      fontSize: '13px',
                      ...(selectionType !== 'sub_industry' ? styles.buttonDisabled : {})
                    }}
                  >
                    SUB-INDUSTRY
                  </button>
                </div>
              </div>

              <div style={{ marginBottom: '25px' }}>
                <label style={{ fontSize: '11px', color: '#808080', display: 'block', marginBottom: '10px' }}>
                  SELECT.{selectionType === 'sector' ? 'SECTOR' : 'SUB_INDUSTRY'}
                </label>
                <select
                  value={selectedValue}
                  onChange={(e) => setSelectedValue(e.target.value)}
                  style={styles.select}
                  disabled={!dataLoaded}
                >
                  <option value="">-- SELECT --</option>
                  {(selectionType === 'sector' ? sectors : subIndustries).map(item => (
                    <option key={item} value={item}>{String(item).toUpperCase()}</option>
                  ))}
                </select>
              </div>

              <button
                onClick={startAnalysis}
                disabled={!selectedValue || isAnalyzing || !dataLoaded}
                style={{
                  ...styles.button,
                  ...(!selectedValue || isAnalyzing || !dataLoaded ? styles.buttonDisabled : {})
                }}
              >
                {isAnalyzing ? 'PROCESSING...' : 'EXECUTE.ANALYSIS'}
              </button>

              {analysisStatus && isAnalyzing && (
                <div style={{ marginTop: '20px' }}>
                  <div style={{ fontSize: '11px', color: '#808080', marginBottom: '8px' }}>
                    PROGRESS: {analysisStatus.progress}%
                  </div>
                  <div style={{ width: '100%', height: '8px', backgroundColor: '#000', border: '2px solid #cccc00' }}>
                    <div style={{
                      width: `${analysisStatus.progress}%`,
                      height: '100%',
                      backgroundColor: '#cccc00',
                      transition: 'width 0.5s'
                    }} />
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Results Panel */}
          <div>
            {analysisResults ? (
              <div>
                <div style={{ ...styles.panel, marginBottom: '20px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                    <h2 style={{ fontSize: '16px', fontWeight: 'bold', color: '#cccc00', margin: 0 }}>
                      ANALYSIS.RESULTS
                    </h2>
                    <button onClick={downloadReport} style={{ ...styles.button, width: 'auto', padding: '8px 20px', fontSize: '13px' }}>
                      EXPORT
                    </button>
                  </div>
                </div>

                {/* Stock Cards */}
                {analysisResults.stock_analyses && analysisResults.stock_analyses.slice(0, 3).map((stock, index) => (
                  <div key={stock.symbol} style={styles.stockCard}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px' }}>
                      <div>
                        <div style={{ fontSize: '11px', color: '#cccc00', marginBottom: '5px' }}>
                          RANK.{String(index + 1).padStart(2, '0')}
                        </div>
                        <h3 style={{ fontSize: '24px', fontWeight: 'bold', color: '#cccc00', margin: 0, fontFamily: 'Instrument Serif, serif' }}>
                          {stock.symbol}
                        </h3>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '11px', color: '#cccc00', marginBottom: '5px' }}>CONF.LVL</div>
                        <div style={{ fontSize: '22px', fontWeight: 'bold', color: '#cccc00' }}>
                          {Math.round(stock.confidence * 100)}%
                        </div>
                      </div>
                    </div>
                    
                    <div style={{ fontSize: '14px', lineHeight: '1.8' }}>
                      <div>CURRENT: ${Number(stock.current_price).toFixed(2)}</div>
                      <div>TARGET: ${Number(stock.target_price).toFixed(2)}</div>
                      <div style={{ color: stock.upside_potential > 0 ? '#00ff00' : '#ff0000', fontWeight: 'bold' }}>
                        UPSIDE: {(stock.upside_potential * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ ...styles.panel, minHeight: '400px', textAlign: 'center', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                <h3 style={{ fontSize: '18px', fontWeight: 'bold', color: '#cccc00', marginBottom: '15px' }}>NO.ANALYSIS.DATA</h3>
                <p style={{ fontSize: '13px', color: '#808080' }}>
                  SELECT PARAMETERS AND EXECUTE ANALYSIS TO VIEW RESULTS
                </p>
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}