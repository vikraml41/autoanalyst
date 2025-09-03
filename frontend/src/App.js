import React, { useState, useEffect, useRef } from 'react';
import { TrendingUp, Activity, BarChart3, ChevronDown, ChevronUp } from 'lucide-react';

const API_URL = window.location.hostname === 'localhost' 
  ? 'http://localhost:8000'
  : 'https://autoanalyst-dz11.onrender.com';

function App() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState('CHECKING');
  const [marketConditions, setMarketConditions] = useState({
    regime: 'Loading',
    fedStance: 'Loading',
    vix: 'Loading',
    recessionRisk: 'Loading'
  });
  const [sectors, setSectors] = useState([]);
  const [subIndustries, setSubIndustries] = useState([]);
  const [analysisType, setAnalysisType] = useState('sector');
  const [selectedTarget, setSelectedTarget] = useState('');
  const [marketCapSize, setMarketCapSize] = useState('all');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [expandedStock, setExpandedStock] = useState(null);
  const [analysisProgress, setAnalysisProgress] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const dropdownRef = useRef(null);

  // Clock and market status update
  useEffect(() => {
    const timer = setInterval(() => {
      const now = new Date();
      setCurrentTime(now);
      
      const hours = now.getUTCHours() - 5; // EST
      const day = now.getDay();
      
      if (day === 0 || day === 6) {
        setMarketStatus('WEEKEND');
      } else if (hours < 9 || (hours === 9 && now.getUTCMinutes() < 30)) {
        setMarketStatus('PRE-MARKET');
      } else if (hours >= 16) {
        setMarketStatus('AFTER-HOURS');
      } else {
        setMarketStatus('OPEN');
      }
    }, 1000);
    
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const initialize = async () => {
      setIsLoading(true);
      await Promise.all([
        fetchMarketConditions(),
        checkDataStatus()
      ]);
      setTimeout(() => setIsLoading(false), 1500);
    };
    initialize();
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
      console.error('Error:', error);
    }
  };

  const checkDataStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/api/stocks/list`);
      if (response.ok) {
        const data = await response.json();
        setSectors(data.sectors || []);
        setSubIndustries(data.sub_industries || []);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  const executeAnalysis = async () => {
    if (!selectedTarget) return;
    setIsAnalyzing(true);
    setResults(null);
    setExpandedStock(null);
    setAnalysisProgress('Initializing analysis...');

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 180000);

    let messageIndex = 0;
    const messages = [
      'Fetching market data...',
      'Analyzing sector conditions...',
      'Running ML models...',
      'Calculating valuations...',
      'Generating predictions...',
      'Finalizing results...',
      'Almost complete...'
    ];

    const progressInterval = setInterval(() => {
      if (messageIndex < messages.length - 1) {
        messageIndex++;
        setAnalysisProgress(messages[messageIndex]);
      }
    }, 5000);

    try {
      const response = await fetch(`${API_URL}/api/analysis`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          analysis_type: analysisType,
          target: selectedTarget,
          market_cap_size: marketCapSize
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      clearInterval(progressInterval);

      if (response.ok) {
        const data = await response.json();
        setResults(data.results);
        setAnalysisProgress('');
      } else {
        setAnalysisProgress('Analysis failed - please try again');
        setTimeout(() => setAnalysisProgress(''), 3000);
      }
    } catch (error) {
      clearInterval(progressInterval);
      if (error.name === 'AbortError') {
        setAnalysisProgress('Analysis is taking longer than expected. Try selecting a smaller sector.');
        setTimeout(() => setAnalysisProgress(''), 5000);
      } else {
        setAnalysisProgress('Connection error - please try again');
        setTimeout(() => setAnalysisProgress(''), 3000);
      }
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const styles = `
    @import url('https://fonts.googleapis.com/css2?family=Questrial&display=swap');
    
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      font-family: 'Avant Garde', 'ITC Avant Garde Gothic', 'Questrial', -apple-system, sans-serif !important;
    }
    
    body {
      font-family: 'Avant Garde', 'ITC Avant Garde Gothic', 'Questrial', -apple-system, sans-serif !important;
      background: #000000;
      overflow-x: hidden;
    }

    /* Loading Screen */
    .loading-screen {
      position: fixed;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      background: #000000;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      z-index: 100000;
    }

    @keyframes holographic-loading {
      0% { background-position: 0% 50%; }
      100% { background-position: 200% 50%; }
    }

    .loading-title {
      font-family: 'Avant Garde', 'ITC Avant Garde Gothic', 'Questrial', sans-serif !important;
      font-size: 72px;
      font-weight: 900;
      letter-spacing: -2px;
      background: linear-gradient(
        90deg,
        #ff00ff,
        #ff44ff,
        #ffffff,
        #00bbff,
        #00ffff,
        #ffffff,
        #ff00ff,
        #ff44ff,
        #00bbff
      );
      background-size: 200% 100%;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      animation: holographic-loading 3s linear infinite;
    }

    @keyframes dots {
      0%, 20% { content: 'loading'; }
      40% { content: 'loading.'; }
      60% { content: 'loading..'; }
      80%, 100% { content: 'loading...'; }
    }

    .loading-text {
      margin-top: 30px;
      color: rgba(255, 255, 255, 0.3);
      font-size: 16px;
      letter-spacing: 2px;
      font-weight: 500;
    }

    .loading-text::after {
      content: 'loading';
      animation: dots 2s infinite;
    }

    /* Top Bar Holographic Animation */
    @keyframes holographic-bar {
      0% { background-position: 0% 50%; }
      100% { background-position: 100% 50%; }
    }

    @keyframes shine {
      0% { left: -100%; }
      100% { left: 200%; }
    }

    @keyframes sparkle {
      0%, 100% { opacity: 0; }
      50% { opacity: 1; }
    }

    .holographic-header {
      background: linear-gradient(
        90deg,
        #ff00ff,
        #ff44ff,
        #ffffff,
        #00bbff,
        #00ffff,
        #ffffff,
        #ff00ff,
        #ff44ff,
        #00bbff
      );
      background-size: 200% 100%;
      animation: holographic-bar 12s linear infinite;
      position: relative;
      overflow: hidden;
    }

    .holographic-header::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: radial-gradient(circle at 20% 50%, rgba(255,255,255,0.8) 0%, transparent 50%),
                  radial-gradient(circle at 80% 50%, rgba(255,255,255,0.8) 0%, transparent 50%),
                  radial-gradient(circle at 50% 50%, rgba(255,255,255,0.6) 0%, transparent 60%);
      mix-blend-mode: overlay;
      animation: sparkle 3s ease-in-out infinite;
    }

    .holographic-header::after {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 50%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.8), transparent);
      animation: shine 8s infinite;
      mix-blend-mode: overlay;
    }

    /* Glass Card */
    .liquid-glass-card {
      position: relative;
      background: rgba(0, 0, 0, 0.5);
      border-radius: 24px;
      padding: 32px;
      backdrop-filter: blur(20px);
      box-shadow: 
        inset 0 0 40px rgba(255, 255, 255, 0.05),
        0 0 40px rgba(255, 255, 255, 0.05);
      overflow: hidden;
    }

    .liquid-glass-card::before {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 24px;
      padding: 2px;
      background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.4),
        rgba(255, 255, 255, 0.1),
        rgba(255, 255, 255, 0.4)
      );
      -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      pointer-events: none;
    }

    .glass-content {
      position: relative;
      z-index: 1;
    }

    /* Holographic Text */
    .holographic-text {
      background: linear-gradient(
        90deg,
        #ffffff,
        #ff80ff,
        #80ffff,
        #ffffff,
        #ffff80,
        #ffffff
      );
      background-size: 200% 100%;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      animation: holographic-slide 8s linear infinite;
      font-weight: 800;
    }

    @keyframes holographic-slide {
      0% { background-position: 0% 50%; }
      100% { background-position: 200% 50%; }
    }

    /* Custom Dropdown */
    .custom-dropdown {
      position: relative;
      width: 100%;
    }

    .dropdown-header {
      background: rgba(0, 0, 0, 0.5);
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 12px;
      padding: 14px;
      color: white;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
    }

    .dropdown-header:hover {
      border-color: rgba(255, 255, 255, 0.5);
      background: rgba(255, 255, 255, 0.05);
    }

    .dropdown-list {
      position: absolute;
      top: calc(100% + 8px);
      left: 0;
      right: 0;
      background: rgba(0, 0, 0, 0.98);
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 12px;
      backdrop-filter: blur(20px);
      max-height: 300px;
      overflow-y: auto;
      z-index: 50000;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.9);
    }

    .dropdown-item {
      padding: 12px 16px;
      color: white;
      cursor: pointer;
      transition: all 0.2s ease;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      font-weight: 500;
    }

    .dropdown-item:last-child {
      border-bottom: none;
    }

    .dropdown-item:hover {
      background: rgba(255, 255, 255, 0.1);
      padding-left: 20px;
    }

    .dropdown-item.selected {
      background: rgba(255, 255, 255, 0.15);
      font-weight: 600;
    }

    /* Tab Buttons */
    .tab-button {
      background: rgba(0, 0, 0, 0.5);
      border: 2px solid rgba(255, 255, 255, 0.2);
      color: white;
      padding: 12px 24px;
      cursor: pointer;
      transition: all 0.3s ease;
      font-weight: 600;
      font-size: 13px;
      letter-spacing: 1px;
      backdrop-filter: blur(10px);
    }

    .tab-button:first-child {
      border-radius: 12px 0 0 12px;
      border-right: none;
    }

    .tab-button:last-child {
      border-radius: 0 12px 12px 0;
    }

    .tab-button.active {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.4);
    }

    /* Market Cap Buttons */
    .cap-button {
      background: transparent;
      border: 1px solid rgba(255, 255, 255, 0.2);
      color: white;
      padding: 10px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 1px;
      transition: all 0.3s ease;
    }

    .cap-button.active {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.4);
    }

    .cap-button:hover:not(.active) {
      background: rgba(255, 255, 255, 0.05);
    }

    .liquid-button {
      position: relative;
      background: rgba(0, 0, 0, 0.5);
      border: 3px solid rgba(255, 255, 255, 0.3);
      border-radius: 100px;
      padding: 16px 32px;
      font-weight: 600;
      font-size: 14px;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: white;
      cursor: pointer;
      overflow: hidden;
      transition: all 0.3s ease;
      backdrop-filter: blur(10px);
    }

    .liquid-button:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 30px rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.5);
      background: rgba(255, 255, 255, 0.05);
    }

    .liquid-button:disabled {
      opacity: 0.3;
      cursor: not-allowed;
    }

    .result-card {
      position: relative;
      background: rgba(0, 0, 0, 0.5);
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 16px;
      overflow: hidden;
      backdrop-filter: blur(10px);
      box-shadow: inset 0 0 20px rgba(255, 255, 255, 0.05);
    }

    .analysis-button {
      margin-top: 20px;
      background: transparent;
      border: 1px solid rgba(255, 255, 255, 0.3);
      color: white;
      padding: 8px 16px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 1px;
      transition: all 0.3s ease;
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .analysis-details {
      margin-top: 20px;
      padding: 20px;
      background: rgba(255, 255, 255, 0.02);
      border-radius: 12px;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .progress-indicator {
      padding: 12px 20px;
      background: rgba(255, 255, 255, 0.05);
      border-radius: 8px;
      margin-top: 12px;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    .logo-text {
      font-family: 'Avant Garde', 'ITC Avant Garde Gothic', 'Questrial', sans-serif !important;
      font-weight: 900;
      font-size: 36px;
      letter-spacing: -1px;
      color: #000000;
    }

    /* Mobile Responsive Fixes */
    @media (max-width: 768px) {
      .grid-main {
        grid-template-columns: 1fr !important;
      }
      
      .loading-title {
        font-size: 48px;
      }
      
      .logo-text {
        font-size: 22px !important;
      }
      
      .header-container {
        padding: 15px 20px !important;
      }
      
      .header-layout {
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 12px !important;
      }
      
      .header-info {
        width: 100%;
        display: flex !important;
        justify-content: space-between !important;
        font-size: 11px !important;
        gap: 10px !important;
      }
      
      .market-cap-grid {
        grid-template-columns: repeat(2, 1fr) !important;
      }
      
      .main-padding {
        padding: 20px !important;
      }
      
      .market-overview-grid {
        grid-template-columns: repeat(2, 1fr) !important;
      }
    }
  `;

  // Loading Screen
  if (isLoading) {
    return (
      <>
        <style>{styles}</style>
        <div className="loading-screen">
          <h1 className="loading-title">doDiligence</h1>
          <div className="loading-text"></div>
        </div>
      </>
    );
  }

  return (
    <>
      <style>{styles}</style>
      <div style={{
        minHeight: '100vh',
        background: '#000000',
        color: 'white',
        position: 'relative'
      }}>
        {/* Holographic Header */}
        <div className="holographic-header header-container" style={{ padding: '20px 40px', position: 'relative' }}>
          <div className="header-layout" style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            position: 'relative',
            zIndex: 2
          }}>
            <h1 className="logo-text">
              doDiligence
            </h1>
            <div className="header-info" style={{ 
              display: 'flex', 
              gap: '30px', 
              alignItems: 'center',
              color: '#000000',
              fontWeight: '600',
              fontSize: '14px'
            }}>
              <div>
                MARKET: <span style={{ fontWeight: '800' }}>{marketStatus}</span>
              </div>
              <div>
                {currentTime.toLocaleTimeString('en-US', { 
                  hour: '2-digit', 
                  minute: '2-digit', 
                  second: '2-digit' 
                })}
              </div>
              <div>
                {currentTime.toLocaleDateString('en-US', { 
                  month: 'short', 
                  day: 'numeric', 
                  year: 'numeric' 
                })}
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="main-padding" style={{ padding: '40px' }}>
          {/* Market Overview */}
          <div className="market-overview-grid" style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '24px',
            marginBottom: '40px'
          }}>
            {[
              { label: 'MARKET REGIME', value: marketConditions.regime },
              { label: 'FED STANCE', value: marketConditions.fedStance },
              { 
                label: 'VOLATILITY INDEX (VIX)', 
                value: marketConditions.vix,
                color: marketConditions.vix < 20 ? '#00ff88' : 
                       marketConditions.vix < 30 ? '#ffaa00' : '#ff3333'
              },
              { label: 'RECESSION RISK', value: marketConditions.recessionRisk }
            ].map((item, idx) => (
              <div key={idx} className="liquid-glass-card">
                <div className="glass-content">
                  <div style={{ 
                    fontSize: '11px', 
                    letterSpacing: '2px',
                    opacity: 0.5,
                    marginBottom: '16px',
                    fontWeight: '600'
                  }}>
                    {item.label}
                  </div>
                  <div style={{ 
                    fontSize: '28px',
                    fontWeight: '700',
                    color: item.color || '#ffffff'
                  }}>
                    {item.value}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Analysis Section */}
          <div className="grid-main" style={{ 
            display: 'grid', 
            gridTemplateColumns: '400px 1fr', 
            gap: '24px' 
          }}>
            {/* Controls */}
            <div className="liquid-glass-card" style={{ overflow: 'visible' }}>
              <div className="glass-content">
                <h2 style={{ 
                  fontSize: '18px', 
                  fontWeight: '700',
                  letterSpacing: '1px',
                  marginBottom: '32px'
                }}>
                  ANALYSIS CONTROL
                </h2>
                
                {/* Tab Buttons */}
                <div style={{ marginBottom: '24px', display: 'flex' }}>
                  <button 
                    className={`tab-button ${analysisType === 'sector' ? 'active' : ''}`}
                    onClick={() => setAnalysisType('sector')}
                  >
                    SECTOR
                  </button>
                  <button 
                    className={`tab-button ${analysisType === 'sub_industry' ? 'active' : ''}`}
                    onClick={() => setAnalysisType('sub_industry')}
                  >
                    SUB-INDUSTRY
                  </button>
                </div>

                {/* Market Cap Filter */}
                <div style={{ marginBottom: '24px' }}>
                  <label style={{ 
                    fontSize: '11px',
                    letterSpacing: '2px',
                    fontWeight: '600',
                    opacity: 0.5,
                    display: 'block',
                    marginBottom: '12px'
                  }}>
                    MARKET CAP FILTER
                  </label>
                  <div className="market-cap-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '8px' }}>
                    {[
                      { value: 'all', label: 'ALL CAPS' },
                      { value: 'large', label: 'LARGE CAP' },
                      { value: 'mid', label: 'MID CAP' },
                      { value: 'small', label: 'SMALL CAP' }
                    ].map(option => (
                      <button
                        key={option.value}
                        className={`cap-button ${marketCapSize === option.value ? 'active' : ''}`}
                        onClick={() => setMarketCapSize(option.value)}
                        style={{
                          background: marketCapSize === option.value 
                            ? 'rgba(255, 255, 255, 0.1)' 
                            : 'transparent',
                          border: `1px solid ${marketCapSize === option.value 
                            ? 'rgba(255, 255, 255, 0.4)' 
                            : 'rgba(255, 255, 255, 0.2)'}`,
                        }}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                  <div style={{ 
                    marginTop: '8px', 
                    fontSize: '10px', 
                    opacity: 0.5,
                    textAlign: 'center'
                  }}>
                    {marketCapSize === 'large' && '>$10B Market Cap'}
                    {marketCapSize === 'mid' && '$2B - $10B Market Cap'}
                    {marketCapSize === 'small' && '$300M - $2B Market Cap'}
                    {marketCapSize === 'all' && 'All Market Capitalizations'}
                  </div>
                </div>

                {/* Custom Dropdown */}
                <div style={{ marginBottom: '32px' }}>
                  <label style={{ 
                    fontSize: '11px',
                    letterSpacing: '2px',
                    fontWeight: '600',
                    opacity: 0.5,
                    display: 'block',
                    marginBottom: '8px'
                  }}>
                    SELECT TARGET
                  </label>
                  <div className="custom-dropdown" ref={dropdownRef}>
                    <div 
                      className="dropdown-header"
                      onClick={() => setDropdownOpen(!dropdownOpen)}
                    >
                      <span>{selectedTarget || 'Choose Target'}</span>
                      <span style={{ transform: dropdownOpen ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.3s' }}>▼</span>
                    </div>
                    {dropdownOpen && (
                      <div className="dropdown-list">
                        {(analysisType === 'sector' ? sectors : subIndustries).map(item => (
                          <div 
                            key={item} 
                            className={`dropdown-item ${selectedTarget === item ? 'selected' : ''}`}
                            onClick={() => {
                              setSelectedTarget(item);
                              setDropdownOpen(false);
                            }}
                          >
                            {item}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <button
                  className="liquid-button"
                  onClick={executeAnalysis}
                  disabled={isAnalyzing || !selectedTarget}
                  style={{ width: '100%' }}
                >
                  {isAnalyzing ? 'PROCESSING' : 'EXECUTE'}
                </button>

                {/* Progress Indicator */}
                {isAnalyzing && analysisProgress && (
                  <div className="progress-indicator">
                    <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.8)', animation: 'pulse 2s ease-in-out infinite' }}>
                      {analysisProgress}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Results */}
            <div className="liquid-glass-card">
              <div className="glass-content">
                <h2 style={{ 
                  fontSize: '18px',
                  fontWeight: '700',
                  letterSpacing: '1px',
                  marginBottom: '32px'
                }}>
                  RESULTS
                </h2>
                
                {results && results.top_stocks ? (
                  <div>
                    {results.top_stocks.map((stock, idx) => (
                      <div key={stock.symbol} className="result-card">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
                          <div style={{ flex: 1 }}>
                            <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', marginBottom: '20px' }}>
                              <div className="holographic-text" style={{ 
                                fontSize: '24px',
                                fontWeight: '800'
                              }}>
                                {stock.symbol}
                              </div>
                              {stock.market_cap && (
                                <div style={{ 
                                  fontSize: '12px', 
                                  opacity: 0.6,
                                  fontWeight: '500'
                                }}>
                                  {stock.market_cap}
                                </div>
                              )}
                            </div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
                              <div>
                                <div style={{ fontSize: '10px', opacity: 0.5, fontWeight: '600', letterSpacing: '1px' }}>CURRENT</div>
                                <div style={{ fontSize: '18px', fontWeight: '700', marginTop: '4px' }}>${stock.metrics.current_price}</div>
                              </div>
                              <div>
                                <div style={{ fontSize: '10px', opacity: 0.5, fontWeight: '600', letterSpacing: '1px' }}>TARGET</div>
                                <div style={{ fontSize: '18px', fontWeight: '700', marginTop: '4px' }}>${stock.metrics.target_price}</div>
                              </div>
                              <div>
                                <div style={{ fontSize: '10px', opacity: 0.5, fontWeight: '600', letterSpacing: '1px' }}>UPSIDE</div>
                                <div style={{ 
                                  fontSize: '18px',
                                  fontWeight: '700',
                                  marginTop: '4px',
                                  color: stock.metrics.upside_potential > 0 ? '#00ff88' : '#ff3333'
                                }}>
                                  {stock.metrics.upside_potential > 0 ? '+' : ''}{stock.metrics.upside_potential}%
                                </div>
                              </div>
                              <div>
                                <div style={{ fontSize: '10px', opacity: 0.5, fontWeight: '600', letterSpacing: '1px' }}>CONFIDENCE</div>
                                <div style={{ fontSize: '18px', fontWeight: '700', marginTop: '4px' }}>{stock.metrics.confidence_score}%</div>
                              </div>
                            </div>
                            
                            {/* Analysis Details Button */}
                            <button
                              className="analysis-button"
                              onClick={() => setExpandedStock(expandedStock === stock.symbol ? null : stock.symbol)}
                            >
                              {expandedStock === stock.symbol ? 'HIDE ANALYSIS' : 'VIEW FULL ANALYSIS'}
                              {expandedStock === stock.symbol ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                            </button>
                            
                            {/* Expanded Analysis Details */}
                            {expandedStock === stock.symbol && stock.analysis_details && (
                              <div className="analysis-details">
                                <h4 style={{ fontSize: '14px', marginBottom: '16px', opacity: 0.8 }}>FUNDAMENTAL ANALYSIS</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', marginBottom: '20px' }}>
                                  {Object.entries(stock.analysis_details.fundamentals || {}).map(([key, value]) => (
                                    <div key={key}>
                                      <span style={{ fontSize: '11px', opacity: 0.5 }}>{key.replace(/_/g, ' ').toUpperCase()}: </span>
                                      <span style={{ fontSize: '13px', fontWeight: '600' }}>
                                        {typeof value === 'number' ? value.toFixed(2) : value}
                                      </span>
                                    </div>
                                  ))}
                                </div>
                                
                                <h4 style={{ fontSize: '14px', marginBottom: '16px', opacity: 0.8 }}>TECHNICAL INDICATORS</h4>
                                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', marginBottom: '20px' }}>
                                  {Object.entries(stock.analysis_details.technicals || {}).map(([key, value]) => (
                                    <div key={key}>
                                      <span style={{ fontSize: '11px', opacity: 0.5 }}>{key.replace(/_/g, ' ').toUpperCase()}: </span>
                                      <span style={{ fontSize: '13px', fontWeight: '600' }}>
                                        {typeof value === 'number' ? value.toFixed(2) : value}
                                      </span>
                                    </div>
                                  ))}
                                </div>

                                {stock.investment_thesis && (
                                  <>
                                    <h4 style={{ fontSize: '14px', marginBottom: '16px', opacity: 0.8 }}>INVESTMENT THESIS</h4>
                                    <ul style={{ fontSize: '12px', lineHeight: '1.8', opacity: 0.9, paddingLeft: '20px' }}>
                                      {stock.investment_thesis.map((reason, idx) => (
                                        <li key={idx}>{reason}</li>
                                      ))}
                                    </ul>
                                  </>
                                )}
                              </div>
                            )}
                          </div>
                          <div style={{ 
                            fontSize: '48px',
                            fontWeight: '900',
                            opacity: 0.1
                          }}>
                            {idx + 1}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ 
                    textAlign: 'center',
                    padding: '80px 20px',
                    opacity: 0.3,
                    fontSize: '12px',
                    letterSpacing: '2px',
                    fontWeight: '600'
                  }}>
                    {isAnalyzing ? 'ANALYZING...' : 'NO DATA AVAILABLE'}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;