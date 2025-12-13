import React, { useState, useEffect } from 'react';
import { TrendingUp, Activity, BarChart3, DollarSign } from 'lucide-react';

// Determine API URL based on environment
const getApiUrl = () => {
  if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    return 'http://localhost:8000';
  }
  // For Render.com deployment - update this to your actual backend URL
  return process.env.REACT_APP_API_URL || 'https://autoanalyst-dz11.onrender.com';
};

const API_URL = getApiUrl();

function App() {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState('CHECKING');
  const [tickerInput, setTickerInput] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [analysisProgress, setAnalysisProgress] = useState('');
  const [isLoading, setIsLoading] = useState(true);

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
    setTimeout(() => setIsLoading(false), 1500);
  }, []);

  const executeAnalysis = async () => {
    if (!tickerInput.trim()) return;

    const ticker = tickerInput.trim().toUpperCase();
    setIsAnalyzing(true);
    setResults(null);
    setAnalysisProgress('Initializing stock analysis...');

    try {
      const response = await fetch(`${API_URL}/api/stock-analysis`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker })
      });

      if (!response.ok) {
        throw new Error('Failed to analyze stock');
      }

      const data = await response.json();
      setResults(data);
      setAnalysisProgress('');
      setIsAnalyzing(false);

    } catch (error) {
      console.error('Analysis error:', error);
      setAnalysisProgress('Failed to analyze stock - please check the ticker and try again');
      setIsAnalyzing(false);
      setTimeout(() => setAnalysisProgress(''), 5000);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !isAnalyzing && tickerInput.trim()) {
      executeAnalysis();
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

    .ticker-input {
      background: rgba(0, 0, 0, 0.5);
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-radius: 12px;
      padding: 16px;
      color: white;
      font-size: 16px;
      font-weight: 600;
      letter-spacing: 2px;
      width: 100%;
      text-transform: uppercase;
      backdrop-filter: blur(10px);
      transition: all 0.3s ease;
    }

    .ticker-input:focus {
      outline: none;
      border-color: rgba(255, 255, 255, 0.6);
      background: rgba(255, 255, 255, 0.05);
    }

    .ticker-input::placeholder {
      color: rgba(255, 255, 255, 0.3);
      letter-spacing: 1px;
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

    .liquid-button:hover:not(:disabled) {
      transform: translateY(-2px);
      box-shadow: 0 10px 30px rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.5);
      background: rgba(255, 255, 255, 0.05);
    }

    .liquid-button:disabled {
      opacity: 0.3;
      cursor: not-allowed;
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

    /* Chart Bars */
    .chart-bar {
      background: linear-gradient(90deg, rgba(255,255,255,0.2), rgba(255,255,255,0.05));
      border-radius: 4px;
      height: 100%;
      position: relative;
      overflow: hidden;
    }

    .chart-bar::before {
      content: '';
      position: absolute;
      top: 0;
      left: -100%;
      width: 100%;
      height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
      animation: shimmer 2s infinite;
    }

    @keyframes shimmer {
      100% { left: 100%; }
    }

    /* Mobile Responsive */
    @media (max-width: 768px) {
      .dashboard-grid {
        grid-template-columns: 1fr !important;
      }

      .loading-title {
        font-size: 48px;
      }

      .logo-text {
        font-size: 22px !important;
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
        <div className="holographic-header" style={{ padding: '20px 40px', position: 'relative' }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            position: 'relative',
            zIndex: 2
          }}>
            <h1 className="logo-text">
              doDiligence
            </h1>
            <div style={{
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
        <div style={{ padding: '40px' }}>
          {/* Analysis Section */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '400px 1fr',
            gap: '24px',
            marginBottom: '40px'
          }}>
            {/* Controls */}
            <div className="liquid-glass-card">
              <div className="glass-content">
                <h2 style={{
                  fontSize: '18px',
                  fontWeight: '700',
                  letterSpacing: '1px',
                  marginBottom: '32px'
                }}>
                  STOCK ANALYZER
                </h2>

                <div style={{ marginBottom: '24px' }}>
                  <label style={{
                    fontSize: '11px',
                    letterSpacing: '2px',
                    fontWeight: '600',
                    opacity: 0.5,
                    display: 'block',
                    marginBottom: '12px'
                  }}>
                    ENTER STOCK TICKER
                  </label>
                  <input
                    type="text"
                    className="ticker-input"
                    placeholder="e.g., AAPL, TSLA, MSFT"
                    value={tickerInput}
                    onChange={(e) => setTickerInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    disabled={isAnalyzing}
                  />
                </div>

                <button
                  className="liquid-button"
                  onClick={executeAnalysis}
                  disabled={isAnalyzing || !tickerInput.trim()}
                  style={{ width: '100%' }}
                >
                  {isAnalyzing ? 'ANALYZING' : 'ANALYZE STOCK'}
                </button>

                {analysisProgress && (
                  <div className="progress-indicator">
                    <div style={{ fontSize: '13px', color: 'rgba(255, 255, 255, 0.8)', animation: 'pulse 2s ease-in-out infinite' }}>
                      {analysisProgress}
                    </div>
                  </div>
                )}

                {/* Info Box */}
                <div style={{
                  marginTop: '32px',
                  padding: '20px',
                  background: 'rgba(255, 255, 255, 0.02)',
                  borderRadius: '12px',
                  border: '1px solid rgba(255, 255, 255, 0.1)'
                }}>
                  <div style={{ fontSize: '11px', letterSpacing: '1px', opacity: 0.6, marginBottom: '12px' }}>
                    ANALYSIS INCLUDES:
                  </div>
                  <div style={{ fontSize: '12px', lineHeight: '1.8', opacity: 0.7 }}>
                    • DCF Valuation (6-Step)<br/>
                    • Revenue Forecasting (5-Year)<br/>
                    • Comparable Companies<br/>
                    • ML Synthesis & Recommendation
                  </div>
                </div>
              </div>
            </div>

            {/* Quick Results Overview */}
            {results && results.executive_summary ? (
              <div className="liquid-glass-card">
                <div className="glass-content">
                  <div style={{ marginBottom: '32px' }}>
                    <div className="holographic-text" style={{ fontSize: '42px', fontWeight: '900', marginBottom: '8px' }}>
                      {results.ticker}
                    </div>
                    <div style={{ fontSize: '18px', opacity: 0.7, marginBottom: '4px' }}>
                      {results.executive_summary.company_name}
                    </div>
                    <div style={{ fontSize: '13px', opacity: 0.5, letterSpacing: '1px' }}>
                      {results.executive_summary.sector}
                    </div>
                  </div>

                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                    gap: '20px',
                    marginBottom: '32px'
                  }}>
                    <div>
                      <div style={{ fontSize: '10px', opacity: 0.5, fontWeight: '600', letterSpacing: '1px', marginBottom: '8px' }}>
                        CURRENT PRICE
                      </div>
                      <div style={{ fontSize: '28px', fontWeight: '700' }}>
                        ${results.executive_summary.current_price?.toFixed(2) || 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '10px', opacity: 0.5, fontWeight: '600', letterSpacing: '1px', marginBottom: '8px' }}>
                        TARGET PRICE
                      </div>
                      <div style={{ fontSize: '28px', fontWeight: '700', color: '#00ff88' }}>
                        ${results.executive_summary.consensus_target?.toFixed(2) || 'N/A'}
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: '10px', opacity: 0.5, fontWeight: '600', letterSpacing: '1px', marginBottom: '8px' }}>
                        UPSIDE POTENTIAL
                      </div>
                      <div style={{
                        fontSize: '28px',
                        fontWeight: '700',
                        color: results.executive_summary.recommendation?.upside_potential > 0 ? '#00ff88' : '#ff3333'
                      }}>
                        {results.executive_summary.recommendation?.upside_potential > 0 ? '+' : ''}
                        {(results.executive_summary.recommendation?.upside_potential * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  {/* Recommendation Badge */}
                  <div style={{
                    padding: '20px',
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '12px',
                    border: '2px solid rgba(255, 255, 255, 0.2)',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '11px', opacity: 0.5, marginBottom: '8px', letterSpacing: '2px' }}>
                      FINAL RECOMMENDATION
                    </div>
                    <div style={{
                      fontSize: '24px',
                      fontWeight: '900',
                      letterSpacing: '2px',
                      color: results.executive_summary.recommendation?.action?.includes('BUY') ? '#00ff88' :
                             results.executive_summary.recommendation?.action?.includes('HOLD') ? '#ffaa00' : '#ff3333'
                    }}>
                      {results.executive_summary.recommendation?.action || 'N/A'}
                    </div>
                  </div>
                </div>
              </div>
            ) : !isAnalyzing && (
              <div className="liquid-glass-card">
                <div className="glass-content" style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: '300px',
                  opacity: 0.3,
                  fontSize: '14px',
                  letterSpacing: '2px',
                  fontWeight: '600'
                }}>
                  ENTER A TICKER TO BEGIN ANALYSIS
                </div>
              </div>
            )}
          </div>

          {/* Detailed Dashboard */}
          {results && results.dcf_analysis && (
            <div className="dashboard-grid" style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(2, 1fr)',
              gap: '24px'
            }}>
              {/* DCF Analysis Card */}
              <div className="liquid-glass-card">
                <div className="glass-content">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                    <TrendingUp size={24} style={{ opacity: 0.7 }} />
                    <h3 style={{ fontSize: '18px', fontWeight: '700', letterSpacing: '1px' }}>
                      DCF VALUATION
                    </h3>
                  </div>

                  <div style={{ marginBottom: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <span style={{ fontSize: '13px', opacity: 0.6 }}>Intrinsic Value</span>
                      <span style={{ fontSize: '16px', fontWeight: '700' }}>
                        ${results.dcf_analysis.valuation?.intrinsic_value_per_share?.toFixed(2) || 'N/A'}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <span style={{ fontSize: '13px', opacity: 0.6 }}>WACC</span>
                      <span style={{ fontSize: '16px', fontWeight: '700' }}>
                        {(results.dcf_analysis.discount_rate?.wacc * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <span style={{ fontSize: '13px', opacity: 0.6 }}>Premium/(Discount)</span>
                      <span style={{
                        fontSize: '16px',
                        fontWeight: '700',
                        color: results.dcf_analysis.valuation?.premium_discount > 0 ? '#00ff88' : '#ff3333'
                      }}>
                        {(results.dcf_analysis.valuation?.premium_discount * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Cash Flow Visualization */}
                  {results.dcf_analysis.cash_flows?.forecasts && results.dcf_analysis.cash_flows.forecasts.length > 0 && (
                    <div style={{ marginTop: '20px' }}>
                      <div style={{ fontSize: '11px', opacity: 0.5, marginBottom: '12px', letterSpacing: '1px' }}>
                        5-YEAR CASH FLOW FORECAST
                      </div>
                      <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end', height: '80px' }}>
                        {results.dcf_analysis.cash_flows.forecasts.slice(0, 5).map((cf, idx) => {
                          const maxCf = Math.max(...results.dcf_analysis.cash_flows.forecasts.slice(0, 5).map(Math.abs));
                          const height = maxCf > 0 ? (Math.abs(cf) / maxCf) * 100 : 10;
                          return (
                            <div key={idx} style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end' }}>
                              <div
                                className="chart-bar"
                                style={{
                                  height: `${height}%`,
                                  minHeight: '10px'
                                }}
                              />
                              <div style={{ fontSize: '10px', opacity: 0.4, marginTop: '4px', textAlign: 'center' }}>
                                Y{idx + 1}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  <div style={{
                    marginTop: '20px',
                    padding: '12px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    borderRadius: '8px',
                    fontSize: '12px',
                    opacity: 0.7,
                    lineHeight: '1.6'
                  }}>
                    {results.dcf_analysis.recommendation}
                  </div>
                </div>
              </div>

              {/* Revenue Forecast Card */}
              <div className="liquid-glass-card">
                <div className="glass-content">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                    <BarChart3 size={24} style={{ opacity: 0.7 }} />
                    <h3 style={{ fontSize: '18px', fontWeight: '700', letterSpacing: '1px' }}>
                      REVENUE FORECAST
                    </h3>
                  </div>

                  <div style={{ marginBottom: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <span style={{ fontSize: '13px', opacity: 0.6 }}>Revenue CAGR</span>
                      <span style={{ fontSize: '16px', fontWeight: '700' }}>
                        {(results.revenue_forecast.growth_analysis?.cagr * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <span style={{ fontSize: '13px', opacity: 0.6 }}>Growth Trend</span>
                      <span style={{ fontSize: '16px', fontWeight: '700' }}>
                        {results.revenue_forecast.growth_analysis?.trend || 'N/A'}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <span style={{ fontSize: '13px', opacity: 0.6 }}>Current Revenue</span>
                      <span style={{ fontSize: '16px', fontWeight: '700' }}>
                        ${(results.revenue_forecast.forecast?.current_revenue / 1e9).toFixed(2)}B
                      </span>
                    </div>
                  </div>

                  {/* Revenue Growth Visualization */}
                  {results.revenue_forecast.forecast?.forecasted_revenue && results.revenue_forecast.forecast.forecasted_revenue.length > 0 && (
                    <div style={{ marginTop: '20px' }}>
                      <div style={{ fontSize: '11px', opacity: 0.5, marginBottom: '12px', letterSpacing: '1px' }}>
                        5-YEAR REVENUE PROJECTION
                      </div>
                      <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-end', height: '80px' }}>
                        {results.revenue_forecast.forecast.forecasted_revenue.slice(0, 5).map((rev, idx) => {
                          const maxRev = Math.max(...results.revenue_forecast.forecast.forecasted_revenue.slice(0, 5));
                          const height = maxRev > 0 ? (rev / maxRev) * 100 : 10;
                          return (
                            <div key={idx} style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'flex-end' }}>
                              <div
                                className="chart-bar"
                                style={{
                                  height: `${height}%`,
                                  minHeight: '10px',
                                  background: 'linear-gradient(90deg, rgba(0,255,136,0.3), rgba(0,255,136,0.1))'
                                }}
                              />
                              <div style={{ fontSize: '10px', opacity: 0.4, marginTop: '4px', textAlign: 'center' }}>
                                Y{idx + 1}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  <div style={{
                    marginTop: '20px',
                    padding: '12px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    borderRadius: '8px',
                    fontSize: '12px',
                    opacity: 0.7,
                    lineHeight: '1.6'
                  }}>
                    {results.revenue_forecast.recommendation}
                  </div>
                </div>
              </div>

              {/* Comparable Companies Card */}
              <div className="liquid-glass-card">
                <div className="glass-content">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                    <Activity size={24} style={{ opacity: 0.7 }} />
                    <h3 style={{ fontSize: '18px', fontWeight: '700', letterSpacing: '1px' }}>
                      PEER COMPARISON
                    </h3>
                  </div>

                  <div style={{ marginBottom: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <span style={{ fontSize: '13px', opacity: 0.6 }}>Peer Group Size</span>
                      <span style={{ fontSize: '16px', fontWeight: '700' }}>
                        {results.comparable_companies.peer_group?.length || 0} companies
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <span style={{ fontSize: '13px', opacity: 0.6 }}>Implied Price</span>
                      <span style={{ fontSize: '16px', fontWeight: '700' }}>
                        ${results.comparable_companies.valuation?.overall_assessment?.average_implied_price?.toFixed(2) || 'N/A'}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <span style={{ fontSize: '13px', opacity: 0.6 }}>Relative Valuation</span>
                      <span style={{
                        fontSize: '16px',
                        fontWeight: '700',
                        color: results.comparable_companies.valuation?.overall_assessment?.overall_upside_downside > 0 ? '#00ff88' : '#ff3333'
                      }}>
                        {(results.comparable_companies.valuation?.overall_assessment?.overall_upside_downside * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  {/* Peer Companies List */}
                  <div style={{ marginTop: '20px' }}>
                    <div style={{ fontSize: '11px', opacity: 0.5, marginBottom: '12px', letterSpacing: '1px' }}>
                      PEER COMPANIES
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                      {results.comparable_companies.peer_group?.slice(1, 5).map((peer, idx) => (
                        <div key={idx} style={{
                          padding: '6px 12px',
                          background: 'rgba(255, 255, 255, 0.05)',
                          borderRadius: '6px',
                          fontSize: '11px',
                          fontWeight: '600',
                          letterSpacing: '1px'
                        }}>
                          {peer}
                        </div>
                      ))}
                    </div>
                  </div>

                  <div style={{
                    marginTop: '20px',
                    padding: '12px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    borderRadius: '8px',
                    fontSize: '12px',
                    opacity: 0.7,
                    lineHeight: '1.6'
                  }}>
                    {results.comparable_companies.recommendation}
                  </div>
                </div>
              </div>

              {/* ML Synthesis Card */}
              <div className="liquid-glass-card">
                <div className="glass-content">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '24px' }}>
                    <DollarSign size={24} style={{ opacity: 0.7 }} />
                    <h3 style={{ fontSize: '18px', fontWeight: '700', letterSpacing: '1px' }}>
                      ML SYNTHESIS
                    </h3>
                  </div>

                  <div style={{ marginBottom: '24px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <span style={{ fontSize: '13px', opacity: 0.6 }}>ML Score</span>
                      <span style={{ fontSize: '16px', fontWeight: '700' }}>
                        {results.ml_synthesis.ml_score?.toFixed(3) || 'N/A'}
                      </span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
                      <span style={{ fontSize: '13px', opacity: 0.6 }}>Model Consensus</span>
                      <span style={{ fontSize: '16px', fontWeight: '700' }}>
                        ${results.ml_synthesis.target_price?.toFixed(2) || 'N/A'}
                      </span>
                    </div>
                  </div>

                  {/* Model Weights Visualization */}
                  <div style={{ marginTop: '20px' }}>
                    <div style={{ fontSize: '11px', opacity: 0.5, marginBottom: '12px', letterSpacing: '1px' }}>
                      MODEL WEIGHTS
                    </div>
                    {Object.entries(results.ml_synthesis.model_weights || {}).map(([model, weight]) => (
                      <div key={model} style={{ marginBottom: '12px' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                          <span style={{ fontSize: '11px', opacity: 0.6, textTransform: 'uppercase' }}>{model}</span>
                          <span style={{ fontSize: '11px', opacity: 0.8 }}>{(weight * 100).toFixed(0)}%</span>
                        </div>
                        <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px' }}>
                          <div style={{
                            width: `${weight * 100}%`,
                            height: '100%',
                            background: 'linear-gradient(90deg, rgba(255,255,255,0.4), rgba(255,255,255,0.2))',
                            borderRadius: '3px'
                          }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Price Range */}
                  <div style={{
                    marginTop: '20px',
                    padding: '16px',
                    background: 'rgba(255, 255, 255, 0.02)',
                    borderRadius: '8px'
                  }}>
                    <div style={{ fontSize: '11px', opacity: 0.5, marginBottom: '12px', letterSpacing: '1px' }}>
                      PRICE SCENARIOS
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '12px' }}>
                      <div>
                        <div style={{ fontSize: '10px', opacity: 0.4 }}>BULL</div>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#00ff88' }}>
                          ${results.executive_summary.recommendation?.price_range?.bull_case?.toFixed(2)}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', opacity: 0.4 }}>BASE</div>
                        <div style={{ fontSize: '14px', fontWeight: '700' }}>
                          ${results.executive_summary.recommendation?.price_range?.base_case?.toFixed(2)}
                        </div>
                      </div>
                      <div>
                        <div style={{ fontSize: '10px', opacity: 0.4 }}>BEAR</div>
                        <div style={{ fontSize: '14px', fontWeight: '700', color: '#ff3333' }}>
                          ${results.executive_summary.recommendation?.price_range?.bear_case?.toFixed(2)}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Full Analysis Text */}
              <div className="liquid-glass-card" style={{ gridColumn: '1 / -1' }}>
                <div className="glass-content">
                  <h3 style={{ fontSize: '18px', fontWeight: '700', letterSpacing: '1px', marginBottom: '20px' }}>
                    COMPREHENSIVE ANALYSIS
                  </h3>
                  <div style={{
                    fontSize: '13px',
                    lineHeight: '1.8',
                    opacity: 0.7,
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'monospace'
                  }}>
                    {results.ml_synthesis.comprehensive_analysis}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App;
