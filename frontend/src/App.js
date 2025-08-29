import React, { useState, useEffect, useRef } from 'react';
import { TrendingUp, Activity, BarChart3, ChevronRight } from 'lucide-react';

const API_URL = window.location.hostname === 'localhost' 
  ? 'http://localhost:8000'
  : 'https://autoanalyst-dz11.onrender.com';

// Holographic Shard Component
const HolographicShard = ({ top, left, size = 100, rotation = 0, delay = 0 }) => {
  return (
    <div 
      className="holographic-shard"
      style={{
        position: 'absolute',
        top: `${top}%`,
        left: `${left}%`,
        width: `${size}px`,
        height: `${size}px`,
        transform: `rotate(${rotation}deg)`,
        animationDelay: `${delay}s`,
        pointerEvents: 'none',
        zIndex: 1
      }}
    />
  );
};

function App() {
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
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);

  useEffect(() => {
    const initialize = async () => {
      await fetchMarketConditions();
      await checkDataStatus();
    };
    initialize();
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
      }
    } catch (error) {
      console.error('Analysis failed:', error);
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
    }
    
    body {
      font-family: 'Avant Garde', 'ITC Avant Garde Gothic', 'Questrial', -apple-system, sans-serif;
      background: #000000;
      overflow-x: hidden;
    }

    /* Holographic Shard Animation */
    @keyframes float-shard {
      0%, 100% { transform: translateY(0px) rotate(0deg); }
      33% { transform: translateY(-20px) rotate(120deg); }
      66% { transform: translateY(10px) rotate(240deg); }
    }

    @keyframes holographic-shift {
      0%, 100% { filter: hue-rotate(0deg) brightness(1); }
      25% { filter: hue-rotate(90deg) brightness(1.2); }
      50% { filter: hue-rotate(180deg) brightness(0.9); }
      75% { filter: hue-rotate(270deg) brightness(1.1); }
    }

    .holographic-shard {
      background: conic-gradient(
        from 0deg at 50% 50%,
        #ff00ff,
        #00ffff,
        #ffff00,
        #ff00ff,
        #00ffff,
        #ff00ff
      );
      clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);
      animation: float-shard 6s ease-in-out infinite, holographic-shift 4s linear infinite;
      opacity: 0.6;
      filter: blur(0.5px);
    }

    /* Liquid Glass Card */
    .liquid-glass-card {
      position: relative;
      background: #000000;
      border-radius: 24px;
      padding: 32px;
      overflow: hidden;
    }

    .liquid-glass-card::before {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 24px;
      padding: 2px;
      background: conic-gradient(
        from 180deg at 50% 50%,
        rgba(255, 255, 255, 0.1),
        rgba(255, 255, 255, 0.3),
        rgba(255, 255, 255, 0.1),
        rgba(255, 255, 255, 0.3),
        rgba(255, 255, 255, 0.1)
      );
      -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
      animation: rotate-border 4s linear infinite;
    }

    @keyframes rotate-border {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    .liquid-glass-card::after {
      content: '';
      position: absolute;
      inset: 2px;
      background: rgba(0, 0, 0, 0.95);
      border-radius: 22px;
      backdrop-filter: blur(40px);
      z-index: -1;
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
        #ff00ff,
        #00ffff,
        #ffffff,
        #ffff00,
        #ffffff,
        #ff00ff
      );
      background-size: 200% 100%;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      animation: holographic-slide 3s linear infinite;
      font-weight: 800;
    }

    @keyframes holographic-slide {
      0% { background-position: 0% 50%; }
      100% { background-position: 200% 50%; }
    }

    /* Logo specific styling */
    .logo-text {
      font-family: 'Avant Garde', 'ITC Avant Garde Gothic', 'Questrial', sans-serif;
      font-weight: 900;
      font-size: 64px;
      letter-spacing: -2px;
    }

    /* Liquid Button */
    .liquid-button {
      position: relative;
      background: #000000;
      border: 2px solid transparent;
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
    }

    .liquid-button::before {
      content: '';
      position: absolute;
      inset: 0;
      border-radius: 100px;
      padding: 2px;
      background: linear-gradient(90deg, rgba(255,255,255,0.2), rgba(255,255,255,0.4), rgba(255,255,255,0.2));
      -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
      -webkit-mask-composite: xor;
      mask-composite: exclude;
    }

    .liquid-button:hover {
      transform: translateY(-2px);
      box-shadow: 0 10px 30px rgba(255, 255, 255, 0.1);
    }

    .liquid-button:disabled {
      opacity: 0.3;
      cursor: not-allowed;
    }

    /* Liquid Select */
    .liquid-select {
      width: 100%;
      background: #000000;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 14px;
      color: white;
      font-size: 14px;
      font-weight: 500;
      outline: none;
      transition: all 0.3s ease;
    }

    .liquid-select:focus {
      border-color: rgba(255, 255, 255, 0.3);
      box-shadow: 0 0 20px rgba(255, 255, 255, 0.05);
    }

    .liquid-select option {
      background: #000000;
      color: white;
    }

    /* Result Card */
    .result-card {
      position: relative;
      background: #000000;
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 16px;
      padding: 24px;
      margin-bottom: 16px;
      overflow: hidden;
    }

    .result-card::before {
      content: '';
      position: absolute;
      top: -2px;
      left: -2px;
      right: -2px;
      bottom: -2px;
      background: linear-gradient(45deg, transparent, rgba(255,0,255,0.2), transparent, rgba(0,255,255,0.2));
      border-radius: 16px;
      opacity: 0;
      transition: opacity 0.3s ease;
      z-index: -1;
    }

    .result-card:hover::before {
      opacity: 1;
    }

    @media (max-width: 768px) {
      .grid-main {
        grid-template-columns: 1fr !important;
      }
      .logo-text {
        font-size: 48px;
      }
    }
  `;

  return (
    <>
      <style>{styles}</style>
      <div style={{
        minHeight: '100vh',
        background: '#000000',
        color: 'white',
        position: 'relative'
      }}>
        {/* Holographic Shards scattered around */}
        <HolographicShard top={10} left={5} size={60} rotation={45} delay={0} />
        <HolographicShard top={20} left={85} size={80} rotation={135} delay={1} />
        <HolographicShard top={50} left={3} size={70} rotation={225} delay={2} />
        <HolographicShard top={70} left={90} size={50} rotation={315} delay={3} />
        <HolographicShard top={80} left={15} size={65} rotation={90} delay={4} />

        {/* Header */}
        <div style={{ padding: '40px 40px 20px' }}>
          <h1 className="holographic-text logo-text">
            doDiligence
          </h1>
        </div>

        {/* Main Content */}
        <div style={{ padding: '0 40px 40px' }}>
          {/* Market Overview */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
            gap: '24px',
            marginBottom: '40px'
          }}>
            {[
              { label: 'MARKET REGIME', value: marketConditions.regime },
              { label: 'FED STANCE', value: marketConditions.fedStance },
              { label: 'VOLATILITY INDEX', value: marketConditions.vix },
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
                    color: '#ffffff'
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
            <div className="liquid-glass-card">
              <div className="glass-content">
                <h2 style={{ 
                  fontSize: '18px', 
                  fontWeight: '700',
                  letterSpacing: '1px',
                  marginBottom: '32px'
                }}>
                  ANALYSIS CONTROL
                </h2>
                
                <div style={{ marginBottom: '24px' }}>
                  <label style={{ 
                    fontSize: '11px',
                    letterSpacing: '2px',
                    fontWeight: '600',
                    opacity: 0.5,
                    display: 'block',
                    marginBottom: '8px'
                  }}>
                    TYPE
                  </label>
                  <select 
                    className="liquid-select"
                    value={analysisType}
                    onChange={(e) => setAnalysisType(e.target.value)}
                  >
                    <option value="sector">Sector Analysis</option>
                    <option value="sub_industry">Sub-Industry Analysis</option>
                  </select>
                </div>

                <div style={{ marginBottom: '32px' }}>
                  <label style={{ 
                    fontSize: '11px',
                    letterSpacing: '2px',
                    fontWeight: '600',
                    opacity: 0.5,
                    display: 'block',
                    marginBottom: '8px'
                  }}>
                    TARGET
                  </label>
                  <select 
                    className="liquid-select"
                    value={selectedTarget}
                    onChange={(e) => setSelectedTarget(e.target.value)}
                  >
                    <option value="">Select Target</option>
                    {(analysisType === 'sector' ? sectors : subIndustries).map(item => (
                      <option key={item} value={item}>{item}</option>
                    ))}
                  </select>
                </div>

                <button
                  className="liquid-button"
                  onClick={executeAnalysis}
                  disabled={isAnalyzing || !selectedTarget}
                  style={{ width: '100%' }}
                >
                  {isAnalyzing ? 'PROCESSING' : 'EXECUTE'}
                </button>
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
                            <div className="holographic-text" style={{ 
                              fontSize: '24px',
                              fontWeight: '800',
                              marginBottom: '20px'
                            }}>
                              {stock.symbol}
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
                    NO DATA AVAILABLE
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