import React, { useState, useEffect, useRef, Suspense } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Environment, MeshDistortMaterial, Float, Text } from '@react-three/drei';
import { motion, AnimatePresence } from 'framer-motion';
import * as THREE from 'three';

const API_URL = window.location.hostname === 'localhost' 
  ? 'http://localhost:8000'
  : 'https://autoanalyst-dz11.onrender.com';

// Liquid Glass Orb Component
function LiquidOrb({ position, scale = 1, distort = 0.4 }) {
  const meshRef = useRef();
  
  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.x = Math.sin(state.clock.elapsedTime) * 0.2;
      meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.2;
    }
  });

  return (
    <Float speed={2} rotationIntensity={0.5} floatIntensity={0.5}>
      <mesh ref={meshRef} position={position} scale={scale}>
        <sphereGeometry args={[1, 64, 64]} />
        <MeshDistortMaterial
          color="#ffffff"
          envMapIntensity={0.9}
          clearcoat={1}
          clearcoatRoughness={0}
          metalness={0.9}
          roughness={0.1}
          distort={distort}
          speed={2}
          transparent
          opacity={0.15}
        />
      </mesh>
    </Float>
  );
}

// Glass Card Component with Refraction
const GlassCard = ({ children, className = '', style = {}, onClick, hover = true }) => {
  const [isHovered, setIsHovered] = useState(false);
  
  return (
    <motion.div
      className={`glass-card-extreme ${className}`}
      style={style}
      onClick={onClick}
      onMouseEnter={() => hover && setIsHovered(true)}
      onMouseLeave={() => hover && setIsHovered(false)}
      animate={{
        scale: isHovered ? 1.02 : 1,
        rotateX: isHovered ? 2 : 0,
        rotateY: isHovered ? -2 : 0,
      }}
      transition={{ type: "spring", stiffness: 300, damping: 20 }}
    >
      <div className="glass-refraction" />
      <div className="glass-content">
        {children}
      </div>
      <div className="glass-glow" />
    </motion.div>
  );
};

function App() {
  // State management
  const [currentTime, setCurrentTime] = useState(new Date());
  const [marketStatus, setMarketStatus] = useState('CHECKING');
  const [marketConditions, setMarketConditions] = useState({
    regime: 'Loading',
    fedStance: 'Loading',
    vix: 'Loading',
    recessionRisk: 'Loading'
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

  // Initialize
  useEffect(() => {
    const initialize = async () => {
      await wakeUpBackend();
      await checkDataStatus();
      await fetchMarketConditions();
      setInterval(fetchMarketConditions, 30000);
    };
    initialize();
  }, []);

  const wakeUpBackend = async () => {
    try {
      const response = await fetch(`${API_URL}/api/health`);
      return response.ok;
    } catch {
      return false;
    }
  };

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
        setDataReady(data.total_stocks > 0);
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

  // Extreme glass styles with refraction
  const styles = `
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@300;400;500;600;700&display=swap');
    
    * {
      font-family: 'Oswald', 'Neue Haas Grotesk', -apple-system, sans-serif;
      font-weight: 500;
    }

    body {
      background: #000000;
      overflow-x: hidden;
    }

    @keyframes holographic-shift {
      0% { 
        background-position: 0% 50%;
        filter: hue-rotate(0deg);
      }
      33% { 
        filter: hue-rotate(120deg);
      }
      66% { 
        filter: hue-rotate(240deg);
      }
      100% { 
        background-position: 100% 50%;
        filter: hue-rotate(360deg);
      }
    }

    @keyframes refraction {
      0% { transform: translateZ(0) rotateX(0deg); }
      50% { transform: translateZ(50px) rotateX(1deg); }
      100% { transform: translateZ(0) rotateX(0deg); }
    }

    .glass-card-extreme {
      position: relative;
      background: rgba(255, 255, 255, 0.02);
      backdrop-filter: blur(40px) saturate(200%);
      -webkit-backdrop-filter: blur(40px) saturate(200%);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 24px;
      overflow: hidden;
      transform-style: preserve-3d;
      perspective: 1000px;
    }

    .glass-card-extreme::before {
      content: '';
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      bottom: 0;
      background: linear-gradient(
        135deg,
        rgba(255, 255, 255, 0.1) 0%,
        transparent 40%,
        transparent 60%,
        rgba(255, 255, 255, 0.05) 100%
      );
      pointer-events: none;
      border-radius: 24px;
    }

    .glass-refraction {
      position: absolute;
      top: -50%;
      left: -50%;
      width: 200%;
      height: 200%;
      background: radial-gradient(
        circle at 30% 30%,
        rgba(255, 255, 255, 0.15) 0%,
        transparent 50%
      );
      filter: blur(40px);
      animation: refraction 8s ease-in-out infinite;
      pointer-events: none;
    }

    .glass-content {
      position: relative;
      z-index: 1;
      padding: 32px;
    }

    .glass-glow {
      position: absolute;
      top: -2px;
      left: -2px;
      right: -2px;
      bottom: -2px;
      background: linear-gradient(
        45deg,
        #ff00ff,
        #00ffff,
        #ffff00,
        #ff00ff
      );
      background-size: 400% 400%;
      opacity: 0;
      filter: blur(20px);
      transition: opacity 0.3s ease;
      animation: holographic-shift 6s ease infinite;
      z-index: -1;
      border-radius: 24px;
    }

    .glass-card-extreme:hover .glass-glow {
      opacity: 0.3;
    }

    .holographic-text-extreme {
      background: linear-gradient(
        135deg,
        #ffffff,
        #ff00ff,
        #00ffff,
        #ffffff,
        #ffff00,
        #ffffff
      );
      background-size: 300% 300%;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      animation: holographic-shift 4s ease infinite;
      font-weight: 700;
      text-shadow: 0 0 40px rgba(255, 255, 255, 0.3);
    }

    .liquid-button {
      position: relative;
      padding: 16px 40px;
      background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
      border: 2px solid rgba(255, 255, 255, 0.2);
      border-radius: 100px;
      color: white;
      font-weight: 600;
      font-size: 16px;
      letter-spacing: 1px;
      overflow: hidden;
      backdrop-filter: blur(20px);
      transition: all 0.3s ease;
      cursor: pointer;
      text-transform: uppercase;
    }

    .liquid-button::before {
      content: '';
      position: absolute;
      top: 50%;
      left: 50%;
      width: 0;
      height: 0;
      background: radial-gradient(circle, rgba(255, 255, 255, 0.3), transparent);
      transition: width 0.6s ease, height 0.6s ease;
      transform: translate(-50%, -50%);
      border-radius: 50%;
    }

    .liquid-button:hover::before {
      width: 300px;
      height: 300px;
    }

    .liquid-button:hover {
      border-color: rgba(255, 255, 255, 0.4);
      transform: translateY(-2px);
      box-shadow: 0 10px 40px rgba(255, 255, 255, 0.2);
    }

    .liquid-select {
      background: rgba(255, 255, 255, 0.03);
      backdrop-filter: blur(20px);
      border: 2px solid rgba(255, 255, 255, 0.1);
      border-radius: 16px;
      color: white;
      padding: 14px 18px;
      font-size: 15px;
      font-weight: 500;
      width: 100%;
      outline: none;
      transition: all 0.3s ease;
    }

    .liquid-select:focus {
      background: rgba(255, 255, 255, 0.06);
      border-color: rgba(255, 255, 255, 0.3);
      box-shadow: 0 0 30px rgba(255, 255, 255, 0.1);
    }

    .liquid-select option {
      background: #000000;
      color: white;
      font-weight: 500;
    }

    @media (max-width: 768px) {
      .glass-content {
        padding: 20px;
      }
      
      .grid-responsive {
        grid-template-columns: 1fr !important;
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
        {/* WebGL Background */}
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          zIndex: 0,
          pointerEvents: 'none'
        }}>
          <Canvas camera={{ position: [0, 0, 5], fov: 75 }}>
            <Suspense fallback={null}>
              <Environment preset="night" />
              <ambientLight intensity={0.2} />
              <pointLight position={[10, 10, 10]} intensity={0.5} />
              <LiquidOrb position={[-3, 2, -5]} scale={1.5} />
              <LiquidOrb position={[4, -2, -8]} scale={2} distort={0.6} />
              <LiquidOrb position={[0, 0, -10]} scale={3} distort={0.3} />
            </Suspense>
          </Canvas>
        </div>

        {/* Content */}
        <div style={{ position: 'relative', zIndex: 1 }}>
          {/* Header */}
          <GlassCard style={{ margin: '20px', borderRadius: '32px' }} hover={false}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h1 className="holographic-text-extreme" style={{
                fontSize: '48px',
                fontWeight: '700',
                letterSpacing: '-2px',
                margin: 0
              }}>
                doDiligence
              </h1>
              <div style={{ display: 'flex', gap: '32px', alignItems: 'center' }}>
                <div style={{ 
                  fontSize: '14px', 
                  fontWeight: '600',
                  letterSpacing: '2px',
                  opacity: 0.8 
                }}>
                  MARKET {marketStatus}
                </div>
              </div>
            </div>
          </GlassCard>

          {/* Main Content */}
          <div style={{ padding: '0 20px 20px' }}>
            {/* Market Overview */}
            <div className="grid-responsive" style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
              gap: '20px',
              marginBottom: '30px'
            }}>
              {[
                { label: 'REGIME', value: marketConditions.regime },
                { label: 'FED', value: marketConditions.fedStance },
                { label: 'VIX', value: marketConditions.vix },
                { label: 'RISK', value: marketConditions.recessionRisk }
              ].map((item, idx) => (
                <GlassCard key={idx}>
                  <div style={{ 
                    fontSize: '12px', 
                    letterSpacing: '3px',
                    opacity: 0.5,
                    marginBottom: '16px',
                    fontWeight: '600'
                  }}>
                    {item.label}
                  </div>
                  <div className="holographic-text-extreme" style={{ 
                    fontSize: '32px',
                    fontWeight: '700'
                  }}>
                    {item.value}
                  </div>
                </GlassCard>
              ))}
            </div>

            {/* Analysis Section */}
            <div className="grid-responsive" style={{ 
              display: 'grid', 
              gridTemplateColumns: '450px 1fr', 
              gap: '20px' 
            }}>
              {/* Controls */}
              <GlassCard>
                <h2 style={{ 
                  fontSize: '20px', 
                  fontWeight: '700',
                  letterSpacing: '2px',
                  marginBottom: '32px',
                  textTransform: 'uppercase'
                }}>
                  Analysis Control
                </h2>
                
                <div style={{ marginBottom: '24px' }}>
                  <label style={{ 
                    fontSize: '12px',
                    letterSpacing: '2px',
                    fontWeight: '600',
                    opacity: 0.6
                  }}>
                    TYPE
                  </label>
                  <select 
                    className="liquid-select"
                    value={analysisType}
                    onChange={(e) => setAnalysisType(e.target.value)}
                    style={{ marginTop: '8px' }}
                  >
                    <option value="sector">Sector Analysis</option>
                    <option value="sub_industry">Sub-Industry Analysis</option>
                  </select>
                </div>

                <div style={{ marginBottom: '32px' }}>
                  <label style={{ 
                    fontSize: '12px',
                    letterSpacing: '2px',
                    fontWeight: '600',
                    opacity: 0.6
                  }}>
                    TARGET
                  </label>
                  <select 
                    className="liquid-select"
                    value={selectedTarget}
                    onChange={(e) => setSelectedTarget(e.target.value)}
                    style={{ marginTop: '8px' }}
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
                  {isAnalyzing ? 'Processing...' : 'Execute'}
                </button>

                {isAnalyzing && (
                  <div style={{ marginTop: '24px' }}>
                    <div style={{
                      height: '8px',
                      background: 'rgba(255, 255, 255, 0.1)',
                      borderRadius: '100px',
                      overflow: 'hidden'
                    }}>
                      <motion.div
                        style={{
                          height: '100%',
                          background: 'linear-gradient(90deg, #ff00ff, #00ffff)',
                          borderRadius: '100px'
                        }}
                        animate={{ width: `${analysisProgress}%` }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                  </div>
                )}
              </GlassCard>

              {/* Results */}
              <GlassCard>
                <h2 style={{ 
                  fontSize: '20px',
                  fontWeight: '700',
                  letterSpacing: '2px',
                  marginBottom: '32px',
                  textTransform: 'uppercase'
                }}>
                  Results
                </h2>
                
                {results && results.top_stocks ? (
                  <AnimatePresence>
                    {results.top_stocks.map((stock, idx) => (
                      <motion.div
                        key={stock.symbol}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: idx * 0.1 }}
                      >
                        <GlassCard style={{ marginBottom: '16px' }}>
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <div>
                              <div className="holographic-text-extreme" style={{ 
                                fontSize: '28px',
                                fontWeight: '700',
                                marginBottom: '16px'
                              }}>
                                {stock.symbol}
                              </div>
                              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
                                <div>
                                  <div style={{ fontSize: '11px', opacity: 0.5, fontWeight: '600', letterSpacing: '1px' }}>CURRENT</div>
                                  <div style={{ fontSize: '20px', fontWeight: '700' }}>${stock.metrics.current_price}</div>
                                </div>
                                <div>
                                  <div style={{ fontSize: '11px', opacity: 0.5, fontWeight: '600', letterSpacing: '1px' }}>TARGET</div>
                                  <div style={{ fontSize: '20px', fontWeight: '700' }}>${stock.metrics.target_price}</div>
                                </div>
                                <div>
                                  <div style={{ fontSize: '11px', opacity: 0.5, fontWeight: '600', letterSpacing: '1px' }}>UPSIDE</div>
                                  <div style={{ 
                                    fontSize: '20px',
                                    fontWeight: '700',
                                    color: stock.metrics.upside_potential > 0 ? '#00ff88' : '#ff0044'
                                  }}>
                                    {stock.metrics.upside_potential}%
                                  </div>
                                </div>
                                <div>
                                  <div style={{ fontSize: '11px', opacity: 0.5, fontWeight: '600', letterSpacing: '1px' }}>CONFIDENCE</div>
                                  <div style={{ fontSize: '20px', fontWeight: '700' }}>{stock.metrics.confidence_score}%</div>
                                </div>
                              </div>
                            </div>
                            <div className="holographic-text-extreme" style={{ fontSize: '48px', opacity: 0.3 }}>
                              {idx + 1}
                            </div>
                          </div>
                        </GlassCard>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                ) : (
                  <div style={{ 
                    textAlign: 'center',
                    padding: '80px 20px',
                    opacity: 0.3,
                    fontSize: '14px',
                    letterSpacing: '2px',
                    fontWeight: '600'
                  }}>
                    NO DATA AVAILABLE
                  </div>
                )}
              </GlassCard>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;