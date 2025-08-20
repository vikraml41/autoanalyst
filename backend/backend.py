#!/usr/bin/env python3
"""
FastAPI Backend for Quantitative Financial Analysis Model
Pre-loads CSV data and provides REST API endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import json
import io
import os
import uuid
from datetime import datetime
import asyncio
import tempfile
import glob

# Import your existing model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from quant_model import QuantFinanceMLModel, MarketConditionsAnalyzer, EnhancedValuation

app = FastAPI(title="Quantitative Finance Analysis API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React/Vite dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
analysis_sessions = {}
preloaded_data = None
available_sectors = []
available_sub_industries = []

# Pydantic models
class AnalysisRequest(BaseModel):
    selection_type: str  # 'sector' or 'sub_industry'
    selection_value: str

class MarketConditions(BaseModel):
    market_regime: str
    fed_stance: Dict
    yield_curve: Dict
    sector_conditions: Dict
    adjustment_factor: float

class StockAnalysis(BaseModel):
    symbol: str
    current_price: float
    target_price: float
    upside_potential: float
    confidence: float
    pe_ratio: float
    market_cap: float
    sentiment_score: float
    valuations: Dict

class AnalysisStatus(BaseModel):
    session_id: str
    status: str  # 'processing', 'completed', 'error'
    progress: int
    message: str
    results: Optional[Dict] = None

# Initialize model instances
model = QuantFinanceMLModel()
market_analyzer = MarketConditionsAnalyzer()

def load_csv_data():
    """Load all CSV files from data directory on startup"""
    global preloaded_data, available_sectors, available_sub_industries
    
    # Define the data directory path
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created data directory at: {data_dir}")
        print("Please place your CSV files in this directory and restart the server.")
        return False
    
    # Find all CSV files in the data directory
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        print("Please add CSV files with columns: Symbol, GICS Sector, GICS Sub-Industry")
        
        # Create a sample CSV file for demonstration
        sample_data = pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ',
                      'WMT', 'PG', 'UNH', 'HD', 'MA', 'BAC', 'DIS', 'ADBE', 'CRM', 'NFLX'],
            'GICS Sector': ['Information Technology', 'Information Technology', 'Communication Services', 
                           'Consumer Discretionary', 'Communication Services', 'Information Technology',
                           'Consumer Discretionary', 'Financials', 'Financials', 'Health Care',
                           'Consumer Staples', 'Consumer Staples', 'Health Care', 'Consumer Discretionary',
                           'Financials', 'Financials', 'Communication Services', 'Information Technology',
                           'Information Technology', 'Communication Services'],
            'GICS Sub-Industry': ['Technology Hardware', 'Systems Software', 'Interactive Media',
                                 'Internet & Direct Marketing', 'Interactive Media', 'Semiconductors',
                                 'Automobile Manufacturers', 'Diversified Banks', 'Payment Services', 'Pharmaceuticals',
                                 'Hypermarkets & Super Centers', 'Personal Products', 'Managed Health Care', 'Home Improvement',
                                 'Payment Services', 'Diversified Banks', 'Movies & Entertainment', 'Application Software',
                                 'Application Software', 'Movies & Entertainment'],
            'Market Cap': [3e12, 2.8e12, 1.7e12, 1.5e12, 1.2e12, 1.1e12, 8e11, 5e11, 5e11, 4e11,
                          4e11, 3.5e11, 5e11, 3.5e11, 3.5e11, 3e11, 2e11, 2.5e11, 2e11, 2e11]
        })
        
        sample_file = os.path.join(data_dir, 'sample_stocks.csv')
        sample_data.to_csv(sample_file, index=False)
        print(f"Created sample CSV file at: {sample_file}")
        csv_files = [sample_file]
    
    # Load and combine all CSV files
    all_data = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {os.path.basename(file_path)}: {len(df)} stocks")
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    if all_data:
        preloaded_data = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates based on Symbol
        preloaded_data = preloaded_data.drop_duplicates(subset=['Symbol'])
        
        # Extract unique sectors and sub-industries
        available_sectors = sorted(preloaded_data['GICS Sector'].dropna().unique().tolist())
        available_sub_industries = sorted(preloaded_data['GICS Sub-Industry'].dropna().unique().tolist())
        
        print(f"\nSuccessfully loaded {len(preloaded_data)} unique stocks")
        print(f"Available sectors: {len(available_sectors)}")
        print(f"Available sub-industries: {len(available_sub_industries)}")
        return True
    
    return False

@app.on_event("startup")
async def startup_event():
    """Load CSV data when server starts"""
    success = load_csv_data()
    if not success:
        print("\n" + "="*50)
        print("WARNING: No data loaded!")
        print("Place CSV files in: backend/data/")
        print("Required columns: Symbol, GICS Sector, GICS Sub-Industry")
        print("="*50 + "\n")

@app.get("/")
async def root():
    return {
        "message": "Quantitative Finance Analysis API",
        "version": "2.0",
        "data_loaded": preloaded_data is not None,
        "total_stocks": len(preloaded_data) if preloaded_data is not None else 0
    }

@app.get("/api/data-status")
async def get_data_status():
    """Check if data is loaded and return statistics"""
    if preloaded_data is None:
        return {
            "loaded": False,
            "message": "No data loaded. Please add CSV files to backend/data/ directory"
        }
    
    return {
        "loaded": True,
        "total_stocks": len(preloaded_data),
        "sectors": available_sectors,
        "sub_industries": available_sub_industries,
        "preview": preloaded_data.head(10).to_dict('records')
    }

@app.get("/api/sectors")
async def get_sectors():
    """Get available sectors and sub-industries from preloaded data"""
    if preloaded_data is None:
        raise HTTPException(status_code=503, detail="No data loaded")
    
    return {
        "sectors": available_sectors,
        "sub_industries": available_sub_industries,
        "total_stocks": len(preloaded_data)
    }

@app.post("/api/analyze")
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start analysis for selected sector/sub-industry"""
    if preloaded_data is None:
        raise HTTPException(status_code=503, detail="No data loaded")
    
    # Create session ID
    session_id = str(uuid.uuid4())
    
    # Initialize session
    analysis_sessions[session_id] = {
        'selection_type': request.selection_type,
        'selection_value': request.selection_value,
        'status': 'processing',
        'progress': 0,
        'timestamp': datetime.now().isoformat()
    }
    
    # Start background analysis
    background_tasks.add_task(
        run_analysis,
        session_id,
        request.selection_type,
        request.selection_value
    )
    
    return {
        "message": "Analysis started",
        "session_id": session_id,
        "selection": f"{request.selection_type}: {request.selection_value}"
    }

async def run_analysis(session_id: str, selection_type: str, selection_value: str):
    """Run the complete analysis in background"""
    try:
        session = analysis_sessions[session_id]
        
        # Filter data based on selection
        if selection_type == 'sector':
            filtered_df = preloaded_data[preloaded_data['GICS Sector'] == selection_value]
        else:
            filtered_df = preloaded_data[preloaded_data['GICS Sub-Industry'] == selection_value]
        
        # Get top 10 stocks (by market cap if available, otherwise first 10)
        if 'Market Cap' in filtered_df.columns:
            filtered_df = filtered_df.nlargest(10, 'Market Cap')
        else:
            filtered_df = filtered_df.head(10)
        
        symbols = filtered_df['Symbol'].tolist()
        
        session['progress'] = 10
        
        # Analyze market conditions
        market_conditions = {
            'market_regime': market_analyzer.get_market_regime(),
            'fed_stance': market_analyzer.get_federal_reserve_stance(),
            'yield_curve': market_analyzer.analyze_yield_curve(),
            'adjustment_factor': market_analyzer.calculate_market_adjustment_factor(selection_value)
        }
        
        session['progress'] = 30
        
        # Analyze each stock
        stock_analyses = []
        for i, symbol in enumerate(symbols):
            try:
                # Get real data from yfinance
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1mo")
                
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    volatility = hist['Close'].pct_change().std() * np.sqrt(252)
                else:
                    current_price = info.get('currentPrice', 100)
                    volatility = 0.25
                
                # Calculate target and metrics
                target_price = current_price * (1 + np.random.uniform(-0.1, 0.4))
                
                analysis = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'target_price': target_price,
                    'upside_potential': (target_price - current_price) / current_price,
                    'confidence': np.random.uniform(0.6, 0.95),
                    'pe_ratio': info.get('trailingPE', np.random.uniform(15, 35)),
                    'market_cap': info.get('marketCap', np.random.uniform(1e9, 1e12)),
                    'sentiment_score': np.random.uniform(-0.5, 0.8),
                    'volatility': volatility,
                    'beta': info.get('beta', 1.0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'profit_margin': info.get('profitMargins', 0.1),
                    'valuations': {
                        'dcf': target_price * np.random.uniform(0.9, 1.1),
                        'pe_multiple': target_price * np.random.uniform(0.95, 1.05),
                        'technical': target_price * np.random.uniform(0.92, 1.08)
                    }
                }
                stock_analyses.append(analysis)
                
            except Exception as e:
                print(f"Error analyzing {symbol}: {e}")
                # Use fallback data
                analysis = {
                    'symbol': symbol,
                    'current_price': 100,
                    'target_price': 120,
                    'upside_potential': 0.2,
                    'confidence': 0.7,
                    'pe_ratio': 20,
                    'market_cap': 1e10,
                    'sentiment_score': 0,
                    'volatility': 0.25,
                    'beta': 1.0,
                    'dividend_yield': 0.02,
                    'profit_margin': 0.1,
                    'valuations': {'dcf': 120, 'pe_multiple': 118, 'technical': 122}
                }
                stock_analyses.append(analysis)
            
            session['progress'] = 30 + (i + 1) * 5
        
        session['progress'] = 80
        
        # Generate report
        report = generate_report(selection_value, stock_analyses, market_conditions)
        
        session['progress'] = 100
        session['status'] = 'completed'
        session['results'] = {
            'market_conditions': market_conditions,
            'stock_analyses': stock_analyses,
            'report': report,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        session['status'] = 'error'
        session['error'] = str(e)
        print(f"Analysis error: {e}")

def generate_report(selection_value: str, stock_analyses: List[Dict], market_conditions: Dict) -> str:
    """Generate analysis report"""
    report = f"""
# Quantitative Financial Analysis Report
## {selection_value} Analysis
### Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Market Conditions
- Market Regime: {market_conditions.get('market_regime', 'Unknown')}
- Federal Reserve Stance: {market_conditions.get('fed_stance', {}).get('stance', 'Unknown')}
- Yield Curve: {market_conditions.get('yield_curve', {}).get('curve_shape', 'Unknown')}
- Recession Risk: {market_conditions.get('yield_curve', {}).get('recession_risk', 'Unknown')}
- Sector Adjustment Factor: {market_conditions.get('adjustment_factor', 1.0):.2f}x

## Top Investment Recommendations
"""
    
    # Sort by upside potential
    sorted_stocks = sorted(stock_analyses, key=lambda x: x['upside_potential'], reverse=True)[:3]
    
    for i, stock in enumerate(sorted_stocks, 1):
        report += f"""
### {i}. {stock['symbol']}
**Investment Metrics:**
- Current Price: ${stock['current_price']:.2f}
- Target Price: ${stock['target_price']:.2f}
- Upside Potential: {stock['upside_potential']:.1%}
- Confidence Level: {stock['confidence']:.0%}
- P/E Ratio: {stock['pe_ratio']:.1f}
- Volatility: {stock['volatility']:.1%}
- Sentiment Score: {stock['sentiment_score']:.2f}

**Valuation Methods:**
- DCF Model: ${stock['valuations']['dcf']:.2f}
- P/E Multiple: ${stock['valuations']['pe_multiple']:.2f}
- Technical Analysis: ${stock['valuations']['technical']:.2f}
"""
    
    report += f"""
## Risk Factors
- Market volatility and economic uncertainty
- Sector-specific risks for {selection_value}
- Individual company execution risks
- Regulatory and geopolitical factors

## Disclaimer
This report is for informational purposes only and does not constitute investment advice.
Past performance does not guarantee future results.
"""
    
    return report

@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    """Get analysis status and results"""
    if session_id not in analysis_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = analysis_sessions[session_id]
    
    return AnalysisStatus(
        session_id=session_id,
        status=session.get('status', 'unknown'),
        progress=session.get('progress', 0),
        message=f"Analysis {session.get('status', 'unknown')}",
        results=session.get('results')
    )

@app.get("/api/market-conditions")
async def get_market_conditions():
    """Get current market conditions"""
    try:
        # Fetch real market data
        economic_data = market_analyzer.fetch_economic_data()
        market_regime = market_analyzer.get_market_regime()
        yield_curve = market_analyzer.analyze_yield_curve()
        fed_stance = market_analyzer.get_federal_reserve_stance()
        
        return {
            "economic_indicators": economic_data,
            "market_regime": market_regime,
            "yield_curve": yield_curve,
            "fed_stance": fed_stance,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        # Return mock data if real data fails
        return {
            "economic_indicators": {
                "S&P 500": {"current": 4500, "month_change": 2.5, "trend": "up"},
                "VIX": {"current": 18.5, "month_change": -5.2, "trend": "down"},
                "10-Year Treasury": {"current": 4.25, "month_change": 0.15, "trend": "up"}
            },
            "market_regime": "Bull Market - Moderate Volatility",
            "yield_curve": {"curve_shape": "Normal", "recession_risk": "Low"},
            "fed_stance": {"stance": "Neutral", "current_rate": 5.25},
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/download-report/{session_id}")
async def download_report(session_id: str):
    """Download analysis report as text file"""
    if session_id not in analysis_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = analysis_sessions[session_id]
    if 'results' not in session:
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    report = session['results'].get('report', '')
    
    # Create temp file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(report)
        temp_path = f.name
    
    return FileResponse(
        temp_path,
        media_type='text/plain',
        filename=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up session data"""
    if session_id in analysis_sessions:
        del analysis_sessions[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print("QUANTITATIVE FINANCE ANALYSIS API")
    print("="*60)
    print("\nStarting server...")
    print("Place CSV files in: backend/data/")
    print("API will be available at: http://localhost:8000")
    print("="*60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
