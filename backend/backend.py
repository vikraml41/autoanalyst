#!/usr/bin/env python3
"""
FastAPI Backend - Docker Version with Full ML Support
Fixed: NaN/Inf handling, proper initialization, error handling
"""

import os
import sys
import glob
import logging
import json
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

print("=== DEBUG: Current directory:", os.getcwd())
print("=== DEBUG: Files in current directory:", os.listdir('.'))
if os.path.exists('data'):
    print("=== DEBUG: Files in data/:", os.listdir('data'))
else:
    print("=== DEBUG: No data/ directory found")



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API - Docker Edition", version="2.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ DATA LOADING FUNCTIONS ============

def load_csv_files():
    """Load CSV files and clean NaN values"""
    data_dirs = ["/app/data", "./data", "data", "../data"]
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            logger.info(f"Found data directory: {data_dir}")
            csv_files = glob.glob(f"{data_dir}/*.csv")
            
            if csv_files:
                dfs = []
                for file in csv_files:
                    try:
                        df = pd.read_csv(file)
                        # Clean the data - replace inf with NaN, then fill NaN
                        df = df.replace([np.inf, -np.inf], np.nan)
                        # For string columns, fill NaN with empty string
                        for col in df.select_dtypes(include=['object']).columns:
                            df[col] = df[col].fillna('')
                        # For numeric columns, fill NaN with 0
                        for col in df.select_dtypes(include=['float64', 'int64']).columns:
                            df[col] = df[col].fillna(0)
                        dfs.append(df)
                        logger.info(f"Loaded and cleaned {len(df)} stocks from {file}")
                    except Exception as e:
                        logger.error(f"Error loading {file}: {e}")
                
                if dfs:
                    combined_df = pd.concat(dfs, ignore_index=True)
                    logger.info(f"✅ Total stocks loaded: {len(combined_df)}")
                    return combined_df
            break
    
    # Sample data if no CSV files found
    logger.warning("No CSV files found, creating sample data")
    sample_data = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
                  'BRK.B', 'V', 'JNJ', 'WMT', 'JPM', 'PG', 'MA', 'UNH'],
        'GICS Sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary',
                       'Technology', 'Technology', 'Consumer Discretionary',
                       'Financials', 'Financials', 'Healthcare', 'Consumer Staples',
                       'Financials', 'Consumer Staples', 'Financials', 'Healthcare'],
        'GICS Sub-Industry': ['Hardware', 'Software', 'Internet', 'E-Commerce',
                             'Semiconductors', 'Social Media', 'Automobiles',
                             'Insurance', 'Payments', 'Pharma', 'Retail',
                             'Banks', 'Consumer', 'Payments', 'Healthcare']
    })
    return sample_data

# ============ INITIALIZE GLOBAL VARIABLES ============

logger.info("=" * 50)
logger.info("Initializing AutoAnalyst Backend...")

# Load CSV data immediately
stocks_data = load_csv_files()
logger.info(f"Stocks data initialized: {len(stocks_data)} stocks")

# Try to import and initialize ML models
ML_AVAILABLE = False
ml_model = None
market_analyzer = None
valuator = None

try:
    logger.info("Attempting to import ML models...")
    from quant_model import QuantFinanceMLModel, MarketConditionsAnalyzer, EnhancedValuation
    
    # Create instances
    ml_model = QuantFinanceMLModel()
    market_analyzer = MarketConditionsAnalyzer()
    valuator = EnhancedValuation()
    
    ML_AVAILABLE = True
    logger.info("✅ ML Models initialized successfully!")
    
except ImportError as e:
    logger.error(f"❌ Could not import ML models: {e}")
    logger.info("Creating fallback models...")
    
    # Fallback classes
    class FallbackMLModel:
        def calculate_ml_score(self, symbol):
            return 0.75
        
        def get_sentiment_score(self, symbol):
            return 0.7
        
        def analyze_stock(self, symbol):
            return {'score': 0.75, 'recommendation': 'hold'}
        
        def get_current_price(self, symbol):
            try:
                ticker = yf.Ticker(symbol)
                return ticker.info.get('currentPrice', 100)
            except:
                return 100
    
    class FallbackMarketAnalyzer:
        def get_market_regime(self):
            try:
                spy = yf.Ticker("SPY")
                hist = spy.history(period="1mo")
                if len(hist) > 0:
                    change = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]
                    if change > 0.02:
                        return "Bull Market - Low Volatility"
                    elif change < -0.02:
                        return "Bear Market - High Volatility"
                return "Neutral Market"
            except:
                return "Neutral Market"
        
        def get_federal_reserve_stance(self):
            return {"stance": "Neutral", "current_rate": 5.25}
        
        def analyze_yield_curve(self):
            return {"curve_shape": "Normal", "recession_risk": "Low"}
        
        def fetch_economic_data(self):
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="1d")
                current_vix = float(vix_hist['Close'].iloc[-1]) if len(vix_hist) > 0 else 20
            except:
                current_vix = 20
            return {"Volatility Index": {"current": current_vix, "trend": "stable"}}
        
        def analyze_sector_conditions(self, sector):
            return {"momentum": "neutral", "adjustment_factor": 1.0}
    
    class FallbackValuator:
        def calculate_intrinsic_value(self, symbol):
            try:
                ticker = yf.Ticker(symbol)
                current = ticker.info.get('currentPrice', 100)
                return current * 1.15
            except:
                return 115
    
    # Use fallback models
    ml_model = FallbackMLModel()
    market_analyzer = FallbackMarketAnalyzer()
    valuator = FallbackValuator()
    ML_AVAILABLE = False
    logger.info("✅ Fallback models created")

# Verify initialization
logger.info("=" * 50)
logger.info("Initialization Summary:")
logger.info(f"  - Stocks Data: {'✅ Loaded' if stocks_data is not None else '❌ Failed'}")
logger.info(f"  - ML Model: {'✅ Loaded' if ml_model is not None else '❌ Failed'}")
logger.info(f"  - Market Analyzer: {'✅ Loaded' if market_analyzer is not None else '❌ Failed'}")
logger.info(f"  - Valuator: {'✅ Loaded' if valuator is not None else '❌ Failed'}")
logger.info(f"  - ML Available: {ML_AVAILABLE}")
logger.info("=" * 50)

# ============ API ENDPOINTS ============

@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info("FastAPI startup event triggered")
    logger.info(f"Current stocks in memory: {len(stocks_data) if stocks_data is not None else 0}")

@app.get("/")
async def root():
    """Root endpoint with system status"""
    return {
        "name": "AutoAnalyst API",
        "version": "2.0.0 Docker",
        "status": "running",
        "ml_enabled": ML_AVAILABLE,
        "data_status": {
            "stocks_loaded": len(stocks_data) if stocks_data is not None else 0,
            "stocks_data_exists": stocks_data is not None,
            "ml_model_exists": ml_model is not None,
            "market_analyzer_exists": market_analyzer is not None,
            "valuator_exists": valuator is not None
        },
        "environment": "Docker on Render"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    components_status = {
        "stocks_data": "✅" if stocks_data is not None and len(stocks_data) > 0 else "❌",
        "ml_model": "✅" if ml_model is not None else "❌",
        "market_analyzer": "✅" if market_analyzer is not None else "❌",
        "valuator": "✅" if valuator is not None else "❌"
    }
    
    return {
        "status": "healthy",
        "components": components_status,
        "ml_available": ML_AVAILABLE,
        "stocks_count": len(stocks_data) if stocks_data is not None else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/stocks/list")
async def get_stocks_list():
    """Get list of available stocks - fixed for NaN/Inf values"""
    try:
        if stocks_data is None:
            return {
                "sectors": [],
                "sub_industries": [],
                "total_stocks": 0,
                "error": "No data loaded"
            }
        
        # Clean the data - remove NaN/Inf
        sectors = []
        sub_industries = []
        
        if 'GICS Sector' in stocks_data.columns:
            sectors = stocks_data['GICS Sector'].dropna().unique().tolist()
            sectors = [s for s in sectors if isinstance(s, str) and s != '']
        
        if 'GICS Sub-Industry' in stocks_data.columns:
            sub_industries = stocks_data['GICS Sub-Industry'].dropna().unique().tolist()
            sub_industries = [s for s in sub_industries if isinstance(s, str) and s != '']
        
        total_stocks = len(stocks_data.dropna(how='all'))
        
        return {
            "sectors": sectors,
            "sub_industries": sub_industries,
            "total_stocks": total_stocks
        }
        
    except Exception as e:
        logger.error(f"Error in get_stocks_list: {str(e)}")
        return {
            "sectors": ["Technology", "Healthcare", "Financials"],
            "sub_industries": ["Software", "Hardware", "Banks"],
            "total_stocks": 15,
            "error": f"Data processing error: {str(e)}"
        }

@app.get("/api/market-conditions")
async def get_market_conditions():
    """Get current market conditions"""
    if market_analyzer is None:
        return {
            "regime": "Neutral Market",
            "fed_stance": "Neutral",
            "vix": 20.0,
            "recession_risk": "Low",
            "ml_powered": False
        }
    
    try:
        regime = market_analyzer.get_market_regime()
        fed_data = market_analyzer.get_federal_reserve_stance()
        
        if ML_AVAILABLE:
            yield_curve = market_analyzer.analyze_yield_curve()
            economic_data = market_analyzer.fetch_economic_data()
            
            return {
                "regime": regime,
                "fed_stance": fed_data.get("stance", "Neutral"),
                "vix": economic_data.get("Volatility Index", {}).get("current", 20),
                "recession_risk": yield_curve.get("recession_risk", "Medium"),
                "ml_powered": True
            }
        else:
            economic_data = market_analyzer.fetch_economic_data()
            return {
                "regime": regime,
                "fed_stance": fed_data.get("stance", "Neutral"),
                "vix": economic_data.get("Volatility Index", {}).get("current", 20),
                "recession_risk": "Low",
                "ml_powered": False
            }
            
    except Exception as e:
        logger.error(f"Market conditions error: {e}")
        return {
            "regime": "Neutral Market",
            "fed_stance": "Neutral",
            "vix": 20.0,
            "recession_risk": "Low",
            "error": str(e)
        }

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest):
    """Run full ML analysis"""
    logger.info(f"Analysis requested: {request.analysis_type} - {request.target}")
    
    if stocks_data is None:
        raise HTTPException(status_code=500, detail="Stock data not initialized")
    
    if ml_model is None:
        raise HTTPException(status_code=500, detail="ML model not initialized")
    
    try:
        # Filter stocks
        if request.analysis_type == "sector":
            filtered = stocks_data[stocks_data['GICS Sector'] == request.target]
        else:
            filtered = stocks_data[stocks_data['GICS Sub-Industry'] == request.target]
        
        if len(filtered) == 0:
            raise HTTPException(status_code=404, detail=f"No stocks found for {request.target}")
        
        logger.info(f"Found {len(filtered)} stocks for analysis")
        
        results = []
        for _, stock in filtered.iterrows():
            symbol = stock['Symbol']
            
            ml_score = ml_model.calculate_ml_score(symbol) if hasattr(ml_model, 'calculate_ml_score') else 0.75
            sentiment = ml_model.get_sentiment_score(symbol) if hasattr(ml_model, 'get_sentiment_score') else 0.7
            
            if valuator and hasattr(valuator, 'calculate_intrinsic_value'):
                target_price = valuator.calculate_intrinsic_value(symbol)
            else:
                current = ml_model.get_current_price(symbol) if hasattr(ml_model, 'get_current_price') else 100
                target_price = current * 1.15
            
            results.append({
                "symbol": symbol,
                "ml_score": ml_score,
                "sentiment": sentiment,
                "target_price": target_price
            })
        
        # Sort and get top 3
        results.sort(key=lambda x: x['ml_score'], reverse=True)
        top_3 = results[:3]
        
        # Format response
        top_stocks = []
        for stock in top_3:
            current_price = ml_model.get_current_price(stock['symbol']) if hasattr(ml_model, 'get_current_price') else 100
            
            top_stocks.append({
                "symbol": stock['symbol'],
                "metrics": {
                    "current_price": current_price,
                    "target_price": stock['target_price'],
                    "upside_potential": ((stock['target_price'] / current_price) - 1) * 100,
                    "confidence_score": int(stock['ml_score'] * 100),
                    "sentiment_score": stock['sentiment'],
                    "ml_score": stock['ml_score']
                }
            })
        
        return {
            "status": "completed",
            "analysis_type": request.analysis_type,
            "target": request.target,
            "results": {
                "top_stocks": top_stocks,
                "market_conditions": {
                    "regime": market_analyzer.get_market_regime() if market_analyzer else "Neutral",
                    "adjustment_factor": 1.05
                }
            },
            "ml_powered": ML_AVAILABLE
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload new CSV file"""
    global stocks_data
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Clean the uploaded data
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].fillna('')
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = df[col].fillna(0)
        
        # Save to data directory
        os.makedirs("/app/data", exist_ok=True)
        file_path = f"/app/data/{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Reload data
        stocks_data = load_csv_files()
        
        return {
            "status": "success",
            "filename": file.filename,
            "rows": len(df),
            "total_stocks": len(stocks_data)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Final check - stocks_data: {stocks_data is not None}")
    logger.info(f"Final check - ml_model: {ml_model is not None}")
    uvicorn.run(app, host="0.0.0.0", port=port)