#!/usr/bin/env python3
"""
FastAPI Backend - Using the ACTUAL quant_model.py
"""

import os
import sys
import glob
import logging
import json
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="3.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add CORS headers manually
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ============ DATA LOADING ============

def load_csv_files():
    """Load CSV files from data directory"""
    data_path = os.environ.get('DATA_PATH', '/app/data')
    if not os.path.exists(data_path):
        data_path = 'data'
    
    logger.info(f"Loading CSV files from: {data_path}")
    csv_pattern = os.path.join(data_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        logger.warning("No CSV files found, using sample data")
        return pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
            'GICS Sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 
                           'Technology', 'Technology', 'Consumer Discretionary'],
            'GICS Sub-Industry': ['Hardware', 'Software', 'Internet', 'E-Commerce',
                                 'Semiconductors', 'Social Media', 'Automobiles']
        })
    
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df = df.replace([np.inf, -np.inf], np.nan)
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].fillna('')
            for col in df.select_dtypes(include=['float64', 'int64']).columns:
                df[col] = df[col].fillna(0)
            all_dfs.append(df)
            logger.info(f"Loaded {len(df)} rows from {csv_file}")
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total stocks loaded: {len(combined)}")
        return combined
    return pd.DataFrame()

# ============ IMPORT YOUR ACTUAL QUANT MODEL ============

# Load data first
stocks_data = load_csv_files()
logger.info(f"Stocks data initialized: {len(stocks_data)} stocks")

# Now import your ACTUAL quant_model.py
ML_AVAILABLE = False
ml_model = None
market_analyzer = None
valuator = None

try:
    logger.info("Importing YOUR quant_model.py...")
    
    # Import your actual model classes
    from quant_model import (
        QuantFinanceMLModel, 
        MarketConditionsAnalyzer, 
        EnhancedValuation
    )
    
    # Initialize YOUR models
    logger.info("Initializing QuantFinanceMLModel...")
    ml_model = QuantFinanceMLModel()
    
    logger.info("Initializing MarketConditionsAnalyzer...")
    market_analyzer = MarketConditionsAnalyzer()
    
    logger.info("Initializing EnhancedValuation...")
    valuator = EnhancedValuation()
    
    ML_AVAILABLE = True
    logger.info("✅ YOUR quant_model.py loaded successfully!")
    
except Exception as e:
    logger.error(f"❌ Error loading quant_model.py: {e}")
    logger.error(f"Error type: {type(e).__name__}")
    logger.error(f"Error details: {str(e)}")
    
    # Only use fallback if YOUR model fails to load
    logger.info("Creating minimal fallback...")
    
    class MinimalFallback:
        def calculate_ml_score(self, symbol):
            logger.warning(f"Using fallback for {symbol}")
            return 0.5
        
        def get_sentiment_score(self, symbol):
            return 0.5
        
        def get_current_price(self, symbol):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                return info.get('currentPrice', info.get('regularMarketPrice', 100))
            except:
                return 100
        
        def calculate_intrinsic_value(self, symbol):
            price = self.get_current_price(symbol)
            return price * 1.1
        
        def get_market_regime(self):
            return "Model Not Loaded"
        
        def get_federal_reserve_stance(self):
            return {"stance": "Unknown"}
        
        def fetch_economic_data(self):
            return {"Volatility Index": {"current": 20}}
        
        def analyze_yield_curve(self):
            return {"recession_risk": "Unknown"}
    
    # Create minimal fallback instances
    ml_model = MinimalFallback()
    market_analyzer = MinimalFallback()
    valuator = MinimalFallback()
    ML_AVAILABLE = False

# Log what we're using
logger.info("=" * 50)
logger.info("Model Status:")
logger.info(f"  - ML Available: {ML_AVAILABLE}")
logger.info(f"  - ML Model Type: {type(ml_model).__name__}")
logger.info(f"  - Market Analyzer Type: {type(market_analyzer).__name__}")
logger.info(f"  - Valuator Type: {type(valuator).__name__}")
logger.info("=" * 50)

# ============ API ENDPOINTS ============

@app.get("/")
async def root():
    return JSONResponse(
        content={
            "name": "AutoAnalyst API",
            "version": "3.0.0",
            "status": "running",
            "ml_enabled": ML_AVAILABLE,
            "model_type": type(ml_model).__name__,
            "stocks_loaded": len(stocks_data)
        },
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.get("/api/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "components": {
                "stocks_data": "✅" if len(stocks_data) > 0 else "❌",
                "ml_model": "✅" if ML_AVAILABLE else "❌",
                "market_analyzer": "✅" if ML_AVAILABLE else "❌",
                "valuator": "✅" if ML_AVAILABLE else "❌"
            },
            "ml_available": ML_AVAILABLE,
            "model_type": type(ml_model).__name__,
            "stocks_count": len(stocks_data),
            "timestamp": datetime.now().isoformat()
        },
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.get("/api/debug-ml")
async def debug_ml():
    """Debug endpoint to check ML status"""
    import os
    import importlib.util
    
    # Check if quant_model.py exists
    quant_model_path = '/app/quant_model.py'
    local_path = 'quant_model.py'
    
    return {
        "ml_available": ML_AVAILABLE,
        "model_type": type(ml_model).__name__,
        "quant_model_exists_app": os.path.exists(quant_model_path),
        "quant_model_exists_local": os.path.exists(local_path),
        "files_in_app": os.listdir('/app') if os.path.exists('/app') else [],
        "can_import": importlib.util.find_spec("quant_model") is not None
    }

@app.get("/api/stocks/list")
async def get_stocks_list():
    sectors = []
    sub_industries = []
    
    if 'GICS Sector' in stocks_data.columns:
        sectors = stocks_data['GICS Sector'].dropna().unique().tolist()
        sectors = [s for s in sectors if isinstance(s, str) and s != '']
    
    if 'GICS Sub-Industry' in stocks_data.columns:
        sub_industries = stocks_data['GICS Sub-Industry'].dropna().unique().tolist()
        sub_industries = [s for s in sub_industries if isinstance(s, str) and s != '']
    
    return JSONResponse(
        content={
            "sectors": sectors,
            "sub_industries": sub_industries,
            "total_stocks": len(stocks_data)
        },
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.get("/api/market-conditions")
async def get_market_conditions():
    try:
        # Use YOUR MarketConditionsAnalyzer
        regime = market_analyzer.get_market_regime() if hasattr(market_analyzer, 'get_market_regime') else "Unknown"
        fed_data = market_analyzer.get_federal_reserve_stance() if hasattr(market_analyzer, 'get_federal_reserve_stance') else {"stance": "Unknown"}
        economic_data = market_analyzer.fetch_economic_data() if hasattr(market_analyzer, 'fetch_economic_data') else {"Volatility Index": {"current": 20}}
        yield_curve = market_analyzer.analyze_yield_curve() if hasattr(market_analyzer, 'analyze_yield_curve') else {"recession_risk": "Unknown"}
        
        return JSONResponse(
            content={
                "regime": regime,
                "fed_stance": fed_data.get("stance", "Unknown"),
                "vix": economic_data.get("Volatility Index", {}).get("current", 20),
                "recession_risk": yield_curve.get("recession_risk", "Unknown"),
                "ml_powered": ML_AVAILABLE
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Market conditions error: {e}")
        return JSONResponse(
            content={
                "regime": "Error",
                "fed_stance": "Error",
                "vix": 20.0,
                "recession_risk": "Error",
                "error": str(e)
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest):
    """Run analysis using YOUR QuantFinanceMLModel"""
    logger.info(f"Analysis requested: {request.analysis_type} - {request.target}")
    
    if stocks_data is None or len(stocks_data) == 0:
        raise HTTPException(status_code=500, detail="Stock data not initialized")
    
    if not ML_AVAILABLE:
        logger.warning("ML model not available, using fallback")
    
    try:
        # Filter stocks
        if request.analysis_type == "sector":
            filtered = stocks_data[stocks_data['GICS Sector'] == request.target]
        else:
            filtered = stocks_data[stocks_data['GICS Sub-Industry'] == request.target]
        
        if len(filtered) == 0:
            raise HTTPException(status_code=404, detail=f"No stocks found for {request.target}")
        
        logger.info(f"Analyzing {len(filtered)} stocks using {type(ml_model).__name__}")
        
        # Use YOUR model's methods
        results = []
        for _, stock in filtered.iterrows():
            symbol = stock['Symbol']
            
            if not symbol or pd.isna(symbol) or symbol == '':
                continue
            
            try:
                # Use YOUR QuantFinanceMLModel methods
                ml_score = ml_model.calculate_ml_score(symbol)
                sentiment = ml_model.get_sentiment_score(symbol)
                
                # Get current price
                if hasattr(ml_model, 'get_current_price'):
                    current_price = ml_model.get_current_price(symbol)
                else:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    current_price = info.get('currentPrice', info.get('regularMarketPrice', 100))
                
                # Use YOUR EnhancedValuation
                if valuator and hasattr(valuator, 'calculate_intrinsic_value'):
                    target_price = valuator.calculate_intrinsic_value(symbol)
                else:
                    target_price = current_price * 1.15
                
                # Skip if we couldn't get valid prices
                if not current_price or not target_price:
                    continue
                
                results.append({
                    "symbol": symbol,
                    "ml_score": ml_score,
                    "sentiment": sentiment,
                    "current_price": current_price,
                    "target_price": target_price
                })
                
                logger.info(f"Analyzed {symbol}: ML Score={ml_score:.2f}, Price=${current_price:.2f}")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        if not results:
            raise HTTPException(status_code=500, detail="No valid analysis results")
        
        # Sort by ML score
        results.sort(key=lambda x: x['ml_score'], reverse=True)
        
        # Get top 3
        top_3 = results[:3]
        
        # Format response
        top_stocks = []
        for stock in top_3:
            upside = ((stock['target_price'] / stock['current_price']) - 1) * 100 if stock['current_price'] > 0 else 0
            
            top_stocks.append({
                "symbol": stock['symbol'],
                "metrics": {
                    "current_price": round(stock['current_price'], 2),
                    "target_price": round(stock['target_price'], 2),
                    "upside_potential": round(upside, 1),
                    "confidence_score": int(stock['ml_score'] * 100),
                    "sentiment_score": stock['sentiment'],
                    "ml_score": stock['ml_score']
                }
            })
        
        return JSONResponse(
            content={
                "status": "completed",
                "analysis_type": request.analysis_type,
                "target": request.target,
                "results": {
                    "top_stocks": top_stocks,
                    "market_conditions": {
                        "regime": market_analyzer.get_market_regime() if hasattr(market_analyzer, 'get_market_regime') else "Unknown",
                        "adjustment_factor": 1.0
                    },
                    "total_analyzed": len(results),
                    "model_used": type(ml_model).__name__
                },
                "ml_powered": ML_AVAILABLE
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Using ML Model: {type(ml_model).__name__}")
    uvicorn.run(app, host="0.0.0.0", port=port)