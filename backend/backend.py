#!/usr/bin/env python3
"""
FastAPI Backend - Using the ACTUAL quant_model.py with diagnostics
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
import warnings
warnings.filterwarnings('ignore')

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
    """Analysis with MAXIMUM error protection"""
    logger.info(f"=" * 50)
    logger.info(f"Analysis START: {request.analysis_type} - {request.target}")
    
    # Build detailed response even if things fail
    response_data = {
        "status": "starting",
        "analysis_type": request.analysis_type,
        "target": request.target,
        "debug_info": {},
        "results": {
            "top_stocks": [],
            "market_conditions": {"regime": "Unknown", "adjustment_factor": 1.0}
        }
    }
    
    try:
        # Check 1: Data exists
        if stocks_data is None:
            response_data["debug_info"]["error"] = "stocks_data is None"
            response_data["status"] = "failed"
            return JSONResponse(content=response_data, status_code=200)
        
        if len(stocks_data) == 0:
            response_data["debug_info"]["error"] = "stocks_data is empty"
            response_data["status"] = "failed"
            return JSONResponse(content=response_data, status_code=200)
        
        response_data["debug_info"]["total_stocks"] = len(stocks_data)
        response_data["debug_info"]["columns"] = list(stocks_data.columns)
        
        # Check 2: Filter stocks
        try:
            if request.analysis_type == "sector":
                filtered = stocks_data[stocks_data['GICS Sector'] == request.target]
            else:
                filtered = stocks_data[stocks_data['GICS Sub-Industry'] == request.target]
            
            response_data["debug_info"]["filtered_count"] = len(filtered)
            
            if len(filtered) == 0:
                response_data["debug_info"]["error"] = f"No stocks found for {request.target}"
                response_data["status"] = "no_data"
                return JSONResponse(content=response_data, status_code=200)
                
        except Exception as e:
            response_data["debug_info"]["filter_error"] = str(e)
            response_data["status"] = "filter_failed"
            return JSONResponse(content=response_data, status_code=200)
        
        # Check 3: Model exists
        response_data["debug_info"]["ml_model_type"] = type(ml_model).__name__ if ml_model else "None"
        response_data["debug_info"]["ml_available"] = ML_AVAILABLE
        
        if ml_model is None:
            response_data["debug_info"]["error"] = "ML model is None"
            response_data["status"] = "no_model"
            return JSONResponse(content=response_data, status_code=200)
        
        # Check 4: Try to analyze stocks
        results = []
        analyze_errors = []
        
        # Only analyze first 10 stocks to avoid timeout
        stocks_to_analyze = filtered.head(10)
        
        for idx, (_, stock) in enumerate(stocks_to_analyze.iterrows()):
            symbol = stock.get('Symbol', '')
            
            if not symbol or pd.isna(symbol):
                continue
            
            logger.info(f"Analyzing {symbol}...")
            
            # Create a result with defaults
            result = {
                "symbol": symbol,
                "ml_score": 0.5,
                "sentiment": 0.5,
                "current_price": 100.0,
                "target_price": 115.0
            }
            
            # Try to get real ML score
            try:
                if hasattr(ml_model, 'calculate_ml_score'):
                    score = ml_model.calculate_ml_score(symbol)
                    if score is not None:
                        result["ml_score"] = float(score)
            except Exception as e:
                analyze_errors.append(f"{symbol}_ml: {str(e)[:50]}")
            
            # Try to get sentiment
            try:
                if hasattr(ml_model, 'get_sentiment_score'):
                    sent = ml_model.get_sentiment_score(symbol)
                    if sent is not None:
                        result["sentiment"] = float(sent)
            except Exception as e:
                analyze_errors.append(f"{symbol}_sent: {str(e)[:50]}")
            
            # Try to get price
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('price')
                if price and price > 0:
                    result["current_price"] = float(price)
                    result["target_price"] = float(price * 1.15)
            except Exception as e:
                analyze_errors.append(f"{symbol}_price: {str(e)[:50]}")
            
            results.append(result)
            
            # Stop if we have 3 good results
            if len(results) >= 3:
                break
        
        response_data["debug_info"]["analyze_errors"] = analyze_errors[:5]  # First 5 errors
        response_data["debug_info"]["results_count"] = len(results)
        
        # Sort and get top 3
        if results:
            results.sort(key=lambda x: x['ml_score'], reverse=True)
            top_3 = results[:3]
            
            # Format for frontend
            for stock in top_3:
                upside = ((stock['target_price'] / stock['current_price']) - 1) * 100
                
                response_data["results"]["top_stocks"].append({
                    "symbol": stock['symbol'],
                    "metrics": {
                        "current_price": round(stock['current_price'], 2),
                        "target_price": round(stock['target_price'], 2),
                        "upside_potential": round(upside, 1),
                        "confidence_score": int(stock['ml_score'] * 100),
                        "sentiment_score": round(stock['sentiment'], 2),
                        "ml_score": round(stock['ml_score'], 3)
                    }
                })
            
            response_data["status"] = "completed"
        else:
            response_data["status"] = "no_results"
            response_data["debug_info"]["error"] = "Could not analyze any stocks"
        
        return JSONResponse(content=response_data, status_code=200, headers={"Access-Control-Allow-Origin": "*"})
        
    except Exception as e:
        # Catch EVERYTHING
        import traceback
        response_data["debug_info"]["exception"] = str(e)
        response_data["debug_info"]["traceback"] = traceback.format_exc()[:500]
        response_data["status"] = "exception"
        logger.error(f"Analysis exception: {e}")
        logger.error(traceback.format_exc())
        
        return JSONResponse(content=response_data, status_code=200, headers={"Access-Control-Allow-Origin": "*"})

@app.get("/api/test-analysis/{symbol}")
async def test_single_stock(symbol: str):
    """Test analysis on a single stock"""
    results = {}
    
    try:
        results['ml_score'] = ml_model.calculate_ml_score(symbol)
    except Exception as e:
        results['ml_score_error'] = str(e)
    
    try:
        results['sentiment'] = ml_model.get_sentiment_score(symbol)
    except Exception as e:
        results['sentiment_error'] = str(e)
    
    try:
        if hasattr(ml_model, 'get_current_price'):
            results['current_price'] = ml_model.get_current_price(symbol)
        else:
            ticker = yf.Ticker(symbol)
            results['current_price'] = ticker.info.get('currentPrice')
    except Exception as e:
        results['price_error'] = str(e)
    
    try:
        if valuator:
            results['target_price'] = valuator.calculate_intrinsic_value(symbol)
    except Exception as e:
        results['target_error'] = str(e)
    
    results['model_type'] = type(ml_model).__name__
    results['ml_available'] = ML_AVAILABLE
    
    return results

@app.get("/api/diagnose")
async def diagnose_system():
    """Complete system diagnostic"""
    import inspect
    
    diagnosis = {
        "ml_model": {
            "exists": ml_model is not None,
            "type": type(ml_model).__name__ if ml_model else "None",
            "methods": []
        },
        "test_results": {},
        "imports": {},
        "errors": []
    }
    
    # Check what methods the ML model has
    if ml_model:
        diagnosis["ml_model"]["methods"] = [m for m in dir(ml_model) if not m.startswith('_')]
    
    # Test if we can import quant_model
    try:
        import quant_model
        diagnosis["imports"]["quant_model"] = "Success"
        diagnosis["imports"]["classes"] = [name for name in dir(quant_model) if not name.startswith('_')]
    except Exception as e:
        diagnosis["imports"]["quant_model"] = f"Failed: {str(e)}"
    
    # Test ML model methods with a real stock
    test_symbol = "AAPL"
    
    # Test calculate_ml_score
    try:
        if ml_model and hasattr(ml_model, 'calculate_ml_score'):
            result = ml_model.calculate_ml_score(test_symbol)
            diagnosis["test_results"]["calculate_ml_score"] = {
                "success": True,
                "result": result
            }
    except Exception as e:
        diagnosis["test_results"]["calculate_ml_score"] = {
            "success": False,
            "error": str(e)[:200]
        }
    
    # Test get_sentiment_score
    try:
        if ml_model and hasattr(ml_model, 'get_sentiment_score'):
            result = ml_model.get_sentiment_score(test_symbol)
            diagnosis["test_results"]["get_sentiment_score"] = {
                "success": True,
                "result": result
            }
    except Exception as e:
        diagnosis["test_results"]["get_sentiment_score"] = {
            "success": False,
            "error": str(e)[:200]
        }
    
    # Test get_current_price
    try:
        if ml_model and hasattr(ml_model, 'get_current_price'):
            result = ml_model.get_current_price(test_symbol)
            diagnosis["test_results"]["get_current_price"] = {
                "success": True,
                "result": result
            }
    except Exception as e:
        diagnosis["test_results"]["get_current_price"] = {
            "success": False,
            "error": str(e)[:200]
        }
    
    # Test basic yfinance
    try:
        import yfinance as yf
        ticker = yf.Ticker(test_symbol)
        info = ticker.info
        price = info.get('currentPrice') or info.get('regularMarketPrice')
        diagnosis["test_results"]["yfinance"] = {
            "success": True,
            "price": price,
            "info_keys": list(info.keys())[:10]  # First 10 keys
        }
    except Exception as e:
        diagnosis["test_results"]["yfinance"] = {
            "success": False,
            "error": str(e)[:200]
        }
    
    return diagnosis

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Using ML Model: {type(ml_model).__name__}")
    uvicorn.run(app, host="0.0.0.0", port=port)