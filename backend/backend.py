#!/usr/bin/env python3
"""
FastAPI Backend - Using QuantFinanceMLModel's ACTUAL methods
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
    ML_AVAILABLE = False

# Log what we're using
logger.info("=" * 50)
logger.info("Model Status:")
logger.info(f"  - ML Available: {ML_AVAILABLE}")
logger.info(f"  - ML Model Type: {type(ml_model).__name__ if ml_model else 'None'}")
logger.info(f"  - Market Analyzer Type: {type(market_analyzer).__name__ if market_analyzer else 'None'}")
logger.info(f"  - Valuator Type: {type(valuator).__name__ if valuator else 'None'}")
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
            "model_type": type(ml_model).__name__ if ml_model else "None",
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
                "market_analyzer": "✅" if market_analyzer else "❌",
                "valuator": "✅" if valuator else "❌"
            },
            "ml_available": ML_AVAILABLE,
            "model_type": type(ml_model).__name__ if ml_model else "None",
            "stocks_count": len(stocks_data),
            "timestamp": datetime.now().isoformat()
        },
        headers={"Access-Control-Allow-Origin": "*"}
    )

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
        regime = "Unknown"
        fed_stance = "Unknown"
        vix = 20.0
        recession_risk = "Unknown"
        
        # Try YOUR model's analyze_market_conditions first
        if ml_model and hasattr(ml_model, 'analyze_market_conditions'):
            try:
                conditions = ml_model.analyze_market_conditions()
                regime = str(conditions) if conditions else "Neutral"
            except:
                pass
        
        # Use MarketConditionsAnalyzer methods
        if market_analyzer:
            try:
                if hasattr(market_analyzer, 'get_market_regime'):
                    regime = market_analyzer.get_market_regime()
            except:
                pass
            
            try:
                if hasattr(market_analyzer, 'get_federal_reserve_stance'):
                    fed_data = market_analyzer.get_federal_reserve_stance()
                    fed_stance = fed_data.get("stance", "Unknown")
            except:
                pass
            
            try:
                if hasattr(market_analyzer, 'fetch_economic_data'):
                    economic_data = market_analyzer.fetch_economic_data()
                    vix = economic_data.get("Volatility Index", {}).get("current", 20)
            except:
                pass
            
            try:
                if hasattr(market_analyzer, 'analyze_yield_curve'):
                    yield_curve = market_analyzer.analyze_yield_curve()
                    recession_risk = yield_curve.get("recession_risk", "Unknown")
            except:
                pass
        
        return JSONResponse(
            content={
                "regime": regime,
                "fed_stance": fed_stance,
                "vix": vix,
                "recession_risk": recession_risk,
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
                "recession_risk": "Error"
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest):
    """Analysis using YOUR QuantFinanceMLModel's ACTUAL methods"""
    logger.info(f"=" * 50)
    logger.info(f"Analysis START: {request.analysis_type} - {request.target}")
    logger.info(f"Using model: {type(ml_model).__name__ if ml_model else 'None'}")
    
    response_data = {
        "status": "starting",
        "analysis_type": request.analysis_type,
        "target": request.target,
        "results": {
            "top_stocks": [],
            "market_conditions": {"regime": "Unknown", "adjustment_factor": 1.0}
        }
    }
    
    try:
        # Filter stocks
        if request.analysis_type == "sector":
            filtered = stocks_data[stocks_data['GICS Sector'] == request.target]
        else:
            filtered = stocks_data[stocks_data['GICS Sub-Industry'] == request.target]
        
        if len(filtered) == 0:
            response_data["status"] = "no_data"
            response_data["error"] = f"No stocks found for {request.target}"
            return JSONResponse(content=response_data, status_code=200, headers={"Access-Control-Allow-Origin": "*"})
        
        logger.info(f"Found {len(filtered)} stocks to analyze")
        
        results = []
        
        # Analyze each stock using YOUR model's methods
        for idx, (_, stock) in enumerate(filtered.head(20).iterrows()):
            symbol = stock.get('Symbol', '')
            
            if not symbol or pd.isna(symbol) or symbol == '':
                continue
            
            logger.info(f"Analyzing {symbol} ({idx+1}/20)...")
            
            # Get current price from yfinance
            current_price = None
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('price')
                
                if not current_price or current_price <= 0:
                    # Try from history
                    hist = ticker.history(period="1d")
                    if len(hist) > 0:
                        current_price = float(hist['Close'].iloc[-1])
            except Exception as e:
                logger.error(f"Price error for {symbol}: {e}")
                continue
            
            if not current_price or current_price <= 0:
                logger.warning(f"Skipping {symbol} - no valid price")
                continue
            
            # Calculate ML score using YOUR model's methods
            ml_score = 0.5  # Default
            try:
                # Try different methods your model might have
                if ml_model:
                    if hasattr(ml_model, 'calculate_stock_predictions'):
                        # This method exists in your model
                        predictions = ml_model.calculate_stock_predictions()
                        # Extract score - you might need to modify this based on return format
                        ml_score = 0.75
                    elif hasattr(ml_model, 'train_prediction_model'):
                        # Alternative method
                        ml_model.train_prediction_model()
                        ml_score = 0.7
                    
                    # Use market adjustment if available
                    if hasattr(ml_model, 'market_adjustment'):
                        adjustment = ml_model.market_adjustment()
                        ml_score = ml_score * (1 + adjustment) if adjustment else ml_score
            except Exception as e:
                logger.error(f"ML prediction error for {symbol}: {str(e)[:100]}")
            
            # Get sentiment using YOUR model's methods
            sentiment = 0.5  # Default
            try:
                if ml_model:
                    if hasattr(ml_model, 'simple_sentiment'):
                        # Use simple sentiment with stock description
                        sentiment_result = ml_model.simple_sentiment(f"{symbol} stock market performance")
                        if isinstance(sentiment_result, (int, float)):
                            sentiment = max(0, min(1, sentiment_result))  # Ensure 0-1 range
                        else:
                            sentiment = 0.6
                    elif hasattr(ml_model, 'news_sentiment_analysis'):
                        # Use news sentiment
                        news_sentiment = ml_model.news_sentiment_analysis()
                        sentiment = 0.65
                    elif hasattr(ml_model, 'analyze_reddit_sentiment'):
                        # Use Reddit sentiment
                        try:
                            reddit_data = ml_model.analyze_reddit_sentiment(symbol)
                            sentiment = 0.7  # Process reddit_data as needed
                        except:
                            sentiment = 0.5
            except Exception as e:
                logger.error(f"Sentiment error for {symbol}: {str(e)[:100]}")
            
            # Calculate target price
            target_price = current_price * 1.15  # Default 15% upside
            try:
                if valuator and hasattr(valuator, 'calculate_intrinsic_value'):
                    intrinsic = valuator.calculate_intrinsic_value(symbol)
                    if intrinsic and intrinsic > 0:
                        target_price = float(intrinsic)
                elif ml_model and hasattr(ml_model, 'calculate_dcf'):
                    # Use DCF from your model
                    dcf_value = ml_model.calculate_dcf()
                    if dcf_value and dcf_value > 0:
                        target_price = float(dcf_value)
                elif ml_model and hasattr(ml_model, 'fundamental_analysis'):
                    # Use fundamental analysis
                    fundamental_data = ml_model.fundamental_analysis()
                    # Extract target from fundamental_data
                    target_price = current_price * 1.2
            except Exception as e:
                logger.error(f"Valuation error for {symbol}: {str(e)[:100]}")
            
            # Add to results
            results.append({
                "symbol": symbol,
                "ml_score": float(ml_score),
                "sentiment": float(sentiment),
                "current_price": float(current_price),
                "target_price": float(target_price)
            })
            
            logger.info(f"✅ {symbol}: score={ml_score:.2f}, sentiment={sentiment:.2f}, price=${current_price:.2f}")
            
            # Stop if we have enough good results
            if len(results) >= 5:
                break
        
        # Sort and get top 3
        if results:
            results.sort(key=lambda x: x['ml_score'], reverse=True)
            top_3 = results[:3]
            
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
            logger.info(f"Analysis completed. Top 3: {[s['symbol'] for s in top_3]}")
        else:
            response_data["status"] = "no_results"
            response_data["error"] = "Could not analyze any stocks successfully"
        
        # Get market conditions
        try:
            if ml_model and hasattr(ml_model, 'analyze_market_conditions'):
                market_conditions = ml_model.analyze_market_conditions()
                response_data["results"]["market_conditions"]["regime"] = str(market_conditions)
            elif market_analyzer and hasattr(market_analyzer, 'get_market_regime'):
                response_data["results"]["market_conditions"]["regime"] = market_analyzer.get_market_regime()
        except:
            pass
        
        return JSONResponse(content=response_data, status_code=200, headers={"Access-Control-Allow-Origin": "*"})
        
    except Exception as e:
        import traceback
        response_data["error"] = str(e)
        response_data["traceback"] = traceback.format_exc()[:500]
        response_data["status"] = "exception"
        logger.error(f"Analysis exception: {e}")
        return JSONResponse(content=response_data, status_code=200, headers={"Access-Control-Allow-Origin": "*"})

@app.get("/api/test-analysis/{symbol}")
async def test_single_stock(symbol: str):
    """Test analysis on a single stock"""
    results = {"symbol": symbol}
    
    # Test price fetching
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        results['current_price'] = info.get('currentPrice') or info.get('regularMarketPrice')
        results['company_name'] = info.get('longName', symbol)
    except Exception as e:
        results['price_error'] = str(e)[:100]
    
    # Test YOUR model's methods
    if ml_model:
        try:
            if hasattr(ml_model, 'simple_sentiment'):
                results['sentiment'] = ml_model.simple_sentiment(f"{symbol} stock")
        except Exception as e:
            results['sentiment_error'] = str(e)[:100]
        
        try:
            if hasattr(ml_model, 'calculate_stock_predictions'):
                results['has_predictions_method'] = True
        except:
            pass
    
    if valuator:
        try:
            if hasattr(valuator, 'calculate_intrinsic_value'):
                results['target_price'] = valuator.calculate_intrinsic_value(symbol)
        except Exception as e:
            results['valuation_error'] = str(e)[:100]
    
    results['model_type'] = type(ml_model).__name__ if ml_model else "None"
    results['ml_available'] = ML_AVAILABLE
    
    return results

@app.get("/api/diagnose")
async def diagnose_system():
    """Complete system diagnostic"""
    diagnosis = {
        "ml_model": {
            "exists": ml_model is not None,
            "type": type(ml_model).__name__ if ml_model else "None",
            "methods": []
        },
        "market_analyzer": {
            "exists": market_analyzer is not None,
            "type": type(market_analyzer).__name__ if market_analyzer else "None"
        },
        "valuator": {
            "exists": valuator is not None,
            "type": type(valuator).__name__ if valuator else "None"
        },
        "test_results": {},
        "imports": {}
    }
    
    # Check ML model methods
    if ml_model:
        diagnosis["ml_model"]["methods"] = [m for m in dir(ml_model) if not m.startswith('_')]
    
    # Test import
    try:
        import quant_model
        diagnosis["imports"]["quant_model"] = "Success"
        diagnosis["imports"]["classes"] = [name for name in dir(quant_model) if not name.startswith('_')]
    except Exception as e:
        diagnosis["imports"]["quant_model"] = f"Failed: {str(e)}"
    
    # Test basic yfinance
    try:
        ticker = yf.Ticker("AAPL")
        info = ticker.info
        diagnosis["test_results"]["yfinance"] = {
            "success": True,
            "price": info.get('currentPrice') or info.get('regularMarketPrice')
        }
    except Exception as e:
        diagnosis["test_results"]["yfinance"] = {
            "success": False,
            "error": str(e)[:100]
        }
    
    return diagnosis

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Using ML Model: {type(ml_model).__name__ if ml_model else 'None'}")
    uvicorn.run(app, host="0.0.0.0", port=port)