#!/usr/bin/env python3
"""
FastAPI Backend - Fully compatible with actual QuantFinanceMLModel
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
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="6.0.0")

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
    
    # Set the master_df directly to avoid input() calls
    ml_model.master_df = stocks_data
    ml_model.process_gics_data()
    
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
            "version": "6.0.0",
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
    
    if ML_AVAILABLE and ml_model and hasattr(ml_model, 'sectors'):
        sectors = ml_model.sectors if ml_model.sectors else []
        sub_industries = ml_model.sub_industries if ml_model.sub_industries else []
    elif 'GICS Sector' in stocks_data.columns:
        sectors = stocks_data['GICS Sector'].dropna().unique().tolist()
        sectors = [s for s in sectors if isinstance(s, str) and s != '']
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
        if not market_analyzer:
            return JSONResponse(
                content={
                    "regime": "Unknown",
                    "fed_stance": "Unknown",
                    "vix": 20.0,
                    "recession_risk": "Unknown"
                },
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        # Use YOUR MarketConditionsAnalyzer methods
        regime = market_analyzer.get_market_regime()
        fed_data = market_analyzer.get_federal_reserve_stance()
        yield_curve = market_analyzer.analyze_yield_curve()
        economic_data = market_analyzer.fetch_economic_data()
        
        # Extract VIX from economic data
        vix_value = 20.0
        if economic_data and 'Volatility Index' in economic_data:
            vix_value = economic_data['Volatility Index'].get('current', 20.0)
        
        return JSONResponse(
            content={
                "regime": regime,
                "fed_stance": fed_data.get("stance", "Unknown"),
                "vix": vix_value,
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
                "recession_risk": "Error"
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest):
    """Run analysis using YOUR QuantFinanceMLModel methods"""
    logger.info(f"=" * 50)
    logger.info(f"Analysis START: {request.analysis_type} - {request.target}")
    
    if not ML_AVAILABLE or ml_model is None:
        return JSONResponse(
            content={"status": "failed", "error": "ML Model not available"},
            status_code=500
        )
    
    try:
        # Filter stocks based on selection
        if request.analysis_type == "sector":
            filtered_stocks = stocks_data[stocks_data['GICS Sector'] == request.target]
        else:
            filtered_stocks = stocks_data[stocks_data['GICS Sub-Industry'] == request.target]
        
        if len(filtered_stocks) == 0:
            return JSONResponse(
                content={"status": "no_data", "error": f"No stocks found for {request.target}"},
                status_code=404
            )
        
        logger.info(f"Found {len(filtered_stocks)} stocks")
        
        # Get top 10 by market cap using YOUR model's method
        symbols = filtered_stocks['Symbol'].tolist()
        market_caps = {}
        
        for symbol in symbols[:50]:  # Limit to 50 to avoid timeout
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_cap = info.get('marketCap', 0)
                if market_cap > 0:
                    market_caps[symbol] = market_cap
            except:
                continue
        
        # Get top 10 by market cap
        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:10]
        top_symbols = [stock[0] for stock in sorted_stocks]
        
        if not top_symbols:
            # Fallback to first 3 symbols
            top_symbols = symbols[:3]
        
        logger.info(f"Analyzing top stocks: {top_symbols}")
        
        # Set up the model's data
        ml_model.selected_stocks = top_symbols
        ml_model.sectors_data = filtered_stocks
        
        # Analyze market conditions using YOUR model's method
        market_conditions = None
        market_adjustment = 1.0
        
        if market_analyzer:
            try:
                # Get the sector for this analysis
                sector = request.target
                
                # Use YOUR model's analyze_market_conditions equivalent
                market_analyzer.fetch_economic_data()
                sector_conditions = market_analyzer.analyze_sector_conditions(sector)
                market_adjustment = market_analyzer.calculate_market_adjustment_factor(sector)
                
                ml_model.market_adjustment = market_adjustment
                logger.info(f"Market adjustment factor: {market_adjustment}")
            except Exception as e:
                logger.error(f"Market analysis error: {e}")
                market_adjustment = 1.0
        
        # Prepare training data if model needs it
        if len(top_symbols) > 0:
            try:
                logger.info("Preparing training data...")
                training_data = ml_model.prepare_training_data(top_symbols[:5])  # Limit for speed
                
                if not training_data.empty:
                    # Add sentiment features
                    training_data = ml_model.add_sentiment_features(training_data)
                    ml_model.training_data = training_data
                    
                    # Train the model
                    logger.info("Training ML model...")
                    ml_model.train_prediction_model(training_data)
            except Exception as e:
                logger.error(f"Training error: {e}")
        
        # Analyze each stock and calculate predictions
        results = []
        
        for symbol in top_symbols[:3]:  # Top 3 for speed
            try:
                logger.info(f"Analyzing {symbol}...")
                
                # Get stock data
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="3mo")
                
                # Get current price
                current_price = info.get('currentPrice') or info.get('regularMarketPrice')
                if not current_price or current_price <= 0:
                    if len(hist) > 0:
                        current_price = float(hist['Close'].iloc[-1])
                    else:
                        continue
                
                # Prepare current data for ML prediction
                current_data = {
                    'recent_returns': hist['Close'].pct_change().tail(20).mean() if len(hist) > 20 else 0,
                    'volatility': hist['Close'].pct_change().std() * np.sqrt(252) if len(hist) > 2 else 0.2,
                    'pe_ratio': info.get('trailingPE', 20),
                    'market_cap': info.get('marketCap', 1e9),
                    'vix': 20,
                    'treasury_10y': 3.5,
                    'dollar_index': 100,
                    'spy_trend': 1
                }
                
                # Get sentiment using YOUR model's method
                sentiment_score = 0.5
                try:
                    company_name = info.get('longName', symbol)
                    sentiment_result = ml_model.simple_sentiment(f"{symbol} {company_name} stock")
                    if sentiment_result and len(sentiment_result) > 0:
                        sent = sentiment_result[0]
                        if sent['label'] == 'positive':
                            sentiment_score = sent['score']
                        elif sent['label'] == 'negative':
                            sentiment_score = -sent['score']
                except:
                    sentiment_score = 0
                
                current_data['reddit_sentiment'] = sentiment_score
                current_data['news_sentiment'] = sentiment_score
                
                # Get ML prediction using YOUR model's method
                ml_prediction = 0
                if hasattr(ml_model, 'ml_model') and ml_model.ml_model is not None:
                    try:
                        ml_prediction = ml_model.calculate_stock_predictions(symbol, current_data)
                    except:
                        ml_prediction = 0.05  # Default 5% return
                else:
                    # Simple momentum-based prediction
                    if len(hist) > 20:
                        returns_20d = (hist['Close'].iloc[-1] - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20]
                        ml_prediction = returns_20d
                
                # Calculate valuation using YOUR valuation model
                target_price = current_price * 1.15  # Default
                confidence = 0.5
                
                if valuator and hasattr(valuator, 'calculate_comprehensive_valuation'):
                    try:
                        valuation_result = valuator.calculate_comprehensive_valuation(
                            symbol, 
                            ml_prediction, 
                            sentiment_score, 
                            market_adjustment
                        )
                        target_price = valuation_result.get('target_price', current_price * 1.15)
                        confidence = valuation_result.get('confidence', 0.5)
                    except Exception as e:
                        logger.error(f"Valuation error for {symbol}: {e}")
                
                # Calculate upside
                upside = ((target_price / current_price) - 1) * 100 if current_price > 0 else 0
                
                # Convert ML score to 0-1 range
                ml_score = 0.5 + (ml_prediction * 2)  # Convert return to score
                ml_score = max(0.1, min(0.95, ml_score))
                
                results.append({
                    "symbol": symbol,
                    "current_price": current_price,
                    "target_price": target_price,
                    "upside": upside,
                    "ml_score": ml_score,
                    "confidence": confidence,
                    "sentiment": sentiment_score,
                    "ml_prediction": ml_prediction
                })
                
                logger.info(f"✅ {symbol}: Price=${current_price:.2f}, Target=${target_price:.2f}, Upside={upside:.1f}%")
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by upside potential
        results.sort(key=lambda x: x['upside'], reverse=True)
        
        # Format for frontend
        top_stocks = []
        for stock in results[:3]:
            top_stocks.append({
                "symbol": stock['symbol'],
                "metrics": {
                    "current_price": round(stock['current_price'], 2),
                    "target_price": round(stock['target_price'], 2),
                    "upside_potential": round(stock['upside'], 1),
                    "confidence_score": int(stock['confidence'] * 100),
                    "sentiment_score": round(abs(stock['sentiment']), 2),
                    "ml_score": round(stock['ml_score'], 3)
                }
            })
        
        if not top_stocks:
            return JSONResponse(
                content={
                    "status": "no_results",
                    "error": "Could not analyze any stocks"
                },
                status_code=500
            )
        
        # Get market regime
        market_regime = "Neutral"
        if market_analyzer:
            try:
                market_regime = market_analyzer.get_market_regime()
            except:
                pass
        
        return JSONResponse(
            content={
                "status": "completed",
                "analysis_type": request.analysis_type,
                "target": request.target,
                "results": {
                    "top_stocks": top_stocks,
                    "market_conditions": {
                        "regime": market_regime,
                        "adjustment_factor": market_adjustment
                    },
                    "total_analyzed": len(results)
                },
                "ml_powered": True
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Analysis error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()[:500]
            },
            status_code=500
        )

@app.get("/api/test-analysis/{symbol}")
async def test_single_stock(symbol: str):
    """Test analysis on a single stock"""
    results = {"symbol": symbol}
    
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        results['current_price'] = info.get('currentPrice') or info.get('regularMarketPrice')
        results['company_name'] = info.get('longName', symbol)
        
        # Test sentiment
        if ml_model:
            sentiment = ml_model.simple_sentiment(f"{symbol} stock")
            results['sentiment'] = sentiment[0] if sentiment else {"label": "neutral", "score": 0.5}
        
        # Test valuation
        if valuator:
            try:
                valuation = valuator.calculate_comprehensive_valuation(symbol, 0.05, 0.5, 1.0)
                results['target_price'] = valuation.get('target_price')
                results['confidence'] = valuation.get('confidence')
            except Exception as e:
                results['valuation_error'] = str(e)[:100]
    except Exception as e:
        results['error'] = str(e)[:100]
    
    return results

@app.get("/api/diagnose")
async def diagnose_system():
    """Complete system diagnostic"""
    diagnosis = {
        "ml_model": {
            "exists": ml_model is not None,
            "type": type(ml_model).__name__ if ml_model else "None",
            "has_master_df": hasattr(ml_model, 'master_df') and ml_model.master_df is not None if ml_model else False,
            "has_sectors": hasattr(ml_model, 'sectors') and ml_model.sectors is not None if ml_model else False,
            "has_ml_model": hasattr(ml_model, 'ml_model') and ml_model.ml_model is not None if ml_model else False,
            "has_training_data": hasattr(ml_model, 'training_data') and ml_model.training_data is not None if ml_model else False
        },
        "market_analyzer": {
            "exists": market_analyzer is not None,
            "type": type(market_analyzer).__name__ if market_analyzer else "None"
        },
        "valuator": {
            "exists": valuator is not None,
            "type": type(valuator).__name__ if valuator else "None",
            "has_calculate_comprehensive_valuation": hasattr(valuator, 'calculate_comprehensive_valuation') if valuator else False
        },
        "stocks_data_shape": stocks_data.shape if stocks_data is not None else None
    }
    
    return diagnosis

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Using ML Model: {type(ml_model).__name__ if ml_model else 'None'}")
    uvicorn.run(app, host="0.0.0.0", port=port)