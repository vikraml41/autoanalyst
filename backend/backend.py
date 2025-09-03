#!/usr/bin/env python3
"""
FastAPI Backend - Full ML with Job Queue System
"""

import os
import sys
import glob
import logging
import json
import time
import random
import uuid
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import threading
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="14.0.0")

# Thread pool
executor = ThreadPoolExecutor(max_workers=5)

# Job storage (in-memory for now)
analysis_jobs = {}
job_lock = threading.Lock()

# Cache
cache = {}
CACHE_DURATION = 3600

# Rate limiting
YAHOO_DELAY = 0.5
last_yahoo_request = 0

# Market data cache
market_data_cache = None
market_data_timestamp = 0

# Market Cap Ranges
MARKET_CAP_RANGES = {
    'large': {'min': 10_000_000_000, 'max': float('inf'), 'label': 'Large Cap (>$10B)'},
    'mid': {'min': 2_000_000_000, 'max': 10_000_000_000, 'label': 'Mid Cap ($2B-$10B)'},
    'small': {'min': 300_000_000, 'max': 2_000_000_000, 'label': 'Small Cap ($300M-$2B)'},
    'all': {'min': 0, 'max': float('inf'), 'label': 'All Market Caps'}
}

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# ============ FIX VIX FETCHING ============

def fetch_vix_with_fallback():
    """Fetch VIX with multiple methods"""
    vix_value = None
    
    try:
        # Method 1: Direct ticker
        logger.info("Fetching VIX...")
        vix = yf.download("^VIX", period="1d", progress=False, timeout=5)
        if not vix.empty:
            vix_value = float(vix['Close'].iloc[-1])
            logger.info(f"✓ VIX from download: {vix_value:.2f}")
            return vix_value
    except Exception as e:
        logger.warning(f"VIX download failed: {e}")
    
    try:
        # Method 2: Ticker info
        vix_ticker = yf.Ticker("^VIX")
        vix_info = vix_ticker.fast_info
        if hasattr(vix_info, 'last_price'):
            vix_value = float(vix_info.last_price)
            logger.info(f"✓ VIX from fast_info: {vix_value:.2f}")
            return vix_value
    except Exception as e:
        logger.warning(f"VIX fast_info failed: {e}")
    
    try:
        # Method 3: History
        vix_ticker = yf.Ticker("^VIX")
        hist = vix_ticker.history(period="5d")
        if not hist.empty:
            vix_value = float(hist['Close'].iloc[-1])
            logger.info(f"✓ VIX from history: {vix_value:.2f}")
            return vix_value
    except Exception as e:
        logger.warning(f"VIX history failed: {e}")
    
    # Default if all fail
    logger.error("All VIX fetch methods failed, using default")
    return 20.0

def get_market_data():
    """Get comprehensive market data"""
    global market_data_cache, market_data_timestamp
    
    # Cache for 15 minutes
    if market_data_cache and (time.time() - market_data_timestamp) < 900:
        return market_data_cache
    
    market_data = {
        'vix': 20.0,
        'spy_price': 500,
        'spy_trend': 1,
        'treasury_10y': 4.3,
        'dollar_index': 105
    }
    
    try:
        # Get VIX
        market_data['vix'] = fetch_vix_with_fallback()
        
        # Get SPY
        try:
            spy_data = yf.download("SPY", period="1mo", progress=False, timeout=5)
            if not spy_data.empty:
                market_data['spy_price'] = float(spy_data['Close'].iloc[-1])
                month_return = (spy_data['Close'].iloc[-1] - spy_data['Close'].iloc[0]) / spy_data['Close'].iloc[0]
                market_data['spy_trend'] = 1 + month_return
                logger.info(f"SPY: ${market_data['spy_price']:.2f}")
        except:
            pass
        
        # Get 10Y Treasury
        try:
            tnx_data = yf.download("^TNX", period="1d", progress=False, timeout=5)
            if not tnx_data.empty:
                market_data['treasury_10y'] = float(tnx_data['Close'].iloc[-1])
                logger.info(f"10Y: {market_data['treasury_10y']:.2f}%")
        except:
            pass
        
    except Exception as e:
        logger.error(f"Market data error: {e}")
    
    market_data_cache = market_data
    market_data_timestamp = time.time()
    
    return market_data

# ============ DATA LOADING ============

def load_csv_files():
    """Load CSV files"""
    data_path = os.environ.get('DATA_PATH', '/app/data')
    if not os.path.exists(data_path):
        data_path = 'data'
    
    logger.info(f"Loading CSV files from: {data_path}")
    csv_pattern = os.path.join(data_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        return pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
            'GICS Sector': ['Technology'] * 7,
            'GICS Sub-Industry': ['Hardware', 'Software', 'Internet', 'E-Commerce',
                                 'Semiconductors', 'Social Media', 'Automobiles']
        })
    
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
            logger.info(f"Loaded {len(df)} rows")
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total stocks: {len(combined)}")
        return combined
    return pd.DataFrame()

stocks_data = load_csv_files()

# Import ML models
ML_AVAILABLE = False
ml_model = None
market_analyzer = None
valuator = None

try:
    logger.info("Importing quant_model.py...")
    from quant_model import (
        QuantFinanceMLModel, 
        MarketConditionsAnalyzer, 
        EnhancedValuation
    )
    
    ml_model = QuantFinanceMLModel()
    ml_model.master_df = stocks_data
    ml_model.process_gics_data()
    
    market_analyzer = MarketConditionsAnalyzer()
    valuator = EnhancedValuation()
    
    ML_AVAILABLE = True
    logger.info("✅ Models loaded successfully!")
    
except Exception as e:
    logger.error(f"❌ Error loading quant_model.py: {e}")
    ML_AVAILABLE = False

# ============ BACKGROUND ANALYSIS ============

def run_analysis_background(job_id, request_data):
    """Run analysis in background to avoid timeout"""
    try:
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'processing',
                'progress': 'Starting analysis...',
                'started': time.time()
            }
        
        analysis_type = request_data['analysis_type']
        target = request_data['target']
        market_cap_size = request_data.get('market_cap_size', 'all')
        
        logger.info(f"Job {job_id}: Analyzing {target} ({market_cap_size})")
        
        # Filter stocks
        if analysis_type == "sector":
            filtered_stocks = stocks_data[stocks_data['GICS Sector'] == target]
        else:
            filtered_stocks = stocks_data[stocks_data['GICS Sub-Industry'] == target]
        
        if len(filtered_stocks) == 0:
            with job_lock:
                analysis_jobs[job_id] = {
                    'status': 'error',
                    'error': f'No stocks found for {target}'
                }
            return
        
        # Update progress
        with job_lock:
            analysis_jobs[job_id]['progress'] = f'Found {len(filtered_stocks)} stocks'
        
        # Get symbols and filter by market cap
        symbols = filtered_stocks['Symbol'].tolist()[:50]
        
        # Filter by market cap if needed
        if market_cap_size != 'all':
            cap_range = MARKET_CAP_RANGES[market_cap_size]
            filtered_symbols = []
            
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    cap = ticker.info.get('marketCap', 0)
                    if cap_range['min'] <= cap < cap_range['max']:
                        filtered_symbols.append((symbol, cap))
                    if len(filtered_symbols) >= 15:
                        break
                    time.sleep(0.3)
                except:
                    continue
            
            filtered_symbols.sort(key=lambda x: x[1], reverse=True)
            analysis_symbols = [s[0] for s in filtered_symbols[:10]]
        else:
            # Get top by market cap
            market_caps = {}
            for symbol in symbols[:30]:
                try:
                    ticker = yf.Ticker(symbol)
                    cap = ticker.info.get('marketCap', 0)
                    if cap > 0:
                        market_caps[symbol] = cap
                    time.sleep(0.3)
                except:
                    continue
            
            sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:10]
            analysis_symbols = [s[0] for s in sorted_stocks]
        
        # Update progress
        with job_lock:
            analysis_jobs[job_id]['progress'] = f'Analyzing {len(analysis_symbols)} stocks'
        
        # Setup ML model
        if ML_AVAILABLE and ml_model:
            ml_model.selected_stocks = analysis_symbols
            ml_model.master_df = stocks_data
            ml_model.process_gics_data()
            
            # Train model
            try:
                training_data = ml_model.prepare_training_data(analysis_symbols[:7])
                if not training_data.empty:
                    training_data = ml_model.add_sentiment_features(training_data)
                    ml_model.train_prediction_model(training_data)
            except:
                pass
        
        # Get market data
        market_data = get_market_data()
        
        # Analyze each stock
        results = []
        for i, symbol in enumerate(analysis_symbols):
            try:
                with job_lock:
                    analysis_jobs[job_id]['progress'] = f'Analyzing {symbol} ({i+1}/{len(analysis_symbols)})'
                
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="3mo")
                
                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                if not current_price and not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                
                if not current_price or current_price <= 0:
                    continue
                
                # Full metrics
                metrics = {
                    'symbol': symbol,
                    'company_name': info.get('longName', symbol),
                    'current_price': current_price,
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'peg_ratio': info.get('pegRatio', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'revenue_growth': info.get('revenueGrowth', 0),
                    'debt_to_equity': info.get('debtToEquity', 0),
                    'roe': info.get('returnOnEquity', 0),
                    'target_price': info.get('targetMeanPrice', current_price)
                }
                
                # ML prediction
                ml_prediction = 0
                if ML_AVAILABLE and ml_model:
                    try:
                        features = {
                            'recent_returns': 0,
                            'volatility': 0.3,
                            'pe_ratio': metrics['pe_ratio'],
                            'market_cap': metrics['market_cap'],
                            'peg_ratio': metrics['peg_ratio'],
                            'profit_margin': metrics['profit_margin'],
                            'revenue_growth': metrics['revenue_growth'],
                            'debt_to_equity': metrics['debt_to_equity'],
                            'roe': metrics['roe'],
                            'price_to_book': info.get('priceToBook', 1),
                            'rsi': 50,
                            'vix': market_data['vix'],  # REAL VIX
                            'treasury_10y': market_data['treasury_10y'],
                            'dollar_index': market_data['dollar_index'],
                            'spy_trend': market_data['spy_trend']
                        }
                        ml_prediction = ml_model.calculate_stock_predictions(symbol, features)
                    except:
                        ml_prediction = 0
                
                # Sentiment
                sentiment_score = 0
                if ML_AVAILABLE and ml_model and hasattr(ml_model, 'simple_sentiment'):
                    try:
                        sentiment_result = ml_model.simple_sentiment(f"{symbol} stock outlook")
                        if sentiment_result:
                            sent = sentiment_result[0]
                            if sent['label'] == 'positive':
                                sentiment_score = sent['score']
                            elif sent['label'] == 'negative':
                                sentiment_score = -sent['score']
                    except:
                        pass
                
                # Valuation
                if valuator:
                    try:
                        val_result = valuator.calculate_comprehensive_valuation(
                            symbol, ml_prediction, sentiment_score, 1.0
                        )
                        metrics['target_price'] = val_result.get('target_price', metrics['target_price'])
                    except:
                        pass
                
                # Calculate scores
                quality_score = 0
                if 0 < metrics['pe_ratio'] < 25:
                    quality_score += 0.2
                if 0 < metrics['peg_ratio'] < 1.5:
                    quality_score += 0.2
                if metrics['roe'] > 0.15:
                    quality_score += 0.2
                if metrics['revenue_growth'] > 0.1:
                    quality_score += 0.2
                
                upside = ((metrics['target_price'] / current_price) - 1) * 100
                
                if upside > 0:
                    results.append({
                        'symbol': symbol,
                        'company_name': metrics['company_name'],
                        'current_price': current_price,
                        'target_price': metrics['target_price'],
                        'upside': upside,
                        'quality_score': quality_score,
                        'ml_prediction': ml_prediction,
                        'sentiment_score': sentiment_score,
                        'metrics': metrics
                    })
                
                time.sleep(0.5)  # Rate limit
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort and get top 3
        results.sort(key=lambda x: x['upside'], reverse=True)
        top_3 = results[:3]
        
        # Format results
        formatted = []
        for stock in top_3:
            formatted.append({
                "symbol": stock['symbol'],
                "company_name": stock['company_name'],
                "market_cap": f"${stock['metrics']['market_cap']/1e9:.2f}B" if stock['metrics']['market_cap'] > 1e9 else f"${stock['metrics']['market_cap']/1e6:.0f}M",
                "metrics": {
                    "current_price": round(stock['current_price'], 2),
                    "target_price": round(stock['target_price'], 2),
                    "upside_potential": round(stock['upside'], 1),
                    "confidence_score": int(stock['quality_score'] * 100),
                    "sentiment_score": round(stock['sentiment_score'], 2),
                    "ml_score": round(stock['ml_prediction'], 3)
                },
                "analysis_details": {
                    "fundamentals": {
                        "pe_ratio": stock['metrics']['pe_ratio'],
                        "peg_ratio": stock['metrics']['peg_ratio'],
                        "roe": stock['metrics']['roe'],
                        "profit_margin": stock['metrics']['profit_margin'],
                        "revenue_growth": stock['metrics']['revenue_growth'],
                        "debt_to_equity": stock['metrics']['debt_to_equity']
                    },
                    "technicals": {
                        "rsi": 50,
                        "momentum_20d": 0,
                        "momentum_60d": 0,
                        "volatility": 0.3
                    },
                    "ml_prediction": stock['ml_prediction'],
                    "quality_score": stock['quality_score']
                }
            })
        
        # Store results
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'completed',
                'results': {
                    'top_stocks': formatted,
                    'market_conditions': {
                        'regime': 'Normal',
                        'vix': market_data['vix']
                    },
                    'total_analyzed': len(results)
                },
                'completed': time.time()
            }
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'error',
                'error': str(e)
            }

# ============ API ENDPOINTS ============

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str
    market_cap_size: str = 'all'

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start analysis job"""
    job_id = str(uuid.uuid4())
    
    # Start background job
    background_tasks.add_task(
        run_analysis_background,
        job_id,
        request.dict()
    )
    
    return JSONResponse(
        content={
            "job_id": job_id,
            "status": "started",
            "message": "Analysis started. Poll /api/analysis/{job_id} for results"
        },
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.get("/api/analysis/{job_id}")
async def get_analysis_status(job_id: str):
    """Get analysis job status"""
    with job_lock:
        if job_id not in analysis_jobs:
            return JSONResponse(
                content={"status": "not_found", "error": "Job not found"},
                status_code=404
            )
        
        job = analysis_jobs[job_id]
        
        if job['status'] == 'completed':
            # Return and clean up
            result = {
                "status": "completed",
                "results": job['results'],
                "ml_powered": ML_AVAILABLE
            }
            del analysis_jobs[job_id]  # Clean up completed job
            return JSONResponse(content=result)
        
        elif job['status'] == 'error':
            result = {
                "status": "error",
                "error": job.get('error', 'Unknown error')
            }
            del analysis_jobs[job_id]  # Clean up failed job
            return JSONResponse(content=result, status_code=500)
        
        else:
            return JSONResponse(
                content={
                    "status": "processing",
                    "progress": job.get('progress', 'Processing...')
                }
            )

@app.get("/api/market-conditions")
async def get_market_conditions():
    """Get real market conditions"""
    try:
        market_data = get_market_data()
        vix = market_data['vix']
        
        # Determine regime
        if vix < 12:
            regime = "Very Low Volatility"
        elif vix < 20:
            regime = "Normal"
        elif vix < 30:
            regime = "Elevated Volatility"
        else:
            regime = "High Volatility"
        
        return JSONResponse(
            content={
                "regime": regime,
                "fed_stance": "Neutral",
                "vix": vix,  # REAL VIX VALUE
                "recession_risk": "Low" if vix < 25 else "Elevated",
                "ml_powered": ML_AVAILABLE
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"Market conditions error: {e}")
        return JSONResponse(
            content={
                "regime": "Unknown",
                "fed_stance": "Unknown",
                "vix": 20.0,
                "recession_risk": "Unknown"
            }
        )

@app.get("/api/stocks/list")
async def get_stocks_list():
    sectors = stocks_data['GICS Sector'].dropna().unique().tolist() if 'GICS Sector' in stocks_data.columns else []
    sub_industries = stocks_data['GICS Sub-Industry'].dropna().unique().tolist() if 'GICS Sub-Industry' in stocks_data.columns else []
    
    return JSONResponse(
        content={
            "sectors": [s for s in sectors if s],
            "sub_industries": [s for s in sub_industries if s],
            "total_stocks": len(stocks_data)
        }
    )

@app.get("/")
async def root():
    return JSONResponse(
        content={
            "name": "AutoAnalyst API",
            "version": "14.0.0",
            "status": "running",
            "ml_enabled": ML_AVAILABLE,
            "stocks_loaded": len(stocks_data)
        }
    )

@app.on_event("startup")
async def startup_event():
    """Fetch initial market data"""
    logger.info("Starting up...")
    get_market_data()
    logger.info(f"Ready - VIX: {market_data_cache['vix'] if market_data_cache else 'N/A'}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)