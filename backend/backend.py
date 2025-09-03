#!/usr/bin/env python3
"""
FastAPI Backend - With Market Cap Selection Feature
"""

import os
import sys
import glob
import logging
import json
import time
import random
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="12.0.0")

# Thread pool
executor = ThreadPoolExecutor(max_workers=5)

# In-memory cache
cache = {}
CACHE_DURATION = 3600

# Rate limiting
YAHOO_DELAY = 0.5
last_yahoo_request = 0

# Market Cap Ranges (in billions)
MARKET_CAP_RANGES = {
    'large': {'min': 10_000_000_000, 'max': float('inf'), 'label': 'Large Cap (>$10B)'},
    'mid': {'min': 2_000_000_000, 'max': 10_000_000_000, 'label': 'Mid Cap ($2B-$10B)'},
    'small': {'min': 300_000_000, 'max': 2_000_000_000, 'label': 'Small Cap ($300M-$2B)'},
    'micro': {'min': 50_000_000, 'max': 300_000_000, 'label': 'Micro Cap ($50M-$300M)'},
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

# ============ RATE LIMITED YAHOO FINANCE ============

def rate_limited_yahoo_request(func):
    """Decorator to rate limit Yahoo Finance requests"""
    def wrapper(*args, **kwargs):
        global last_yahoo_request
        
        elapsed = time.time() - last_yahoo_request
        if elapsed < YAHOO_DELAY:
            time.sleep(YAHOO_DELAY - elapsed)
        
        last_yahoo_request = time.time()
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                if "timed out" in str(e).lower() or "connection" in str(e).lower():
                    wait_time = (attempt + 1) * 2 + random.uniform(0, 1)
                    logger.warning(f"Yahoo Finance timeout, retry {attempt + 1}/{max_retries} after {wait_time:.1f}s")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
        
        return None
    return wrapper

# ============ CACHING SYSTEM ============

def get_cache_key(prefix, *args):
    """Generate cache key"""
    key_str = f"{prefix}:{':'.join(str(arg) for arg in args)}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_from_cache(key):
    """Get from cache if not expired"""
    if key in cache:
        data, timestamp = cache[key]
        if time.time() - timestamp < CACHE_DURATION:
            return data
        else:
            del cache[key]
    return None

def set_cache(key, data):
    """Set cache with timestamp"""
    cache[key] = (data, time.time())

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
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'CRM'],
            'GICS Sector': ['Technology'] * 10,
            'GICS Sub-Industry': ['Hardware', 'Software', 'Internet', 'E-Commerce',
                                 'Semiconductors', 'Social Media', 'Automobiles',
                                 'Semiconductors', 'Semiconductors', 'Software']
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

# Load data
stocks_data = load_csv_files()
logger.info(f"Stocks data initialized: {len(stocks_data)} stocks")

# Import models
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

# ============ MARKET CAP FILTERING ============

def filter_by_market_cap(symbols, market_cap_size='all'):
    """Filter stocks by market cap category"""
    if market_cap_size == 'all':
        return symbols
    
    cap_range = MARKET_CAP_RANGES.get(market_cap_size, MARKET_CAP_RANGES['all'])
    filtered_symbols = []
    
    logger.info(f"Filtering for {cap_range['label']}")
    
    for symbol in symbols:
        try:
            # Check cache first
            cache_key = get_cache_key('market_cap', symbol)
            market_cap = get_from_cache(cache_key)
            
            if market_cap is None:
                ticker = yf.Ticker(symbol)
                market_cap = ticker.info.get('marketCap', 0)
                if market_cap > 0:
                    set_cache(cache_key, market_cap)
                time.sleep(0.2)
            
            if cap_range['min'] <= market_cap < cap_range['max']:
                filtered_symbols.append((symbol, market_cap))
                logger.info(f"✓ {symbol}: ${market_cap/1e9:.2f}B")
            
            # Limit checks to avoid timeout
            if len(filtered_symbols) >= 20:
                break
                
        except Exception as e:
            logger.warning(f"Error checking {symbol}: {e}")
            continue
    
    # Sort by market cap and return symbols
    filtered_symbols.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in filtered_symbols]

# ============ STOCK FETCHING ============

@rate_limited_yahoo_request
def fetch_single_stock_safe(symbol):
    """Safely fetch single stock data"""
    try:
        cache_key = get_cache_key('stock_info', symbol)
        cached_data = get_from_cache(cache_key)
        if cached_data:
            return symbol, cached_data
        
        session = yf.utils.get_session()
        session.timeout = 10
        
        ticker = yf.Ticker(symbol, session=session)
        
        info = {}
        try:
            info = ticker.info
        except Exception as e:
            logger.warning(f"Failed to get info for {symbol}: {e}")
        
        hist = pd.DataFrame()
        try:
            hist = ticker.history(period="3mo", timeout=10)
        except Exception as e:
            logger.warning(f"Failed to get history for {symbol}: {e}")
        
        if info or not hist.empty:
            data = {
                'info': info,
                'history': hist.to_dict() if not hist.empty else {}
            }
            set_cache(cache_key, data)
            return symbol, data
        
        return symbol, None
        
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return symbol, None

def batch_fetch_stocks_sequential(symbols):
    """Fetch stocks sequentially with rate limiting"""
    results = {}
    
    for i, symbol in enumerate(symbols):
        try:
            logger.info(f"Fetching {symbol} ({i+1}/{len(symbols)})...")
            
            cache_key = get_cache_key('stock_info', symbol)
            cached_data = get_from_cache(cache_key)
            
            if cached_data:
                results[symbol] = cached_data
                logger.info(f"✓ {symbol} (cached)")
            else:
                _, data = fetch_single_stock_safe(symbol)
                if data:
                    results[symbol] = data
                    logger.info(f"✓ {symbol} fetched")
                else:
                    logger.warning(f"✗ {symbol} failed")
            
            if i < len(symbols) - 1:
                time.sleep(0.3)
                
        except Exception as e:
            logger.error(f"Error with {symbol}: {e}")
            continue
    
    return results

# ============ ANALYSIS ============

def analyze_stock_simple(symbol, stock_data, ml_model=None):
    """Simple stock analysis"""
    try:
        info = stock_data.get('info', {})
        hist_dict = stock_data.get('history', {})
        
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        if not current_price and hist_dict.get('Close'):
            close_prices = list(hist_dict['Close'].values())
            if close_prices:
                current_price = close_prices[-1]
        
        if not current_price or current_price <= 0:
            return None
        
        # Get market cap for display
        market_cap = info.get('marketCap', 0)
        market_cap_str = f"${market_cap/1e9:.2f}B" if market_cap > 1e9 else f"${market_cap/1e6:.0f}M"
        
        metrics = {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'current_price': current_price,
            'market_cap': market_cap,
            'market_cap_str': market_cap_str,
            'pe_ratio': info.get('trailingPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'profit_margin': info.get('profitMargins', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'roe': info.get('returnOnEquity', 0),
            'analyst_rating': info.get('recommendationMean', 3),
            'target_mean_price': info.get('targetMeanPrice', current_price),
            'target_high_price': info.get('targetHighPrice', current_price)
        }
        
        # Calculate scores
        quality_score = 0
        reasons = []
        
        if 0 < metrics['pe_ratio'] < 25:
            quality_score += 0.2
            reasons.append(f"Good P/E ratio: {metrics['pe_ratio']:.1f}")
        
        if 0 < metrics['peg_ratio'] < 1.5:
            quality_score += 0.2
            reasons.append(f"Attractive PEG: {metrics['peg_ratio']:.2f}")
        
        if metrics['roe'] > 0.15:
            quality_score += 0.2
            reasons.append(f"Strong ROE: {metrics['roe']*100:.1f}%")
        
        if metrics['revenue_growth'] > 0.1:
            quality_score += 0.2
            reasons.append(f"Good growth: {metrics['revenue_growth']*100:.1f}%")
        
        if metrics['analyst_rating'] < 2.5:
            quality_score += 0.2
            reasons.append("Strong analyst rating")
        
        upside = ((metrics['target_mean_price'] / current_price) - 1) * 100 if current_price > 0 else 0
        
        # ML prediction
        ml_prediction = 0
        if ml_model and hasattr(ml_model, 'calculate_stock_predictions'):
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
                    'vix': 20,
                    'treasury_10y': 3.5,
                    'dollar_index': 100,
                    'spy_trend': 1
                }
                ml_prediction = ml_model.calculate_stock_predictions(symbol, features)
            except:
                ml_prediction = 0
        
        combined_score = (upside * 0.5) + (quality_score * 100 * 0.5) + (ml_prediction * 100 * 0.2)
        
        return {
            'symbol': symbol,
            'company_name': metrics['company_name'],
            'market_cap_str': market_cap_str,
            'current_price': current_price,
            'target_price': metrics['target_mean_price'],
            'upside': upside,
            'quality_score': quality_score,
            'ml_prediction': ml_prediction,
            'combined_score': combined_score,
            'metrics': metrics,
            'reasons': reasons
        }
        
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}")
        return None

# ============ ANALYSIS REQUEST MODEL ============

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str
    market_cap_size: str = 'all'  # New field with default

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest):
    """Analysis with market cap filtering"""
    start_time = time.time()
    logger.info(f"="*50)
    logger.info(f"Analysis: {request.analysis_type} - {request.target}")
    logger.info(f"Market Cap Filter: {request.market_cap_size}")
    
    # Check cache
    cache_key = get_cache_key('analysis', request.analysis_type, request.target, request.market_cap_size)
    cached_result = get_from_cache(cache_key)
    if cached_result:
        logger.info("Returning cached result")
        return JSONResponse(content=cached_result, headers={"Access-Control-Allow-Origin": "*"})
    
    try:
        # Filter by sector/industry
        if request.analysis_type == "sector":
            filtered_stocks = stocks_data[stocks_data['GICS Sector'] == request.target]
        else:
            filtered_stocks = stocks_data[stocks_data['GICS Sub-Industry'] == request.target]
        
        if len(filtered_stocks) == 0:
            return JSONResponse(
                content={"status": "no_data", "error": f"No stocks found for {request.target}"},
                status_code=404
            )
        
        logger.info(f"Found {len(filtered_stocks)} stocks in {request.target}")
        
        # Get symbols
        all_symbols = filtered_stocks['Symbol'].tolist()
        
        # Filter by market cap
        if request.market_cap_size != 'all':
            logger.info(f"Filtering by market cap: {request.market_cap_size}")
            filtered_symbols = filter_by_market_cap(all_symbols[:50], request.market_cap_size)
            
            if not filtered_symbols:
                return JSONResponse(
                    content={
                        "status": "no_data", 
                        "error": f"No {MARKET_CAP_RANGES[request.market_cap_size]['label']} stocks found in {request.target}"
                    },
                    status_code=404
                )
            
            analysis_symbols = filtered_symbols[:10]
        else:
            # Get top stocks by market cap
            market_caps = {}
            for symbol in all_symbols[:30]:
                try:
                    cache_key = get_cache_key('market_cap', symbol)
                    cap = get_from_cache(cache_key)
                    
                    if cap is None:
                        ticker = yf.Ticker(symbol)
                        cap = ticker.info.get('marketCap', 0)
                        if cap > 0:
                            set_cache(cache_key, cap)
                        time.sleep(0.2)
                    
                    if cap and cap > 0:
                        market_caps[symbol] = cap
                except:
                    continue
            
            if market_caps:
                sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:10]
                analysis_symbols = [stock[0] for stock in sorted_stocks]
            else:
                analysis_symbols = all_symbols[:10]
        
        logger.info(f"Analyzing {len(analysis_symbols)} stocks: {', '.join(analysis_symbols)}")
        
        # Fetch stock data
        stock_data_dict = batch_fetch_stocks_sequential(analysis_symbols)
        
        if not stock_data_dict:
            return JSONResponse(
                content={"status": "error", "error": "Failed to fetch stock data"},
                status_code=500
            )
        
        # Setup ML if available
        if ML_AVAILABLE and ml_model:
            try:
                ml_model.selected_stocks = list(stock_data_dict.keys())
                ml_model.master_df = stocks_data
                ml_model.process_gics_data()
            except:
                pass
        
        # Analyze stocks
        results = []
        for symbol, data in stock_data_dict.items():
            if data:
                analysis = analyze_stock_simple(symbol, data, ml_model)
                if analysis and analysis['upside'] > 0:
                    results.append(analysis)
                    logger.info(f"✓ {symbol} ({analysis['market_cap_str']}): Score={analysis['combined_score']:.1f}, Upside={analysis['upside']:.1f}%")
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Get top 3
        top_3 = results[:3]
        
        # Format results
        formatted_results = []
        for rank, stock in enumerate(top_3, 1):
            formatted_results.append({
                "symbol": stock['symbol'],
                "company_name": stock['company_name'],
                "market_cap": stock['market_cap_str'],
                "metrics": {
                    "current_price": round(stock['current_price'], 2),
                    "target_price": round(stock['target_price'], 2),
                    "upside_potential": round(stock['upside'], 1),
                    "confidence_score": int(stock['quality_score'] * 100),
                    "sentiment_score": 0.5,
                    "ml_score": round(stock['ml_prediction'], 3) if stock['ml_prediction'] else 0
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
                },
                "investment_thesis": stock['reasons']
            })
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Analysis completed in {elapsed:.1f} seconds")
        
        result_data = {
            "status": "completed",
            "analysis_type": request.analysis_type,
            "target": request.target,
            "market_cap_filter": MARKET_CAP_RANGES[request.market_cap_size]['label'],
            "results": {
                "top_stocks": formatted_results,
                "market_conditions": {
                    "regime": "Normal",
                    "adjustment_factor": 1.0
                },
                "total_analyzed": len(results),
                "total_qualified": len([r for r in results if r['upside'] > 0])
            },
            "ml_powered": ML_AVAILABLE
        }
        
        # Cache result
        set_cache(cache_key, result_data)
        
        return JSONResponse(
            content=result_data,
            headers={"Access-Control-Allow-Origin": "*"}
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Analysis error: {e}")
        logger.error(traceback.format_exc())
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500
        )

# Keep other endpoints the same...
@app.get("/")
async def root():
    return JSONResponse(
        content={
            "name": "AutoAnalyst API",
            "version": "12.0.0",
            "status": "running",
            "ml_enabled": ML_AVAILABLE,
            "stocks_loaded": len(stocks_data)
        }
    )

@app.get("/api/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "ml_available": ML_AVAILABLE,
            "stocks_count": len(stocks_data),
            "timestamp": datetime.now().isoformat()
        }
    )

@app.get("/api/stocks/list")
async def get_stocks_list():
    sectors = []
    sub_industries = []
    
    if 'GICS Sector' in stocks_data.columns:
        sectors = stocks_data['GICS Sector'].dropna().unique().tolist()
        sectors = [s for s in sectors if isinstance(s, str) and s != '']
        sub_industries = stocks_data['GICS Sub-Industry'].dropna().unique().tolist()
        sub_industries = [s for s in sub_industries if isinstance(s, str) and s != '']
    
    return JSONResponse(
        content={
            "sectors": sectors,
            "sub_industries": sub_industries,
            "total_stocks": len(stocks_data),
            "market_cap_categories": list(MARKET_CAP_RANGES.keys())
        }
    )

@app.get("/api/market-conditions")
async def get_market_conditions():
    try:
        return JSONResponse(
            content={
                "regime": "Normal",
                "fed_stance": "Neutral",
                "vix": 20.0,
                "recession_risk": "Low",
                "ml_powered": ML_AVAILABLE
            }
        )
    except:
        return JSONResponse(
            content={
                "regime": "Normal",
                "fed_stance": "Neutral",
                "vix": 20.0,
                "recession_risk": "Low"
            }
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)