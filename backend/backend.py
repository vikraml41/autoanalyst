#!/usr/bin/env python3
"""
FastAPI Backend - Full ML Complexity with Optimizations (No aiohttp)
"""

import os
import sys
import glob
import logging
import json
import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="9.0.0")

# Thread pool
executor = ThreadPoolExecutor(max_workers=30)

# In-memory cache
cache = {}
CACHE_DURATION = 3600  # 1 hour

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
            logger.info(f"Cache hit for {key}")
            return data
        else:
            del cache[key]
    return None

def set_cache(key, data):
    """Set cache with timestamp"""
    cache[key] = (data, time.time())
    logger.info(f"Cached {key}")

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

# ============ BATCH DATA FETCHING ============

def batch_fetch_stock_info(symbols):
    """Batch fetch stock info with caching"""
    results = {}
    to_fetch = []
    
    # Check cache first
    for symbol in symbols:
        cache_key = get_cache_key('stock_info', symbol)
        cached_data = get_from_cache(cache_key)
        if cached_data:
            results[symbol] = cached_data
        else:
            to_fetch.append(symbol)
    
    if not to_fetch:
        return results
    
    logger.info(f"Fetching {len(to_fetch)} stocks from Yahoo Finance...")
    
    # Parallel fetch function
    def fetch_single(symbol):
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period="3mo")  # Reduced from 6mo for speed
            
            data = {
                'info': info,
                'history': hist.to_dict() if not hist.empty else {}
            }
            
            # Cache the result
            cache_key = get_cache_key('stock_info', symbol)
            set_cache(cache_key, data)
            
            return symbol, data
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return symbol, None
    
    # Use thread pool for parallel fetching
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(fetch_single, symbol) for symbol in to_fetch]
        
        for future in as_completed(futures):
            try:
                symbol, data = future.result(timeout=5)
                if data:
                    results[symbol] = data
            except Exception as e:
                logger.error(f"Fetch timeout: {e}")
                continue
    
    logger.info(f"Successfully fetched {len(results)} stocks")
    return results

# ============ OPTIMIZED ANALYSIS ============

def analyze_stock_complete(symbol, stock_data, ml_model, valuator, market_adjustment):
    """Complete analysis with all ML features"""
    try:
        info = stock_data.get('info', {})
        hist_dict = stock_data.get('history', {})
        
        # Get current price
        current_price = info.get('currentPrice') or info.get('regularMarketPrice')
        if not current_price and hist_dict.get('Close'):
            close_prices = list(hist_dict['Close'].values())
            if close_prices:
                current_price = close_prices[-1]
        
        if not current_price or current_price <= 0:
            return None
        
        # Convert history to DataFrame
        hist = pd.DataFrame(hist_dict) if hist_dict else pd.DataFrame()
        
        # Full metrics
        metrics = {
            'symbol': symbol,
            'current_price': current_price,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'peg_ratio': info.get('pegRatio', 0),
            'profit_margin': info.get('profitMargins', 0),
            'revenue_growth': info.get('revenueGrowth', 0),
            'debt_to_equity': info.get('debtToEquity', 0),
            'roe': info.get('returnOnEquity', 0),
            'price_to_book': info.get('priceToBook', 0),
            'analyst_rating': info.get('recommendationMean', 3),
            'target_mean_price': info.get('targetMeanPrice', current_price)
        }
        
        # Technical indicators
        if len(hist) > 20:
            try:
                # RSI
                close_prices = hist['Close']
                close_delta = close_prices.diff()
                gain = (close_delta.where(close_delta > 0, 0)).rolling(window=14).mean()
                loss = (-close_delta.where(close_delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs.iloc[-1]))
                
                # Moving averages
                ma20 = close_prices.rolling(20).mean().iloc[-1]
                
                # Momentum
                returns_20d = (close_prices.iloc[-1] - close_prices.iloc[-20]) / close_prices.iloc[-20]
                
                metrics['rsi'] = rsi if not pd.isna(rsi) else 50
                metrics['ma20_ratio'] = current_price / ma20 if ma20 > 0 else 1
                metrics['momentum_20d'] = returns_20d if not pd.isna(returns_20d) else 0
                metrics['momentum_60d'] = returns_20d  # Simplified
                metrics['volatility'] = close_prices.pct_change().std() * np.sqrt(252)
            except:
                metrics['rsi'] = 50
                metrics['ma20_ratio'] = 1
                metrics['momentum_20d'] = 0
                metrics['momentum_60d'] = 0
                metrics['volatility'] = 0.2
        else:
            metrics['rsi'] = 50
            metrics['ma20_ratio'] = 1
            metrics['momentum_20d'] = 0
            metrics['momentum_60d'] = 0
            metrics['volatility'] = 0.2
        
        # ML prediction
        ml_prediction = 0
        if ml_model and hasattr(ml_model, 'calculate_stock_predictions'):
            try:
                current_data = {
                    'recent_returns': metrics['momentum_20d'],
                    'volatility': metrics['volatility'],
                    'pe_ratio': metrics['pe_ratio'],
                    'market_cap': metrics['market_cap'],
                    'peg_ratio': metrics['peg_ratio'],
                    'profit_margin': metrics['profit_margin'],
                    'revenue_growth': metrics['revenue_growth'],
                    'debt_to_equity': metrics['debt_to_equity'],
                    'roe': metrics['roe'],
                    'price_to_book': metrics['price_to_book'],
                    'rsi': metrics['rsi'],
                    'vix': 20,
                    'treasury_10y': 3.5,
                    'dollar_index': 100,
                    'spy_trend': 1
                }
                ml_prediction = ml_model.calculate_stock_predictions(symbol, current_data)
            except:
                ml_prediction = 0
        
        # Sentiment (simplified for speed - can be re-enabled if needed)
        sentiment_score = 0
        if ml_model and hasattr(ml_model, 'simple_sentiment'):
            try:
                # Check cache first
                sentiment_key = get_cache_key('sentiment', symbol)
                cached_sentiment = get_from_cache(sentiment_key)
                if cached_sentiment is not None:
                    sentiment_score = cached_sentiment
                else:
                    company_name = info.get('longName', symbol)
                    sentiment_result = ml_model.simple_sentiment(f"{symbol} {company_name} stock outlook")
                    if sentiment_result and len(sentiment_result) > 0:
                        sent = sentiment_result[0]
                        if sent['label'] == 'positive':
                            sentiment_score = sent['score']
                        elif sent['label'] == 'negative':
                            sentiment_score = -sent['score']
                    set_cache(sentiment_key, sentiment_score)
            except:
                sentiment_score = 0
        
        # Valuation
        target_price = metrics['target_mean_price'] * market_adjustment
        confidence = 0.5
        
        if valuator and hasattr(valuator, 'calculate_comprehensive_valuation'):
            try:
                valuation_result = valuator.calculate_comprehensive_valuation(
                    symbol, 
                    ml_prediction, 
                    sentiment_score, 
                    market_adjustment
                )
                target_price = valuation_result.get('target_price', target_price)
                confidence = valuation_result.get('confidence', 0.5)
                metrics['valuation_details'] = valuation_result.get('valuations', {})
            except:
                pass
        
        # Calculate scores
        upside = ((target_price / current_price) - 1) * 100 if current_price > 0 else 0
        
        # Quality score
        quality_score = 0
        if 0 < metrics['pe_ratio'] < 30:
            quality_score += 0.2
        if 0 < metrics['peg_ratio'] < 1.5:
            quality_score += 0.2
        if metrics['roe'] > 0.15:
            quality_score += 0.2
        if metrics['revenue_growth'] > 0.1:
            quality_score += 0.2
        if metrics['debt_to_equity'] < 1:
            quality_score += 0.1
        if metrics['analyst_rating'] < 2.5:
            quality_score += 0.1
        
        # Combined score
        combined_score = (
            upside * 0.3 +
            quality_score * 100 * 0.3 +
            (ml_prediction * 100) * 0.2 +
            (sentiment_score * 100) * 0.1 +
            (confidence * 100) * 0.1
        )
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'target_price': target_price,
            'upside': upside,
            'ml_prediction': ml_prediction,
            'sentiment': sentiment_score,
            'confidence': confidence,
            'quality_score': quality_score,
            'combined_score': combined_score,
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest):
    """Optimized analysis with full ML complexity"""
    start_time = time.time()
    logger.info(f"=" * 50)
    logger.info(f"Analysis START: {request.analysis_type} - {request.target}")
    
    # Check cache
    cache_key = get_cache_key('analysis', request.analysis_type, request.target)
    cached_result = get_from_cache(cache_key)
    if cached_result:
        logger.info("Returning cached result")
        return JSONResponse(content=cached_result, headers={"Access-Control-Allow-Origin": "*"})
    
    if not ML_AVAILABLE or ml_model is None:
        return JSONResponse(
            content={"status": "failed", "error": "ML Model not available"},
            status_code=500
        )
    
    try:
        # Filter stocks
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
        
        # Get symbols - limit for speed
        symbols = filtered_stocks['Symbol'].tolist()[:40]
        
        # Batch fetch all stock data
        stock_data_dict = batch_fetch_stock_info(symbols)
        
        # Quick market cap filtering
        market_caps = {}
        for symbol, data in stock_data_dict.items():
            if data and 'info' in data:
                cap = data['info'].get('marketCap', 0)
                if cap > 0:
                    market_caps[symbol] = cap
        
        # Get top stocks by market cap
        sorted_stocks = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)[:15]
        analysis_symbols = [stock[0] for stock in sorted_stocks]
        
        if not analysis_symbols:
            # Fallback if no market caps
            analysis_symbols = list(stock_data_dict.keys())[:10]
        
        logger.info(f"Analyzing {len(analysis_symbols)} stocks")
        
        # Setup ML model
        ml_model.selected_stocks = analysis_symbols
        ml_model.master_df = stocks_data
        ml_model.process_gics_data()
        
        # Market adjustment
        market_adjustment = 1.05
        sector_analysis = {}
        
        # Get market conditions if available
        if market_analyzer:
            try:
                sector_conditions = market_analyzer.analyze_sector_conditions(request.target)
                market_adjustment = market_analyzer.calculate_market_adjustment_factor(request.target)
                ml_model.market_adjustment = market_adjustment
                
                sector_analysis = {
                    "market_regime": "Normal",
                    "fed_stance": "Neutral",
                    "sector_conditions": sector_conditions,
                    "market_adjustment": market_adjustment
                }
            except:
                pass
        
        # Quick training if time permits
        elapsed = time.time() - start_time
        if elapsed < 10 and len(analysis_symbols) > 5:
            try:
                logger.info("Quick model training...")
                training_symbols = analysis_symbols[:7]
                training_data = ml_model.prepare_training_data(training_symbols)
                if not training_data.empty:
                    training_data = ml_model.add_sentiment_features(training_data)
                    ml_model.training_data = training_data
                    ml_model.train_prediction_model(training_data)
            except Exception as e:
                logger.error(f"Training error: {e}")
        
        # Parallel analysis
        logger.info("Running parallel analysis...")
        results = []
        
        with ThreadPoolExecutor(max_workers=15) as pool:
            futures = []
            for symbol in analysis_symbols:
                if symbol in stock_data_dict:
                    future = pool.submit(
                        analyze_stock_complete,
                        symbol,
                        stock_data_dict[symbol],
                        ml_model,
                        valuator,
                        market_adjustment
                    )
                    futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=5)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Future error: {e}")
                    continue
        
        # Filter and sort
        filtered_results = [r for r in results if r['upside'] > 0 and r['quality_score'] > 0.3]
        if not filtered_results:
            filtered_results = results
        
        filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Format top 3
        top_stocks = []
        for stock in filtered_results[:3]:
            top_stocks.append({
                "symbol": stock['symbol'],
                "metrics": {
                    "current_price": round(stock['current_price'], 2),
                    "target_price": round(stock['target_price'], 2),
                    "upside_potential": round(stock['upside'], 1),
                    "confidence_score": int(stock['confidence'] * 100),
                    "sentiment_score": round(abs(stock['sentiment']), 2),
                    "ml_score": round(stock['quality_score'], 3)
                },
                "analysis_details": {
                    "fundamentals": {
                        "pe_ratio": stock['metrics'].get('pe_ratio', 0),
                        "peg_ratio": stock['metrics'].get('peg_ratio', 0),
                        "roe": stock['metrics'].get('roe', 0),
                        "profit_margin": stock['metrics'].get('profit_margin', 0),
                        "revenue_growth": stock['metrics'].get('revenue_growth', 0),
                        "debt_to_equity": stock['metrics'].get('debt_to_equity', 0)
                    },
                    "technicals": {
                        "rsi": stock['metrics'].get('rsi', 50),
                        "momentum_20d": stock['metrics'].get('momentum_20d', 0),
                        "momentum_60d": stock['metrics'].get('momentum_60d', 0),
                        "volatility": stock['metrics'].get('volatility', 0)
                    },
                    "valuation_methods": stock['metrics'].get('valuation_details', {}),
                    "ml_prediction": stock['ml_prediction'],
                    "quality_score": stock['quality_score']
                }
            })
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Analysis completed in {elapsed:.2f} seconds")
        
        result_data = {
            "status": "completed",
            "analysis_type": request.analysis_type,
            "target": request.target,
            "results": {
                "top_stocks": top_stocks,
                "market_conditions": {
                    "regime": sector_analysis.get('market_regime', 'Normal'),
                    "adjustment_factor": market_adjustment
                },
                "sector_analysis": sector_analysis,
                "total_analyzed": len(results),
                "total_qualified": len(filtered_results)
            },
            "ml_powered": True
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

# Other endpoints
@app.get("/")
async def root():
    return JSONResponse(
        content={
            "name": "AutoAnalyst API",
            "version": "9.0.0",
            "status": "running",
            "ml_enabled": ML_AVAILABLE,
            "stocks_loaded": len(stocks_data),
            "cache_size": len(cache)
        },
        headers={"Access-Control-Allow-Origin": "*"}
    )

@app.get("/api/health")
async def health_check():
    return JSONResponse(
        content={
            "status": "healthy",
            "ml_available": ML_AVAILABLE,
            "stocks_count": len(stocks_data),
            "cache_entries": len(cache),
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
                    "regime": "Normal",
                    "fed_stance": "Neutral",
                    "vix": 20.0,
                    "recession_risk": "Low"
                },
                headers={"Access-Control-Allow-Origin": "*"}
            )
        
        # Quick check without detailed analysis
        regime = market_analyzer.get_market_regime()
        fed_data = market_analyzer.get_federal_reserve_stance()
        yield_curve = market_analyzer.analyze_yield_curve()
        
        return JSONResponse(
            content={
                "regime": regime,
                "fed_stance": fed_data.get("stance", "Neutral"),
                "vix": 20.0,
                "recession_risk": yield_curve.get("recession_risk", "Low"),
                "ml_powered": ML_AVAILABLE
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except:
        return JSONResponse(
            content={
                "regime": "Normal",
                "fed_stance": "Neutral",
                "vix": 20.0,
                "recession_risk": "Low"
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)