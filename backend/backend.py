#!/usr/bin/env python3
"""
FastAPI Backend - Full ML with Proper Training and Parallel Processing
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
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import hashlib
import threading
import asyncio
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="15.0.0")

# Thread pools - increased for parallel processing
executor = ThreadPoolExecutor(max_workers=10)
analysis_executor = ThreadPoolExecutor(max_workers=3)

# Job storage
analysis_jobs = {}
job_lock = threading.Lock()

# Cache
cache = {}
CACHE_DURATION = 1800  # 30 minutes

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

# ============ PROPER VIX FETCHING ============

def fetch_real_vix():
    """Fetch real VIX value"""
    try:
        vix = yf.download("^VIX", period="2d", progress=False, timeout=10)
        if not vix.empty:
            return float(vix['Close'].iloc[-1])
    except:
        pass
    
    try:
        vix_ticker = yf.Ticker("^VIX")
        hist = vix_ticker.history(period="5d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
    except:
        pass
    
    return 20.0

def get_market_data():
    """Get comprehensive market data"""
    global market_data_cache, market_data_timestamp
    
    if market_data_cache and (time.time() - market_data_timestamp) < 900:
        return market_data_cache
    
    market_data = {
        'vix': fetch_real_vix(),
        'spy_price': 500,
        'spy_trend': 1,
        'treasury_10y': 4.3,
        'dollar_index': 105
    }
    
    try:
        spy_data = yf.download("SPY", period="1mo", progress=False, timeout=10)
        if not spy_data.empty:
            market_data['spy_price'] = float(spy_data['Close'].iloc[-1])
            month_return = (spy_data['Close'].iloc[-1] - spy_data['Close'].iloc[0]) / spy_data['Close'].iloc[0]
            market_data['spy_trend'] = 1 + month_return
    except:
        pass
    
    market_data_cache = market_data
    market_data_timestamp = time.time()
    logger.info(f"Market data updated - VIX: {market_data['vix']:.2f}")
    
    return market_data

# ============ DATA LOADING ============

def load_csv_files():
    """Load CSV files"""
    data_path = os.environ.get('DATA_PATH', '/app/data')
    if not os.path.exists(data_path):
        data_path = 'data'
    
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
        except Exception as e:
            logger.error(f"Error loading {csv_file}: {e}")
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Total stocks loaded: {len(combined)}")
        return combined
    return pd.DataFrame()

stocks_data = load_csv_files()

# Import ML models
ML_AVAILABLE = False
ml_model = None
market_analyzer = None
valuator = None

try:
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
    logger.info("✅ ML Models loaded successfully!")
    
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    ML_AVAILABLE = False

# ============ IMPROVED STOCK FETCHING ============

def fetch_stock_data_batch(symbols, max_workers=5):
    """Fetch multiple stocks in parallel with proper data"""
    results = {}
    
    def fetch_single(symbol):
        try:
            ticker = yf.Ticker(symbol)
            
            # Get all the data we need
            info = ticker.info
            hist = ticker.history(period="3mo")
            
            # Get proper financial metrics
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            if not current_price and not hist.empty:
                current_price = float(hist['Close'].iloc[-1])
            
            # Calculate technical indicators
            technicals = {}
            if not hist.empty and len(hist) > 20:
                # RSI
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                technicals['rsi'] = float(100 - (100 / (1 + rs)).iloc[-1])
                
                # Momentum
                technicals['momentum_20d'] = float((hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1))
                if len(hist) > 60:
                    technicals['momentum_60d'] = float((hist['Close'].iloc[-1] / hist['Close'].iloc[-60] - 1))
                else:
                    technicals['momentum_60d'] = technicals['momentum_20d']
                
                # Volatility
                technicals['volatility'] = float(hist['Close'].pct_change().std() * np.sqrt(252))
            else:
                technicals = {'rsi': 50, 'momentum_20d': 0, 'momentum_60d': 0, 'volatility': 0.3}
            
            return symbol, {
                'info': info,
                'history': hist.to_dict() if not hist.empty else {},
                'technicals': technicals,
                'current_price': current_price
            }
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return symbol, None
    
    # Fetch in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_single, symbol): symbol for symbol in symbols}
        
        for future in as_completed(futures):
            try:
                symbol, data = future.result(timeout=15)
                if data:
                    results[symbol] = data
                    logger.info(f"✓ Fetched {symbol}")
            except TimeoutError:
                symbol = futures[future]
                logger.warning(f"✗ Timeout fetching {symbol}")
            except Exception as e:
                symbol = futures[future]
                logger.error(f"✗ Error fetching {symbol}: {e}")
    
    return results

# ============ COMPREHENSIVE ANALYSIS FUNCTION ============

def analyze_stock_comprehensive(symbol, stock_data, ml_model, market_data, sector_data):
    """Full comprehensive analysis with proper ML"""
    try:
        info = stock_data.get('info', {})
        technicals = stock_data.get('technicals', {})
        current_price = stock_data.get('current_price', 0)
        
        if not current_price or current_price <= 0:
            return None
        
        # Get ALL financial metrics properly
        metrics = {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'current_price': current_price,
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE') or info.get('forwardPE') or 0,
            'peg_ratio': info.get('pegRatio') or 0,
            'profit_margin': info.get('profitMargins') or 0,
            'revenue_growth': info.get('revenueGrowth') or info.get('quarterlyRevenueGrowth') or 0,
            'debt_to_equity': info.get('debtToEquity') or 0,
            'roe': info.get('returnOnEquity') or 0,
            'price_to_book': info.get('priceToBook') or 0,
            'forward_pe': info.get('forwardPE') or 0,
            'ev_to_revenue': info.get('enterpriseToRevenue') or 0,
            'ev_to_ebitda': info.get('enterpriseToEbitda') or 0,
            'beta': info.get('beta') or 1,
            'dividend_yield': info.get('dividendYield') or 0,
            'payout_ratio': info.get('payoutRatio') or 0,
            'current_ratio': info.get('currentRatio') or 0,
            'quick_ratio': info.get('quickRatio') or 0,
            'gross_margins': info.get('grossMargins') or 0,
            'operating_margins': info.get('operatingMargins') or 0,
            'ebitda_margins': info.get('ebitdaMargins') or 0,
            'target_mean_price': info.get('targetMeanPrice') or current_price,
            'target_high_price': info.get('targetHighPrice') or current_price,
            'target_low_price': info.get('targetLowPrice') or current_price,
            'analyst_count': info.get('numberOfAnalystOpinions') or 0,
            'recommendation_mean': info.get('recommendationMean') or 3
        }
        
        # Calculate quality score based on multiple factors
        quality_score = 0
        reasons = []
        
        # P/E Analysis
        if 0 < metrics['pe_ratio'] < 20:
            quality_score += 0.15
            reasons.append(f"Attractive P/E ratio: {metrics['pe_ratio']:.1f}")
        elif 20 <= metrics['pe_ratio'] < 30:
            quality_score += 0.08
            reasons.append(f"Reasonable P/E ratio: {metrics['pe_ratio']:.1f}")
        
        # PEG Analysis
        if 0 < metrics['peg_ratio'] < 1:
            quality_score += 0.15
            reasons.append(f"Excellent PEG ratio: {metrics['peg_ratio']:.2f}")
        elif 1 <= metrics['peg_ratio'] < 1.5:
            quality_score += 0.08
            reasons.append(f"Good PEG ratio: {metrics['peg_ratio']:.2f}")
        
        # ROE Analysis
        if metrics['roe'] > 0.2:
            quality_score += 0.15
            reasons.append(f"Strong ROE: {metrics['roe']*100:.1f}%")
        elif metrics['roe'] > 0.15:
            quality_score += 0.08
            reasons.append(f"Good ROE: {metrics['roe']*100:.1f}%")
        
        # Growth Analysis
        if metrics['revenue_growth'] > 0.15:
            quality_score += 0.15
            reasons.append(f"Strong growth: {metrics['revenue_growth']*100:.1f}%")
        elif metrics['revenue_growth'] > 0.08:
            quality_score += 0.08
            reasons.append(f"Solid growth: {metrics['revenue_growth']*100:.1f}%")
        
        # Profitability
        if metrics['profit_margin'] > 0.2:
            quality_score += 0.1
            reasons.append(f"High profit margins: {metrics['profit_margin']*100:.1f}%")
        
        # Analyst Rating
        if metrics['recommendation_mean'] < 2:
            quality_score += 0.1
            reasons.append("Strong buy rating from analysts")
        elif metrics['recommendation_mean'] < 2.5:
            quality_score += 0.05
            reasons.append("Buy rating from analysts")
        
        # Debt Analysis
        if 0 <= metrics['debt_to_equity'] < 0.5:
            quality_score += 0.1
            reasons.append(f"Low debt: D/E {metrics['debt_to_equity']:.2f}")
        
        # Technical Analysis
        if technicals.get('rsi', 50) > 50 and technicals.get('rsi', 50) < 70:
            quality_score += 0.05
            reasons.append(f"Good technical momentum: RSI {technicals.get('rsi', 50):.0f}")
        
        # ML Prediction - Train model specifically for this stock's sector
        ml_prediction = 0
        ml_confidence = 0
        
        if ML_AVAILABLE and ml_model:
            try:
                # Prepare features with actual market data
                features = {
                    'recent_returns': technicals.get('momentum_20d', 0),
                    'volatility': technicals.get('volatility', 0.3),
                    'pe_ratio': metrics['pe_ratio'] if metrics['pe_ratio'] > 0 else 20,
                    'market_cap': metrics['market_cap'],
                    'peg_ratio': metrics['peg_ratio'] if metrics['peg_ratio'] > 0 else 1.5,
                    'profit_margin': metrics['profit_margin'],
                    'revenue_growth': metrics['revenue_growth'],
                    'debt_to_equity': metrics['debt_to_equity'],
                    'roe': metrics['roe'],
                    'price_to_book': metrics['price_to_book'] if metrics['price_to_book'] > 0 else 1,
                    'rsi': technicals.get('rsi', 50),
                    'vix': market_data['vix'],
                    'treasury_10y': market_data['treasury_10y'],
                    'dollar_index': market_data['dollar_index'],
                    'spy_trend': market_data['spy_trend']
                }
                
                # Train on sector-specific data if available
                if sector_data and len(sector_data) > 5:
                    try:
                        ml_model.selected_stocks = list(sector_data.keys())
                        training_data = ml_model.prepare_training_data(list(sector_data.keys())[:10])
                        if not training_data.empty:
                            ml_model.train_prediction_model(training_data)
                    except:
                        pass
                
                # Get prediction
                ml_prediction = ml_model.calculate_stock_predictions(symbol, features)
                
                # Adjust based on market conditions
                if market_data['vix'] > 30:
                    ml_prediction *= 0.8  # Reduce confidence in high volatility
                elif market_data['vix'] < 15:
                    ml_prediction *= 1.1  # Increase confidence in low volatility
                
                # Set confidence based on prediction strength
                ml_confidence = min(abs(ml_prediction) * 100, 100)
                
                if ml_prediction > 0.1:
                    reasons.append(f"ML predicts {ml_prediction*100:.1f}% return")
                
            except Exception as e:
                logger.error(f"ML prediction error for {symbol}: {e}")
                ml_prediction = 0
        
        # Sentiment Analysis
        sentiment_score = 0
        if ML_AVAILABLE and ml_model and hasattr(ml_model, 'simple_sentiment'):
            try:
                sentiment_text = f"{symbol} {metrics['company_name']} stock investment outlook"
                sentiment_result = ml_model.simple_sentiment(sentiment_text)
                if sentiment_result:
                    sent = sentiment_result[0]
                    if sent['label'] == 'positive':
                        sentiment_score = sent['score']
                    elif sent['label'] == 'negative':
                        sentiment_score = -sent['score']
            except:
                sentiment_score = 0
        
        # Enhanced Valuation
        if valuator and ML_AVAILABLE:
            try:
                val_result = valuator.calculate_comprehensive_valuation(
                    symbol, 
                    ml_prediction, 
                    sentiment_score,
                    market_data['vix'] / 20  # Volatility adjustment
                )
                if val_result and 'target_price' in val_result:
                    metrics['target_mean_price'] = val_result['target_price']
            except:
                pass
        
        # Calculate upside potential
        upside = ((metrics['target_mean_price'] / current_price) - 1) * 100 if current_price > 0 else 0
        
        # Combined score with emphasis on upside and quality
        combined_score = (
            upside * 0.4 +  # 40% weight on upside
            quality_score * 100 * 0.3 +  # 30% weight on quality
            ml_prediction * 100 * 0.2 +  # 20% weight on ML
            (100 - technicals.get('rsi', 50)) * 0.1  # 10% weight on oversold conditions
        )
        
        return {
            'symbol': symbol,
            'company_name': metrics['company_name'],
            'current_price': current_price,
            'target_price': metrics['target_mean_price'],
            'upside': upside,
            'quality_score': quality_score,
            'ml_prediction': ml_prediction,
            'ml_confidence': ml_confidence,
            'sentiment_score': sentiment_score,
            'combined_score': combined_score,
            'metrics': metrics,
            'technicals': technicals,
            'reasons': reasons,
            'analyst_count': metrics['analyst_count'],
            'recommendation': metrics['recommendation_mean']
        }
        
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}")
        return None

# ============ BACKGROUND ANALYSIS WITH PARALLEL PROCESSING ============

def run_analysis_background(job_id, request_data):
    """Run analysis in background with proper parallel processing"""
    try:
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'processing',
                'progress': 'Initializing analysis...',
                'started': time.time()
            }
        
        analysis_type = request_data['analysis_type']
        target = request_data['target']
        market_cap_size = request_data.get('market_cap_size', 'all')
        
        logger.info(f"Job {job_id}: Analyzing {target} ({market_cap_size})")
        
        # Get market data first
        market_data = get_market_data()
        
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
            analysis_jobs[job_id]['progress'] = f'Found {len(filtered_stocks)} stocks, filtering by market cap...'
        
        # Get symbols and filter by market cap
        all_symbols = filtered_stocks['Symbol'].tolist()
        
        # First, quickly get market caps to filter
        market_caps = {}
        cap_range = MARKET_CAP_RANGES.get(market_cap_size, MARKET_CAP_RANGES['all'])
        
        # Batch fetch basic info for market cap filtering
        with ThreadPoolExecutor(max_workers=10) as pool:
            def get_market_cap(symbol):
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.fast_info
                    cap = info.market_cap if hasattr(info, 'market_cap') else ticker.info.get('marketCap', 0)
                    return symbol, cap
                except:
                    return symbol, 0
            
            futures = {pool.submit(get_market_cap, symbol): symbol for symbol in all_symbols[:100]}
            
            for future in as_completed(futures):
                try:
                    symbol, cap = future.result(timeout=5)
                    if cap > 0 and cap_range['min'] <= cap < cap_range['max']:
                        market_caps[symbol] = cap
                except:
                    continue
        
        # Sort by market cap and select top candidates
        sorted_symbols = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
        analysis_symbols = [s[0] for s in sorted_symbols[:20]]  # Analyze top 20 to ensure we get good ones
        
        if len(analysis_symbols) < 3:
            # If not enough in market cap range, take largest available
            analysis_symbols = [s[0] for s in sorted_symbols[:10]]
        
        # Update progress
        with job_lock:
            analysis_jobs[job_id]['progress'] = f'Fetching detailed data for {len(analysis_symbols)} stocks...'
        
        # Fetch all stock data in parallel
        stock_data_dict = fetch_stock_data_batch(analysis_symbols, max_workers=8)
        
        if not stock_data_dict:
            with job_lock:
                analysis_jobs[job_id] = {
                    'status': 'error',
                    'error': 'Failed to fetch stock data'
                }
            return
        
        # Update progress
        with job_lock:
            analysis_jobs[job_id]['progress'] = f'Analyzing {len(stock_data_dict)} stocks with ML models...'
        
        # Analyze each stock
        results = []
        for i, (symbol, data) in enumerate(stock_data_dict.items()):
            if data:
                with job_lock:
                    analysis_jobs[job_id]['progress'] = f'Analyzing {symbol} ({i+1}/{len(stock_data_dict)})'
                
                analysis = analyze_stock_comprehensive(
                    symbol, data, ml_model, market_data, stock_data_dict
                )
                
                if analysis and analysis['upside'] > 5:  # Only consider stocks with >5% upside
                    results.append(analysis)
                    logger.info(f"✓ {symbol}: Upside={analysis['upside']:.1f}%, Score={analysis['combined_score']:.1f}")
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Get top 3 (or best available)
        top_picks = results[:3]
        
        # If we have less than 3, try to add more from lower upside threshold
        if len(top_picks) < 3:
            for analysis in results[3:]:
                if analysis['upside'] > 3:  # Lower threshold for filling
                    top_picks.append(analysis)
                    if len(top_picks) >= 3:
                        break
        
        # Format results
        formatted = []
        for stock in top_picks:
            formatted.append({
                "symbol": stock['symbol'],
                "company_name": stock['company_name'],
                "market_cap": f"${market_caps.get(stock['symbol'], 0)/1e9:.2f}B" if market_caps.get(stock['symbol'], 0) > 1e9 else f"${market_caps.get(stock['symbol'], 0)/1e6:.0f}M",
                "metrics": {
                    "current_price": round(stock['current_price'], 2),
                    "target_price": round(stock['target_price'], 2),
                    "upside_potential": round(stock['upside'], 1),
                    "confidence_score": int(stock['quality_score'] * 100),
                    "sentiment_score": round(stock['sentiment_score'], 2),
                    "ml_score": round(stock['ml_prediction'], 4)
                },
                "analysis_details": {
                    "fundamentals": {
                        "pe_ratio": round(stock['metrics']['pe_ratio'], 2) if stock['metrics']['pe_ratio'] > 0 else "N/A",
                        "peg_ratio": round(stock['metrics']['peg_ratio'], 2) if stock['metrics']['peg_ratio'] > 0 else "N/A",
                        "roe": round(stock['metrics']['roe'], 3),
                        "profit_margin": round(stock['metrics']['profit_margin'], 3),
                        "revenue_growth": round(stock['metrics']['revenue_growth'], 3),
                        "debt_to_equity": round(stock['metrics']['debt_to_equity'], 2),
                        "price_to_book": round(stock['metrics']['price_to_book'], 2) if stock['metrics']['price_to_book'] > 0 else "N/A",
                        "beta": round(stock['metrics']['beta'], 2)
                    },
                    "technicals": stock['technicals'],
                    "ml_prediction": stock['ml_prediction'],
                    "ml_confidence": stock['ml_confidence'],
                    "quality_score": stock['quality_score']
                },
                "investment_thesis": stock['reasons'],
                "analyst_rating": stock['recommendation'],
                "analyst_count": stock['analyst_count']
            })
        
        # Store results
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'completed',
                'results': {
                    'top_stocks': formatted,
                    'market_conditions': {
                        'regime': 'Elevated Volatility' if market_data['vix'] > 20 else 'Normal',
                        'vix': market_data['vix']
                    },
                    'total_analyzed': len(results),
                    'total_qualified': len([r for r in results if r['upside'] > 5])
                },
                'completed': time.time()
            }
        
        elapsed = time.time() - analysis_jobs[job_id]['started']
        logger.info(f"Job {job_id} completed in {elapsed:.1f} seconds")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
            "message": "Analysis started"
        }
    )

@app.get("/api/analysis/{job_id}")
async def get_analysis_status(job_id: str):
    """Get analysis job status"""
    with job_lock:
        if job_id not in analysis_jobs:
            return JSONResponse(
                content={"status": "not_found"},
                status_code=404
            )
        
        job = analysis_jobs[job_id]
        
        if job['status'] == 'completed':
            result = {
                "status": "completed",
                "results": job['results'],
                "ml_powered": ML_AVAILABLE
            }
            del analysis_jobs[job_id]
            return JSONResponse(content=result)
        
        elif job['status'] == 'error':
            result = {
                "status": "error",
                "error": job.get('error', 'Unknown error')
            }
            del analysis_jobs[job_id]
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
                "vix": vix,
                "recession_risk": "Low" if vix < 25 else "Elevated",
                "ml_powered": ML_AVAILABLE
            }
        )
    except:
        return JSONResponse(
            content={
                "regime": "Unknown",
                "vix": 20.0
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
            "version": "15.0.0",
            "ml_enabled": ML_AVAILABLE,
            "stocks_loaded": len(stocks_data)
        }
    )

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting up...")
    get_market_data()
    logger.info(f"Ready - VIX: {market_data_cache['vix'] if market_data_cache else 'N/A'}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)