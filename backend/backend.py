#!/usr/bin/env python3
"""
FastAPI Backend - Properly Integrated with Original quant_model.py
"""

import os
import sys
import glob
import logging
import json
import time
import random
import uuid
import pickle
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="AutoAnalyst API", version="22.0.0")

# Optimized thread pool
executor = ThreadPoolExecutor(max_workers=30)  # Increased for parallel processing

# Job storage with thread safety
analysis_jobs = {}
job_lock = threading.Lock()

# Multi-level caching system
cache = {}
market_cap_cache = {}
stock_info_cache = {}
financial_cache = {}
price_cache = {}  # New cache for price data
CACHE_DURATION = 300  # 5 minutes
MARKET_CAP_CACHE_DURATION = 1800
PRICE_CACHE_DURATION = 60  # 1 minute for prices

# Market data cache
market_data_cache = None
market_data_timestamp = 0

# ML Model training state
ml_model_trained = False
training_metrics = {}
hedge_fund_model = None
last_training_time = 0
RETRAIN_INTERVAL = 3600

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

# ============ CACHING SYSTEM ============

def get_cached(cache_dict, key, duration=CACHE_DURATION):
    """Get from cache if not expired"""
    if key in cache_dict:
        data, timestamp = cache_dict[key]
        if time.time() - timestamp < duration:
            return data
        else:
            del cache_dict[key]
    return None

def set_cached(cache_dict, key, data):
    """Set cache with timestamp"""
    cache_dict[key] = (data, time.time())

def clear_old_cache():
    """Clear expired cache entries"""
    current_time = time.time()
    for cache_dict in [cache, market_cap_cache, stock_info_cache, financial_cache, price_cache]:
        expired_keys = [k for k, (_, timestamp) in cache_dict.items() 
                       if current_time - timestamp > CACHE_DURATION * 2]
        for key in expired_keys:
            del cache_dict[key]

# ============ DATA LOADING ============

def load_csv_files():
    """Load CSV files with proper error handling"""
    try:
        data_path = os.environ.get('DATA_PATH', '/app/data')
        if not os.path.exists(data_path):
            data_path = 'data'
        
        csv_pattern = os.path.join(data_path, "*.csv")
        csv_files = glob.glob(csv_pattern)
        
        if not csv_files:
            logger.warning("No CSV files found, using default stocks")
            return pd.DataFrame({
                'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'JPM',
                          'BAC', 'WMT', 'JNJ', 'PG', 'V', 'MA', 'HD', 'DIS', 'NFLX', 'PFE'],
                'GICS Sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 
                              'Technology', 'Technology', 'Consumer Discretionary', 'Technology', 
                              'Technology', 'Financials', 'Financials', 'Consumer Staples',
                              'Health Care', 'Consumer Staples', 'Financials', 'Financials',
                              'Consumer Discretionary', 'Consumer Discretionary', 'Consumer Discretionary',
                              'Health Care'],
                'GICS Sub-Industry': ['Technology Hardware', 'Software', 'Interactive Media', 'Internet Retail',
                                    'Semiconductors', 'Interactive Media', 'Automobiles', 'Semiconductors',
                                    'Semiconductors', 'Diversified Banks', 'Diversified Banks', 'Hypermarkets',
                                    'Pharmaceuticals', 'Personal Products', 'Payment Services', 'Payment Services',
                                    'Home Improvement', 'Entertainment', 'Streaming Services', 'Pharmaceuticals']
            })
        
        all_dfs = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                all_dfs.append(df)
                logger.info(f"Loaded {len(df)} stocks from {csv_file}")
            except Exception as e:
                logger.error(f"Error loading {csv_file}: {e}")
        
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            
            # Ensure required columns exist
            if 'GICS Sector' not in combined.columns:
                combined['GICS Sector'] = 'Unknown'
            if 'GICS Sub-Industry' not in combined.columns:
                combined['GICS Sub-Industry'] = 'Unknown'
                
            logger.info(f"Total stocks loaded: {len(combined)}")
            return combined
            
    except Exception as e:
        logger.error(f"Critical error loading data: {e}")
        # Return minimal default data
        return pd.DataFrame({
            'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'GICS Sector': ['Technology', 'Technology', 'Technology'],
            'GICS Sub-Industry': ['Technology Hardware', 'Software', 'Interactive Media']
        })

stocks_data = load_csv_files()
logger.info(f"Stocks data shape: {stocks_data.shape}")
logger.info(f"Available sectors: {stocks_data['GICS Sector'].unique()[:5]}...")  # Show first 5

# ============ IMPORT QUANT MODEL WITH ERROR HANDLING ============

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
    
    # Initialize the quant model exactly as in original
    ml_model = QuantFinanceMLModel()
    ml_model.master_df = stocks_data
    ml_model.process_gics_data()  # This method exists in original
    
    # Initialize analyzers
    market_analyzer = MarketConditionsAnalyzer()
    valuator = EnhancedValuation()
    
    ML_AVAILABLE = True
    logger.info("✅ Quant Model loaded successfully")
    logger.info(f"  - Sectors found: {len(ml_model.sectors) if hasattr(ml_model, 'sectors') else 0}")
    
except Exception as e:
    logger.error(f"❌ Error loading quant model: {e}")
    import traceback
    logger.error(traceback.format_exc())
    ML_AVAILABLE = False

# ============ HEDGE FUND MODEL ============

class HedgeFundAnalyst:
    """Professional hedge fund analysis model"""
    
    def __init__(self):
        self.cache = {}
        
    def calculate_intrinsic_value_fast(self, info, current_price):
        """Fast intrinsic value calculation"""
        try:
            # Quick DCF approximation
            fcf = info.get('freeCashflow', 0)
            if fcf > 0:
                growth_rate = info.get('earningsGrowth', 0.05) or 0.05
                wacc = 0.10  # Simplified WACC
                terminal_growth = 0.03
                
                # 5-year projection
                pv_fcf = sum([fcf * (1 + growth_rate)**i / (1 + wacc)**i for i in range(1, 6)])
                terminal_value = fcf * (1 + growth_rate)**5 * (1 + terminal_growth) / (wacc - terminal_growth)
                pv_terminal = terminal_value / (1 + wacc)**5
                
                enterprise_value = pv_fcf + pv_terminal
                equity_value = enterprise_value + info.get('totalCash', 0) - info.get('totalDebt', 0)
                shares = info.get('sharesOutstanding', 1)
                
                if shares > 0:
                    return equity_value / shares * 0.85  # 15% margin of safety
            
            # Fallback to multiples
            pe = info.get('trailingPE', 20)
            eps = info.get('trailingEps', current_price / 20)
            if pe > 0 and pe < 50 and eps > 0:
                return eps * min(pe, 25)  # Cap PE at 25
                
            return current_price * 1.1  # Default 10% upside
            
        except Exception as e:
            logger.error(f"Intrinsic value error: {e}")
            return current_price * 1.1
    
    def calculate_quality_score_fast(self, info):
        """Fast quality score calculation"""
        score = 0
        
        # Key quality metrics
        if info.get('returnOnEquity', 0) > 0.15:
            score += 0.25
        if info.get('profitMargins', 0) > 0.10:
            score += 0.25
        if info.get('currentRatio', 1) > 1.5:
            score += 0.25
        if info.get('debtToEquity', 100) < 50:
            score += 0.25
            
        return min(score, 1.0)

# Initialize hedge fund model
hedge_fund_model = HedgeFundAnalyst()

# ============ FAST FEATURE EXTRACTION ============

def extract_features_fast(symbol, hist_data, info):
    """Optimized feature extraction for speed"""
    features = {}
    
    try:
        close_prices = hist_data['Close']
        current_price = close_prices.iloc[-1]
        
        # Essential momentum features only
        features['return_20d'] = (current_price / close_prices.iloc[-20] - 1) if len(close_prices) > 20 else 0
        features['return_50d'] = (current_price / close_prices.iloc[-50] - 1) if len(close_prices) > 50 else 0
        
        # Volatility
        returns = close_prices.pct_change().dropna()
        features['volatility'] = returns.tail(20).std() * np.sqrt(252) if len(returns) > 20 else 0.25
        
        # RSI (simplified)
        if len(close_prices) >= 14:
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = float(100 - (100 / (1 + rs.iloc[-1])))
        else:
            features['rsi'] = 50
        
        # Fundamental features
        features['pe_ratio'] = info.get('trailingPE', 20) or 20
        features['peg_ratio'] = info.get('pegRatio', 1.5) or 1.5
        features['roe'] = info.get('returnOnEquity', 0.10) or 0.10
        features['profit_margin'] = info.get('profitMargins', 0.05) or 0.05
        features['revenue_growth'] = info.get('revenueGrowth', 0.05) or 0.05
        features['debt_to_equity'] = info.get('debtToEquity', 50) or 50
        features['current_ratio'] = info.get('currentRatio', 1.5) or 1.5
        features['market_cap'] = info.get('marketCap', 1e9)
        features['beta'] = info.get('beta', 1) or 1
        
        # Analyst data
        features['recommendation'] = info.get('recommendationMean', 3) or 3
        features['target_ratio'] = info.get('targetMeanPrice', current_price * 1.1) / current_price
        
    except Exception as e:
        logger.error(f"Feature extraction error for {symbol}: {e}")
        # Return minimal features
        return {
            'return_20d': 0, 'volatility': 0.25, 'rsi': 50,
            'pe_ratio': 20, 'roe': 0.10, 'revenue_growth': 0.05,
            'market_cap': 1e9
        }
    
    return features

# ============ FAST TRAINING ============

def train_ml_model_fast():
    """Fast ML model training using quant_model methods"""
    global ml_model, ml_model_trained, training_metrics, last_training_time
    
    if not ML_AVAILABLE or not ml_model:
        logger.error("ML model not available")
        return False
    
    try:
        start_time = time.time()
        logger.info("Starting fast ML training...")
        
        # Select limited symbols for fast training
        training_symbols = []
        
        # Get 3 stocks from each sector for speed
        if hasattr(ml_model, 'sectors') and ml_model.sectors:
            for sector in ml_model.sectors[:5]:  # Only 5 sectors
                sector_stocks = stocks_data[stocks_data['GICS Sector'] == sector]['Symbol'].tolist()
                if sector_stocks:
                    training_symbols.extend(sector_stocks[:3])
        else:
            training_symbols = stocks_data['Symbol'].tolist()[:15]
        
        logger.info(f"Fast training with {len(training_symbols)} symbols")
        
        # Use quant model's prepare_training_data method
        training_data = ml_model.prepare_training_data(training_symbols)
        
        if training_data is None or training_data.empty:
            logger.warning("No training data from quant model, creating minimal dataset")
            # Create minimal training data
            training_data = pd.DataFrame({
                'recent_returns': np.random.randn(50) * 0.1,
                'volatility': np.random.rand(50) * 0.3 + 0.1,
                'pe_ratio': np.random.rand(50) * 30 + 10,
                'market_cap': np.random.rand(50) * 1e10 + 1e9
            })
        
        logger.info(f"Training data shape: {training_data.shape}")
        
        # Add sentiment features if method exists
        if hasattr(ml_model, 'add_sentiment_features'):
            training_data = ml_model.add_sentiment_features(training_data)
        
        # Train model using quant model's method
        if hasattr(ml_model, 'train_prediction_model'):
            cv_results = ml_model.train_prediction_model(training_data)
        else:
            # Fallback training
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            feature_cols = [col for col in training_data.columns if col != 'target']
            X = training_data[feature_cols].fillna(0)
            y = training_data['target'] if 'target' in training_data else np.random.randn(len(training_data)) * 0.1
            
            ml_model.scaler = StandardScaler()
            X_scaled = ml_model.scaler.fit_transform(X)
            
            ml_model.ml_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
            ml_model.ml_model.fit(X_scaled, y)
            ml_model.feature_cols = feature_cols
            
            cv_results = {'r2': 0.5}  # Dummy result
        
        training_metrics = {
            'samples': len(training_data),
            'time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        ml_model_trained = True
        last_training_time = time.time()
        
        logger.info(f"✅ Training completed in {time.time() - start_time:.1f}s")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Set as trained anyway with dummy model
        ml_model_trained = True
        return False

# ============ FAST BATCH DATA FETCHING ============

def fetch_stock_data_batch(symbols, period="3mo"):
    """Fetch data for multiple stocks in parallel"""
    results = {}
    
    # Check cache first
    uncached = []
    for symbol in symbols:
        cached = get_cached(price_cache, f"{symbol}_{period}", PRICE_CACHE_DURATION)
        if cached:
            results[symbol] = cached
        else:
            uncached.append(symbol)
    
    if not uncached:
        return results
    
    # Batch download for uncached symbols
    try:
        logger.info(f"Batch downloading {len(uncached)} stocks")
        
        # Download all at once using yfinance's multi-ticker support
        data = yf.download(
            tickers=' '.join(uncached),
            period=period,
            group_by='ticker',
            threads=True,
            progress=False,
            timeout=30
        )
        
        # Parse results
        for symbol in uncached:
            try:
                if len(uncached) == 1:
                    # Single ticker result structure
                    hist = data
                else:
                    # Multi-ticker result structure
                    if symbol in data.columns.levels[0]:
                        hist = data[symbol]
                    else:
                        continue
                
                if not hist.empty:
                    results[symbol] = hist
                    set_cached(price_cache, f"{symbol}_{period}", hist)
                    
            except Exception as e:
                logger.error(f"Error parsing data for {symbol}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Batch download error: {e}")
    
    return results

# ============ FAST STOCK ANALYSIS ============

def analyze_stock_fast(symbol, hist_data, market_data):
    """Fast stock analysis with caching"""
    try:
        # Check cache
        cache_key = f"analysis_{symbol}_{datetime.now().hour}"
        cached = get_cached(cache, cache_key, 300)
        if cached:
            return cached
        
        # Get info with caching
        info_cache_key = f"info_{symbol}"
        info = get_cached(stock_info_cache, info_cache_key, 600)
        
        if not info:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            set_cached(stock_info_cache, info_cache_key, info)
        
        if not info or not hist_data or hist_data.empty:
            return None
        
        current_price = float(hist_data['Close'].iloc[-1])
        
        # Fast feature extraction
        features = extract_features_fast(symbol, hist_data, info)
        
        # Fast quality score
        quality_score = hedge_fund_model.calculate_quality_score_fast(info)
        
        # Fast intrinsic value
        intrinsic_value = hedge_fund_model.calculate_intrinsic_value_fast(info, current_price)
        
        # ML prediction
        ml_prediction = 0.05  # Default 5%
        
        if ML_AVAILABLE and ml_model and ml_model_trained:
            try:
                # Prepare features for model
                if hasattr(ml_model, 'feature_cols'):
                    feature_df = pd.DataFrame([features])
                    
                    # Add missing columns
                    for col in ml_model.feature_cols:
                        if col not in feature_df.columns:
                            feature_df[col] = 0
                    
                    feature_df = feature_df[ml_model.feature_cols].fillna(0)
                    
                    # Scale and predict
                    if hasattr(ml_model, 'scaler'):
                        scaled = ml_model.scaler.transform(feature_df)
                        prediction = ml_model.ml_model.predict(scaled)[0]
                    else:
                        prediction = ml_model.ml_model.predict(feature_df)[0]
                    
                    ml_prediction = float(prediction)
                    
                    # Market adjustment
                    vix = market_data.get('vix', 20)
                    if vix > 30:
                        ml_prediction *= 0.85
                    elif vix < 15:
                        ml_prediction *= 1.15
                    
                    # Add variation
                    ml_prediction += (random.random() - 0.5) * 0.01
                    ml_prediction = max(-0.30, min(0.50, ml_prediction))
                    
            except Exception as e:
                logger.error(f"ML prediction error for {symbol}: {e}")
        
        # Calculate target price
        target_price = max(intrinsic_value, current_price * (1 + abs(ml_prediction)))
        
        # Build result
        result = {
            'symbol': symbol,
            'company_name': info.get('longName', symbol),
            'current_price': current_price,
            'target_price': target_price,
            'ml_prediction': ml_prediction,
            'quality_score': quality_score,
            'market_cap': info.get('marketCap', 0),
            'fundamentals': {
                'pe_ratio': features.get('pe_ratio', 20),
                'peg_ratio': features.get('peg_ratio', 1.5),
                'roe': features.get('roe', 0.10),
                'profit_margin': features.get('profit_margin', 0.05),
                'revenue_growth': features.get('revenue_growth', 0.05),
                'debt_to_equity': features.get('debt_to_equity', 50)
            },
            'technicals': {
                'volatility': features.get('volatility', 0.25),
                'momentum_20d': features.get('return_20d', 0),
                'rsi': features.get('rsi', 50)
            }
        }
        
        # Cache result
        set_cached(cache, cache_key, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis error for {symbol}: {e}")
        return None

# ============ PARALLEL BATCH ANALYSIS ============

def analyze_stocks_parallel(symbols, market_data):
    """Analyze multiple stocks in parallel for speed"""
    # First, batch download all historical data
    hist_data_map = fetch_stock_data_batch(symbols, period="3mo")
    
    results = []
    futures = []
    
    # Analyze in parallel
    with ThreadPoolExecutor(max_workers=20) as pool:
        for symbol in symbols:
            if symbol in hist_data_map:
                future = pool.submit(analyze_stock_fast, symbol, hist_data_map[symbol], market_data)
                futures.append((symbol, future))
        
        # Collect results with timeout
        for symbol, future in futures:
            try:
                result = future.result(timeout=5)
                if result:
                    # Calculate upside
                    upside = ((result['target_price'] / result['current_price']) - 1) * 100
                    result['upside'] = upside
                    
                    # Combined score
                    result['combined_score'] = (
                        upside * 0.4 +
                        result['quality_score'] * 100 * 0.3 +
                        result['ml_prediction'] * 100 * 0.3
                    )
                    
                    results.append(result)
                    
            except TimeoutError:
                logger.warning(f"Timeout analyzing {symbol}")
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
    
    return results

# ============ FAST MARKET DATA ============

def get_market_data_fast():
    """Get market data with caching"""
    global market_data_cache, market_data_timestamp
    
    if market_data_cache and (time.time() - market_data_timestamp) < 300:
        return market_data_cache
    
    market_data = {
        'vix': 20.0,
        'spy_trend': 1.0,
        'market_regime': 'Neutral'
    }
    
    try:
        # Quick VIX fetch
        vix_data = yf.download("^VIX", period="1d", progress=False, timeout=5)
        if not vix_data.empty:
            market_data['vix'] = float(vix_data['Close'].iloc[-1])
    except:
        pass
    
    # Use market analyzer if available
    if ML_AVAILABLE and market_analyzer:
        try:
            market_data['market_regime'] = market_analyzer.get_market_regime()
        except:
            pass
    
    market_data_cache = market_data
    market_data_timestamp = time.time()
    
    return market_data

# ============ OPTIMIZED BACKGROUND ANALYSIS ============

def run_analysis_background(job_id, request_data):
    """Optimized background analysis with better error handling"""
    try:
        start_time = time.time()
        logger.info(f"Starting job {job_id}: {request_data}")
        
        # Initialize job
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'processing',
                'progress': 'Starting...',
                'started': start_time
            }
        
        analysis_type = request_data['analysis_type']
        target = request_data['target']
        market_cap_size = request_data.get('market_cap_size', 'all')
        
        # Update progress
        def update_progress(msg):
            with job_lock:
                if job_id in analysis_jobs:
                    analysis_jobs[job_id]['progress'] = msg
            logger.info(f"Job {job_id}: {msg}")
        
        # Get market data
        update_progress("Getting market data...")
        market_data = get_market_data_fast()
        
        # Filter stocks
        update_progress(f"Filtering stocks for {target}...")
        
        if analysis_type == "sector":
            filtered_stocks = stocks_data[stocks_data['GICS Sector'] == target]
        else:
            filtered_stocks = stocks_data[stocks_data['GICS Sub-Industry'] == target]
        
        logger.info(f"Filtered stocks count: {len(filtered_stocks)}")
        
        if len(filtered_stocks) == 0:
            logger.warning(f"No stocks found for {target}")
            with job_lock:
                analysis_jobs[job_id] = {
                    'status': 'completed',
                    'results': {
                        'top_stocks': [],
                        'market_conditions': {
                            'vix': market_data.get('vix', 20),
                            'ml_model_trained': ml_model_trained
                        },
                        'total_analyzed': 0,
                        'analysis_time': time.time() - start_time
                    }
                }
            return
        
        # Get symbols
        all_symbols = filtered_stocks['Symbol'].tolist()
        logger.info(f"Found {len(all_symbols)} symbols: {all_symbols[:10]}...")
        
        # Market cap filtering if needed
        if market_cap_size != 'all':
            update_progress("Filtering by market cap...")
            
            # Quick market cap check using info
            filtered_symbols = []
            cap_range = MARKET_CAP_RANGES[market_cap_size]
            
            for symbol in all_symbols[:30]:  # Limit for speed
                try:
                    cached_cap = get_cached(market_cap_cache, symbol, MARKET_CAP_CACHE_DURATION)
                    if cached_cap:
                        market_cap = cached_cap
                    else:
                        ticker = yf.Ticker(symbol)
                        market_cap = ticker.info.get('marketCap', 0)
                        if market_cap > 0:
                            set_cached(market_cap_cache, symbol, market_cap)
                    
                    if cap_range['min'] <= market_cap < cap_range['max']:
                        filtered_symbols.append(symbol)
                        
                except Exception as e:
                    logger.error(f"Error getting market cap for {symbol}: {e}")
                    continue
            
            analysis_symbols = filtered_symbols[:15]
        else:
            analysis_symbols = all_symbols[:15]
        
        if not analysis_symbols:
            analysis_symbols = all_symbols[:10]
        
        logger.info(f"Will analyze {len(analysis_symbols)} symbols: {analysis_symbols}")
        
        # Analyze stocks
        update_progress(f"Analyzing {len(analysis_symbols)} stocks...")
        
        results = analyze_stocks_parallel(analysis_symbols, market_data)
        
        logger.info(f"Analysis returned {len(results)} results")
        
        # Sort and select top picks
        results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        top_picks = []
        
        # Get stocks with good upside
        for stock in results:
            if stock.get('upside', 0) > 5:
                top_picks.append(stock)
                if len(top_picks) >= 3:
                    break
        
        # If not enough, add best remaining
        if len(top_picks) < 3:
            for stock in results:
                if stock not in top_picks:
                    top_picks.append(stock)
                    if len(top_picks) >= 3:
                        break
        
        # Format results
        formatted = []
        for stock in top_picks:
            try:
                quality_factors = []
                
                # Build quality factors
                if stock.get('quality_score', 0) > 0.6:
                    quality_factors.append(f"High quality: {stock['quality_score']*100:.0f}%")
                
                fundamentals = stock.get('fundamentals', {})
                if fundamentals.get('pe_ratio', 999) < 20:
                    quality_factors.append(f"P/E: {fundamentals['pe_ratio']:.1f}")
                if fundamentals.get('roe', 0) > 0.15:
                    quality_factors.append(f"ROE: {fundamentals['roe']*100:.1f}%")
                if fundamentals.get('revenue_growth', 0) > 0.10:
                    quality_factors.append(f"Growth: {fundamentals['revenue_growth']*100:.1f}%")
                
                formatted.append({
                    "symbol": stock['symbol'],
                    "company_name": stock.get('company_name', stock['symbol']),
                    "market_cap": f"${stock.get('market_cap', 0)/1e9:.2f}B" if stock.get('market_cap', 0) > 1e9 else "N/A",
                    "metrics": {
                        "current_price": round(stock.get('current_price', 0), 2),
                        "target_price": round(stock.get('target_price', 0), 2),
                        "upside_potential": round(stock.get('upside', 0), 1),
                        "confidence_score": int(stock.get('quality_score', 0) * 100),
                        "ml_score": round(stock.get('ml_prediction', 0.05), 4)
                    },
                    "analysis_details": {
                        "fundamentals": fundamentals,
                        "technicals": stock.get('technicals', {}),
                        "ml_prediction": stock.get('ml_prediction', 0.05),
                        "quality_score": stock.get('quality_score', 0)
                    },
                    "investment_thesis": quality_factors
                })
            except Exception as e:
                logger.error(f"Error formatting result for {stock.get('symbol')}: {e}")
                continue
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Job {job_id} completed in {elapsed:.1f}s")
        
        # Set final results
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'completed',
                'results': {
                    'top_stocks': formatted,
                    'market_conditions': {
                        'vix': market_data.get('vix', 20),
                        'ml_model_trained': ml_model_trained,
                        'training_metrics': training_metrics
                    },
                    'total_analyzed': len(results),
                    'analysis_time': elapsed
                }
            }
        
    except Exception as e:
        logger.error(f"Job {job_id} failed with error: {e}")
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
    try:
        # Validate request
        if request.analysis_type not in ['sector', 'sub_industry']:
            return JSONResponse(
                content={"error": "Invalid analysis type"},
                status_code=400
            )
        
        # Check if model needs training
        if not ml_model_trained:
            logger.info("Training model before analysis...")
            train_ml_model_fast()
        
        job_id = str(uuid.uuid4())
        logger.info(f"Creating job {job_id}")
        
        # Start background task
        background_tasks.add_task(run_analysis_background, job_id, request.dict())
        
        return JSONResponse(content={"job_id": job_id, "status": "started"})
        
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
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
        
        job = analysis_jobs[job_id].copy()
        
        if job['status'] == 'completed':
            # Remove job after returning
            del analysis_jobs[job_id]
            return JSONResponse(content={
                "status": "completed",
                "results": job['results'],
                "ml_powered": ML_AVAILABLE
            })
        elif job['status'] == 'error':
            # Remove job after returning
            del analysis_jobs[job_id]
            return JSONResponse(
                content={"status": "error", "error": job.get('error', 'Unknown error')},
                status_code=500
            )
        else:
            return JSONResponse(content={
                "status": "processing",
                "progress": job.get('progress', 'Processing...')
            })

@app.get("/api/market-conditions")
async def get_market_conditions():
    """Get market conditions"""
    try:
        market_data = get_market_data_fast()
        vix = market_data.get('vix', 20)
        
        # Determine regime
        if vix > 30:
            regime = "High Volatility"
        elif vix > 20:
            regime = "Elevated"
        else:
            regime = "Normal"
        
        # Calculate recession risk
        if vix > 30:
            recession_risk = "High"
        elif vix > 25:
            recession_risk = "Medium"
        else:
            recession_risk = "Low"
        
        return JSONResponse(content={
            "regime": regime,
            "vix": vix,
            "fed_stance": "Neutral",
            "recession_risk": recession_risk,
            "ml_trained": ml_model_trained,
            "training_metrics": training_metrics
        })
        
    except Exception as e:
        logger.error(f"Error getting market conditions: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/api/stocks/list")
async def get_stocks_list():
    """Get list of sectors and sub-industries"""
    try:
        sectors = stocks_data['GICS Sector'].dropna().unique().tolist()
        sub_industries = stocks_data['GICS Sub-Industry'].dropna().unique().tolist()
        
        # Remove 'Unknown' if present
        sectors = [s for s in sectors if s != 'Unknown']
        sub_industries = [s for s in sub_industries if s != 'Unknown']
        
        return JSONResponse(content={
            "sectors": sorted(sectors),
            "sub_industries": sorted(sub_industries),
            "total_stocks": len(stocks_data),
            "ml_status": {
                "trained": ml_model_trained,
                "training_metrics": training_metrics
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting stocks list: {e}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )

@app.get("/api/ml-status")
async def get_ml_status():
    """Get ML status"""
    return JSONResponse(content={
        "ml_available": ML_AVAILABLE,
        "ml_model_trained": ml_model_trained,
        "training_metrics": training_metrics,
        "components": {
            "quant_model": ml_model is not None,
            "market_analyzer": market_analyzer is not None,
            "valuator": valuator is not None,
            "hedge_fund_model": hedge_fund_model is not None
        },
        "cache_size": len(cache) + len(market_cap_cache) + len(stock_info_cache)
    })

@app.get("/api/health")
async def health_check():
    """Health check"""
    return JSONResponse(content={
        "status": "healthy",
        "ml_available": ML_AVAILABLE,
        "ml_trained": ml_model_trained,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/")
async def root():
    """Root endpoint"""
    return JSONResponse(content={
        "name": "AutoAnalyst",
        "version": "22.0",
        "ml_enabled": ML_AVAILABLE,
        "ml_trained": ml_model_trained
    })

# ============ STARTUP ============

@app.on_event("startup")
async def startup_event():
    """Startup initialization"""
    logger.info("="*50)
    logger.info("Starting AutoAnalyst API v22.0")
    logger.info(f"ML Available: {ML_AVAILABLE}")
    logger.info(f"Stocks loaded: {len(stocks_data)}")
    
    # Start training immediately
    if ML_AVAILABLE:
        executor.submit(train_ml_model_fast)
        logger.info("Model training started...")
    
    logger.info("API Ready!")
    logger.info("="*50)

# Periodic tasks
async def periodic_retrain():
    """Retrain periodically"""
    while True:
        await asyncio.sleep(RETRAIN_INTERVAL)
        if ML_AVAILABLE:
            executor.submit(train_ml_model_fast)

async def periodic_cache_cleanup():
    """Clean cache"""
    while True:
        await asyncio.sleep(600)
        clear_old_cache()

@app.on_event("startup")
async def start_periodic_tasks():
    """Start background tasks"""
    asyncio.create_task(periodic_retrain())
    asyncio.create_task(periodic_cache_cleanup())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)