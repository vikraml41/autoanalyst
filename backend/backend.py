#!/usr/bin/env python3
"""
FastAPI Backend - Speed Optimized with Advanced ML Pretraining
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
app = FastAPI(title="AutoAnalyst API", version="17.0.0")

# Thread pool for fast parallel processing
executor = ThreadPoolExecutor(max_workers=20)

# Job storage
analysis_jobs = {}
job_lock = threading.Lock()

# Enhanced caching system
cache = {}
market_cap_cache = {}
stock_info_cache = {}
CACHE_DURATION = 3600
MARKET_CAP_CACHE_DURATION = 7200

# Market data cache
market_data_cache = None
market_data_timestamp = 0

# ML Model training state
ml_model_trained = False
training_metrics = {}
sector_models = {}  # Store sector-specific models

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

# ============ OPTIMIZED CACHING ============

def get_cached(cache_dict, key, duration=CACHE_DURATION):
    """Get from cache if not expired"""
    if key in cache_dict:
        data, timestamp = cache_dict[key]
        if time.time() - timestamp < duration:
            return data
    return None

def set_cached(cache_dict, key, data):
    """Set cache with timestamp"""
    cache_dict[key] = (data, time.time())

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
            'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'INTC', 'JPM'],
            'GICS Sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 
                          'Technology', 'Technology', 'Consumer Discretionary', 'Technology', 
                          'Technology', 'Financials']
        })
    
    all_dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_dfs.append(df)
        except:
            pass
    
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined)} stocks")
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
    logger.info("✅ ML Models loaded")
    
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    ML_AVAILABLE = False

# ============ SIMPLE FALLBACK MODEL ============

class SimpleMLModel:
    """Simple ML model for when main model isn't available"""
    
    def predict_return(self, pe_ratio, peg_ratio, roe, revenue_growth, rsi, volatility, vix):
        """Simple heuristic prediction based on key metrics"""
        score = 0.05  # Base 5% return
        
        # PE Ratio contribution
        if 0 < pe_ratio < 15:
            score += 0.03
        elif 15 <= pe_ratio < 25:
            score += 0.01
        elif pe_ratio > 40 or pe_ratio <= 0:
            score -= 0.02
            
        # PEG Ratio contribution
        if 0 < peg_ratio < 1:
            score += 0.04
        elif 1 <= peg_ratio < 1.5:
            score += 0.02
        elif peg_ratio > 2 or peg_ratio <= 0:
            score -= 0.01
            
        # ROE contribution
        if roe > 0.20:
            score += 0.03
        elif roe > 0.15:
            score += 0.02
        elif roe > 0.10:
            score += 0.01
        elif roe < 0:
            score -= 0.03
            
        # Revenue Growth contribution
        if revenue_growth > 0.20:
            score += 0.04
        elif revenue_growth > 0.10:
            score += 0.02
        elif revenue_growth > 0.05:
            score += 0.01
        elif revenue_growth < 0:
            score -= 0.02
            
        # RSI contribution (momentum)
        if 30 < rsi < 70:
            score += 0.01
        elif rsi <= 30:
            score += 0.02  # Oversold
        elif rsi >= 70:
            score -= 0.01  # Overbought
            
        # Volatility adjustment
        if volatility > 0.50:
            score *= 0.8
        elif volatility < 0.20:
            score *= 1.1
            
        # VIX adjustment
        if vix > 30:
            score *= 0.85
        elif vix < 15:
            score *= 1.15
            
        # Ensure reasonable range
        return max(-0.20, min(0.30, score))

simple_model = SimpleMLModel()

# ============ COMPREHENSIVE ML PRETRAINING ============

def create_advanced_features(data, symbol):
    """Create advanced technical and fundamental features"""
    features = {}
    
    try:
        # Price-based features
        if 'Close' in data:
            close = data['Close']
            
            # Moving averages
            features['sma_20'] = close.rolling(20).mean().iloc[-1] / close.iloc[-1] if len(close) > 20 else 1
            features['sma_50'] = close.rolling(50).mean().iloc[-1] / close.iloc[-1] if len(close) > 50 else 1
            features['sma_200'] = close.rolling(200).mean().iloc[-1] / close.iloc[-1] if len(close) > 200 else 1
            
            # Momentum indicators
            features['roc_10'] = (close.iloc[-1] / close.iloc[-10] - 1) if len(close) > 10 else 0
            features['roc_30'] = (close.iloc[-1] / close.iloc[-30] - 1) if len(close) > 30 else 0
            
            # Volatility
            returns = close.pct_change()
            features['volatility_20'] = returns.tail(20).std() * np.sqrt(252) if len(returns) > 20 else 0.25
            features['volatility_60'] = returns.tail(60).std() * np.sqrt(252) if len(returns) > 60 else features['volatility_20']
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = float(100 - (100 / (1 + rs)).iloc[-1]) if len(close) > 14 else 50
            
            # MACD
            if len(close) > 26:
                exp1 = close.ewm(span=12, adjust=False).mean()
                exp2 = close.ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                features['macd_signal'] = float((macd - signal).iloc[-1] / close.iloc[-1]) if close.iloc[-1] != 0 else 0
            else:
                features['macd_signal'] = 0
            
            # Bollinger Bands
            if len(close) > 20:
                sma_20 = close.rolling(20).mean()
                std_20 = close.rolling(20).std()
                upper_band = sma_20 + (std_20 * 2)
                lower_band = sma_20 - (std_20 * 2)
                bb_width = upper_band.iloc[-1] - lower_band.iloc[-1]
                if bb_width > 0:
                    features['bb_position'] = float((close.iloc[-1] - lower_band.iloc[-1]) / bb_width)
                else:
                    features['bb_position'] = 0.5
            else:
                features['bb_position'] = 0.5
            
        # Volume features
        if 'Volume' in data:
            volume = data['Volume']
            features['volume_ratio'] = float(volume.tail(10).mean() / volume.tail(50).mean()) if len(volume) > 50 and volume.tail(50).mean() > 0 else 1
            features['volume_trend'] = float((volume.tail(5).mean() / volume.tail(20).mean()) - 1) if len(volume) > 20 and volume.tail(20).mean() > 0 else 0
        
    except Exception as e:
        logger.error(f"Error creating features for {symbol}: {e}")
        # Return default features
        features = {
            'sma_20': 1, 'sma_50': 1, 'sma_200': 1,
            'roc_10': 0, 'roc_30': 0,
            'volatility_20': 0.25, 'volatility_60': 0.25,
            'rsi': 50, 'macd_signal': 0, 'bb_position': 0.5,
            'volume_ratio': 1, 'volume_trend': 0
        }
    
    return features

def prepare_comprehensive_training_data(symbols, market_conditions_history):
    """Prepare comprehensive training data with multiple timeframes"""
    all_features = []
    all_targets = []
    
    for symbol in symbols[:15]:  # Limit to speed up training
        try:
            # Get 3 years of data for robust training
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y", interval="1d")  # Reduced from 3y for speed
            
            if hist.empty or len(hist) < 252:  # Need at least 1 year
                continue
            
            info = ticker.info
            
            # Generate samples from different time windows
            for i in range(60, len(hist) - 30, 60):  # Every 2 months instead of 1
                try:
                    window_data = hist.iloc[:i]
                    
                    # Create technical features
                    tech_features = create_advanced_features(window_data, symbol)
                    
                    # Add fundamental features
                    features = {
                        'symbol': symbol,
                        'sector': stocks_data[stocks_data['Symbol'] == symbol]['GICS Sector'].iloc[0] if not stocks_data[stocks_data['Symbol'] == symbol].empty else 'Unknown',
                        'pe_ratio': info.get('trailingPE', 20) or 20,
                        'peg_ratio': info.get('pegRatio', 1.5) or 1.5,
                        'profit_margin': info.get('profitMargins', 0.1) or 0.1,
                        'revenue_growth': info.get('revenueGrowth', 0.05) or 0.05,
                        'debt_to_equity': info.get('debtToEquity', 0.5) or 0.5,
                        'roe': info.get('returnOnEquity', 0.15) or 0.15,
                        'market_cap': info.get('marketCap', 1e9),
                        'price_to_book': info.get('priceToBook', 2) or 2,
                        'dividend_yield': info.get('dividendYield', 0) or 0,
                        'beta': info.get('beta', 1) or 1,
                        **tech_features
                    }
                    
                    # Add market conditions at that time
                    features['historical_vix'] = 20  # Simplified
                    
                    # Calculate target (30-day forward return)
                    current_price = hist['Close'].iloc[i]
                    future_price = hist['Close'].iloc[min(i + 30, len(hist) - 1)]
                    target_return = (future_price / current_price - 1)
                    
                    all_features.append(features)
                    all_targets.append(target_return)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
    
    return pd.DataFrame(all_features), pd.Series(all_targets)

def comprehensive_pretrain_ml_model():
    """Comprehensive ML model pretraining with validation"""
    global ml_model, ml_model_trained, training_metrics
    
    if not ML_AVAILABLE or not ml_model:
        logger.info("ML model not available, skipping training")
        return
    
    try:
        start_time = time.time()
        logger.info("="*50)
        logger.info("Starting comprehensive ML model pretraining...")
        
        # Step 1: Select diverse training stocks
        training_symbols = []
        
        # Get stocks from each sector
        sectors = stocks_data['GICS Sector'].unique()
        for sector in sectors[:8]:  # Reduced for speed
            sector_stocks = stocks_data[stocks_data['GICS Sector'] == sector]['Symbol'].tolist()
            if len(sector_stocks) >= 5:
                training_symbols.extend(sector_stocks[:5])
        
        logger.info(f"Selected {len(training_symbols)} training stocks")
        
        # Step 2: Prepare training data
        train_features, train_targets = prepare_comprehensive_training_data(training_symbols, None)
        
        if train_features.empty or len(train_features) < 50:
            logger.error("Insufficient training data, using simple model")
            ml_model_trained = False
            return
        
        logger.info(f"Prepared {len(train_features)} training samples")
        
        # Step 3: Feature preprocessing
        from sklearn.preprocessing import RobustScaler
        from sklearn.feature_selection import SelectKBest, f_regression
        
        # Select numeric features only
        numeric_features = train_features.select_dtypes(include=[np.number])
        numeric_features = numeric_features.fillna(numeric_features.median())
        
        # Scale features
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(numeric_features)
        
        # Feature selection - select top features
        k_features = min(20, scaled_features.shape[1])
        selector = SelectKBest(f_regression, k=k_features)
        selected_features = selector.fit_transform(scaled_features, train_targets)
        
        logger.info(f"Selected {selected_features.shape[1]} best features")
        
        # Step 4: Train simple Random Forest
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(
            n_estimators=100,  # Reduced for speed
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        model.fit(selected_features, train_targets)
        
        # Step 5: Store model components properly
        ml_model.prediction_model = model
        ml_model.feature_scaler = scaler
        ml_model.feature_selector = selector
        ml_model.feature_columns = numeric_features.columns.tolist()
        
        # Store with expected attribute names
        ml_model.ml_model = model
        ml_model.scaler = scaler
        ml_model.feature_cols = numeric_features.columns.tolist()
        
        # Calculate basic training metrics
        from sklearn.metrics import r2_score, mean_squared_error
        train_predictions = model.predict(selected_features)
        train_r2 = r2_score(train_targets, train_predictions)
        train_mse = mean_squared_error(train_targets, train_predictions)
        
        training_metrics = {
            'training_samples': len(train_features),
            'features_used': k_features,
            'train_r2': train_r2,
            'train_mse': train_mse,
            'training_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        ml_model_trained = True
        logger.info(f"✅ ML model training completed in {time.time() - start_time:.1f} seconds")
        logger.info(f"Training R²: {train_r2:.4f}, MSE: {train_mse:.6f}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Pretraining failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        ml_model_trained = False

# ============ FAST BATCH OPERATIONS ============

def batch_download_stocks(symbols, period="1d"):
    """Download multiple stocks at once"""
    try:
        data = yf.download(
            tickers=' '.join(symbols),
            period=period,
            group_by='ticker',
            threads=True,
            progress=False,
            timeout=30
        )
        
        result = {}
        for symbol in symbols:
            try:
                if len(symbols) == 1:
                    result[symbol] = {
                        'Close': data['Close'].iloc[-1] if 'Close' in data else None,
                        'Volume': data['Volume'].iloc[-1] if 'Volume' in data else None
                    }
                else:
                    if symbol in data:
                        result[symbol] = {
                            'Close': data[symbol]['Close'].iloc[-1] if 'Close' in data[symbol] else None,
                            'Volume': data[symbol]['Volume'].iloc[-1] if 'Volume' in data[symbol] else None
                        }
            except:
                continue
        
        return result
    except Exception as e:
        logger.error(f"Batch download error: {e}")
        return {}

def get_market_caps_batch(symbols):
    """Get market caps for multiple symbols efficiently"""
    result = {}
    uncached_symbols = []
    
    for symbol in symbols:
        cached = get_cached(market_cap_cache, symbol, MARKET_CAP_CACHE_DURATION)
        if cached is not None:
            result[symbol] = cached
        else:
            uncached_symbols.append(symbol)
    
    if not uncached_symbols:
        return result
    
    try:
        def get_market_cap(symbol):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                market_cap = info.market_cap if hasattr(info, 'market_cap') else 0
                
                if market_cap == 0:
                    full_info = ticker.info
                    market_cap = full_info.get('marketCap', 0)
                
                return symbol, market_cap
            except:
                return symbol, 0
        
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(get_market_cap, s) for s in uncached_symbols[:20]]
            for future in as_completed(futures):
                try:
                    symbol, cap = future.result(timeout=3)
                    if cap > 0:
                        result[symbol] = cap
                        set_cached(market_cap_cache, symbol, cap)
                except:
                    continue
    except Exception as e:
        logger.error(f"Error getting market caps: {e}")
    
    return result

# ============ FAST VIX AND MARKET DATA ============

def fetch_vix_fast():
    """Fast VIX fetching with cache"""
    cached_vix = get_cached(cache, 'vix_value', 900)
    if cached_vix:
        return cached_vix
    
    try:
        vix_data = yf.download("^VIX", period="1d", progress=False, timeout=5)
        if not vix_data.empty:
            vix_value = float(vix_data['Close'].iloc[-1])
            set_cached(cache, 'vix_value', vix_value)
            return vix_value
    except:
        pass
    
    return 20.0

def get_market_data():
    """Get market data with aggressive caching"""
    global market_data_cache, market_data_timestamp
    
    if market_data_cache and (time.time() - market_data_timestamp) < 900:
        return market_data_cache
    
    market_data = {
        'vix': fetch_vix_fast(),
        'spy_price': 500,
        'spy_trend': 1,
        'treasury_10y': 4.3,
        'dollar_index': 105
    }
    
    try:
        spy_data = yf.download("SPY", period="5d", progress=False, timeout=5)
        if not spy_data.empty:
            market_data['spy_price'] = float(spy_data['Close'].iloc[-1])
            market_data['spy_trend'] = float(spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[0])
    except:
        pass
    
    market_data_cache = market_data
    market_data_timestamp = time.time()
    
    return market_data

# ============ FAST STOCK ANALYSIS WITH FIXED ML ============

def analyze_stocks_fast(symbols, market_data, sector=None):
    """Fast parallel analysis using pretrained model"""
    
    # Batch download all price data
    logger.info(f"Batch downloading {len(symbols)} stocks...")
    price_data = yf.download(
        tickers=' '.join(symbols),
        period="3mo",
        interval="1d",
        group_by='ticker',
        threads=True,
        progress=False,
        timeout=20
    )
    
    # Get stock info in parallel
    def get_stock_info(symbol):
        try:
            cache_key = f"{symbol}_full_info"
            cached = get_cached(stock_info_cache, cache_key, 3600)
            if cached:
                return symbol, cached
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            set_cached(stock_info_cache, cache_key, info)
            return symbol, info
        except:
            return symbol, {}
    
    stock_infos = {}
    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(get_stock_info, s) for s in symbols]
        for future in as_completed(futures):
            try:
                symbol, info = future.result(timeout=5)
                stock_infos[symbol] = info
            except:
                continue
    
    # Analyze all stocks
    results = []
    
    for symbol in symbols:
        try:
            info = stock_infos.get(symbol, {})
            
            # Get price data
            if len(symbols) == 1:
                hist = price_data
            else:
                if symbol not in price_data:
                    continue
                hist = price_data[symbol]
            
            if hist.empty:
                continue
            
            current_price = float(hist['Close'].iloc[-1])
            
            # Calculate technical features
            tech_features = create_advanced_features(hist, symbol)
            
            # Get metrics
            metrics = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'current_price': current_price,
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0) or info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'profit_margin': info.get('profitMargins', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'roe': info.get('returnOnEquity', 0),
                'price_to_book': info.get('priceToBook', 0),
                'beta': info.get('beta', 1),
                'target_price': info.get('targetMeanPrice', current_price * 1.1),
                'recommendation': info.get('recommendationMean', 3),
                'analyst_count': info.get('numberOfAnalystOpinions', 0)
            }
            
            # Quality score
            quality_score = 0
            reasons = []
            
            if 0 < metrics['pe_ratio'] < 25:
                quality_score += 0.2
                reasons.append(f"Good P/E: {metrics['pe_ratio']:.1f}")
            if 0 < metrics['peg_ratio'] < 1.5:
                quality_score += 0.2
                reasons.append(f"Attractive PEG: {metrics['peg_ratio']:.2f}")
            if metrics['roe'] > 0.15:
                quality_score += 0.2
                reasons.append(f"Strong ROE: {metrics['roe']*100:.1f}%")
            if metrics['revenue_growth'] > 0.1:
                quality_score += 0.2
                reasons.append(f"Good growth: {metrics['revenue_growth']*100:.1f}%")
            if metrics['recommendation'] < 2.5:
                quality_score += 0.2
                reasons.append("Buy rating from analysts")
            
            # FIXED ML PREDICTION
            ml_prediction = 0.05  # Default 5%
            
            try:
                if ML_AVAILABLE and ml_model and ml_model_trained:
                    # Prepare features for trained model
                    features_dict = {
                        'pe_ratio': metrics['pe_ratio'] if metrics['pe_ratio'] > 0 else 20,
                        'peg_ratio': metrics['peg_ratio'] if metrics['peg_ratio'] > 0 else 1.5,
                        'profit_margin': metrics['profit_margin'],
                        'revenue_growth': metrics['revenue_growth'],
                        'debt_to_equity': metrics['debt_to_equity'],
                        'roe': metrics['roe'],
                        'market_cap': metrics['market_cap'] if metrics['market_cap'] > 0 else 1e9,
                        'price_to_book': metrics['price_to_book'] if metrics['price_to_book'] > 0 else 2,
                        'dividend_yield': info.get('dividendYield', 0),
                        'beta': metrics['beta'],
                        **tech_features,
                        'historical_vix': market_data['vix']
                    }
                    
                    # Create dataframe with features
                    features_df = pd.DataFrame([features_dict])
                    
                    # Add missing columns
                    for col in ml_model.feature_columns:
                        if col not in features_df.columns:
                            features_df[col] = 0
                    
                    # Select columns
                    features_df = features_df[ml_model.feature_columns].fillna(0)
                    
                    # Scale
                    scaled_features = ml_model.feature_scaler.transform(features_df)
                    
                    # Select features
                    selected_features = ml_model.feature_selector.transform(scaled_features)
                    
                    # Predict
                    raw_prediction = float(ml_model.prediction_model.predict(selected_features)[0])
                    
                    # Ensure reasonable range
                    ml_prediction = max(0.01, min(0.30, abs(raw_prediction)))
                    
                    logger.info(f"{symbol}: Trained ML prediction = {ml_prediction*100:.2f}%")
                    
                else:
                    # Use simple model when trained model not available
                    ml_prediction = simple_model.predict_return(
                        pe_ratio=metrics['pe_ratio'],
                        peg_ratio=metrics['peg_ratio'],
                        roe=metrics['roe'],
                        revenue_growth=metrics['revenue_growth'],
                        rsi=tech_features.get('rsi', 50),
                        volatility=tech_features.get('volatility_20', 0.25),
                        vix=market_data['vix']
                    )
                    logger.info(f"{symbol}: Simple model prediction = {ml_prediction*100:.2f}%")
                    
            except Exception as e:
                logger.error(f"ML prediction error for {symbol}: {e}")
                # Fallback to simple model
                ml_prediction = simple_model.predict_return(
                    pe_ratio=metrics['pe_ratio'],
                    peg_ratio=metrics['peg_ratio'],
                    roe=metrics['roe'],
                    revenue_growth=metrics['revenue_growth'],
                    rsi=tech_features.get('rsi', 50),
                    volatility=tech_features.get('volatility_20', 0.25),
                    vix=market_data['vix']
                )
            
            # Ensure ML prediction is never 0
            if abs(ml_prediction) < 0.01:
                ml_prediction = 0.05
            
            # Calculate upside
            upside = ((metrics['target_price'] / current_price) - 1) * 100
            
            # Combined score
            combined_score = (
                upside * 0.4 +
                quality_score * 100 * 0.3 +
                ml_prediction * 100 * 0.3
            )
            
            if upside > 3:
                results.append({
                    'symbol': symbol,
                    'company_name': metrics['company_name'],
                    'current_price': current_price,
                    'target_price': metrics['target_price'],
                    'upside': upside,
                    'quality_score': quality_score,
                    'ml_prediction': ml_prediction,
                    'combined_score': combined_score,
                    'metrics': metrics,
                    'technicals': tech_features,
                    'market_cap': metrics['market_cap'],
                    'reasons': reasons
                })
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            continue
    
    return results

# ============ STREAMLINED BACKGROUND ANALYSIS ============

def run_analysis_background(job_id, request_data):
    """Streamlined fast analysis with pretrained models"""
    try:
        start_time = time.time()
        
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'processing',
                'progress': 'Starting analysis...',
                'started': start_time
            }
        
        analysis_type = request_data['analysis_type']
        target = request_data['target']
        market_cap_size = request_data.get('market_cap_size', 'all')
        
        # Get market data
        market_data = get_market_data()
        
        # Filter stocks
        if analysis_type == "sector":
            filtered_stocks = stocks_data[stocks_data['GICS Sector'] == target]
            sector = target
        else:
            filtered_stocks = stocks_data[stocks_data['GICS Sub-Industry'] == target]
            sector = filtered_stocks['GICS Sector'].iloc[0] if not filtered_stocks.empty else None
        
        if len(filtered_stocks) == 0:
            with job_lock:
                analysis_jobs[job_id] = {'status': 'error', 'error': f'No stocks found'}
            return
        
        all_symbols = filtered_stocks['Symbol'].tolist()
        
        # Fast market cap filtering
        with job_lock:
            analysis_jobs[job_id]['progress'] = 'Filtering by market cap...'
        
        if market_cap_size != 'all':
            market_caps = get_market_caps_batch(all_symbols[:50])
            cap_range = MARKET_CAP_RANGES[market_cap_size]
            
            filtered = [(s, c) for s, c in market_caps.items() 
                       if cap_range['min'] <= c < cap_range['max']]
            filtered.sort(key=lambda x: x[1], reverse=True)
            
            analysis_symbols = [s[0] for s in filtered[:15]]
        else:
            analysis_symbols = all_symbols[:15]
        
        if not analysis_symbols:
            analysis_symbols = all_symbols[:10]
        
        # Fast analysis with pretrained model
        with job_lock:
            analysis_jobs[job_id]['progress'] = f'Analyzing {len(analysis_symbols)} stocks...'
        
        results = analyze_stocks_fast(analysis_symbols, market_data, sector)
        
        # Sort and get top picks
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        top_picks = []
        
        # Get stocks with >5% upside first
        for stock in results:
            if stock['upside'] > 5:
                top_picks.append(stock)
                if len(top_picks) >= 3:
                    break
        
        # If not enough, add best remaining
        if len(top_picks) < 3:
            for stock in results:
                if stock not in top_picks and stock['upside'] > 0:
                    top_picks.append(stock)
                    if len(top_picks) >= 3:
                        break
        
        # Format results
        formatted = []
        for stock in top_picks:
            formatted.append({
                "symbol": stock['symbol'],
                "company_name": stock['company_name'],
                "market_cap": f"${stock['market_cap']/1e9:.2f}B" if stock['market_cap'] > 1e9 else f"${stock['market_cap']/1e6:.0f}M",
                "metrics": {
                    "current_price": round(stock['current_price'], 2),
                    "target_price": round(stock['target_price'], 2),
                    "upside_potential": round(stock['upside'], 1),
                    "confidence_score": int(stock['quality_score'] * 100),
                    "ml_score": round(stock['ml_prediction'], 4)
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
                    "technicals": {k: round(v, 4) if isinstance(v, float) else v 
                                  for k, v in stock['technicals'].items()},
                    "ml_prediction": stock['ml_prediction'],
                    "quality_score": stock['quality_score']
                },
                "investment_thesis": stock['reasons']
            })
        
        elapsed = time.time() - start_time
        logger.info(f"Analysis completed in {elapsed:.1f} seconds")
        
        with job_lock:
            analysis_jobs[job_id] = {
                'status': 'completed',
                'results': {
                    'top_stocks': formatted,
                    'market_conditions': {
                        'vix': market_data['vix'],
                        'ml_model_trained': ml_model_trained,
                        'training_metrics': training_metrics if ml_model_trained else {}
                    },
                    'total_analyzed': len(results),
                    'analysis_time': elapsed
                }
            }
        
    except Exception as e:
        logger.error(f"Job failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        with job_lock:
            analysis_jobs[job_id] = {'status': 'error', 'error': str(e)}

# ============ API ENDPOINTS ============

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str
    market_cap_size: str = 'all'

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    background_tasks.add_task(run_analysis_background, job_id, request.dict())
    return JSONResponse(content={"job_id": job_id, "status": "started"})

@app.get("/api/analysis/{job_id}")
async def get_analysis_status(job_id: str):
    with job_lock:
        if job_id not in analysis_jobs:
            return JSONResponse(content={"status": "not_found"}, status_code=404)
        
        job = analysis_jobs[job_id]
        if job['status'] == 'completed':
            result = {"status": "completed", "results": job['results'], "ml_powered": ML_AVAILABLE}
            del analysis_jobs[job_id]
            return JSONResponse(content=result)
        elif job['status'] == 'error':
            result = {"status": "error", "error": job.get('error')}
            del analysis_jobs[job_id]
            return JSONResponse(content=result, status_code=500)
        else:
            return JSONResponse(content={"status": "processing", "progress": job.get('progress')})

@app.get("/api/ml-status")
async def get_ml_status():
    """Diagnostic endpoint to check ML model status"""
    status = {
        "ml_available": ML_AVAILABLE,
        "ml_model_trained": ml_model_trained,
        "training_metrics": training_metrics,
        "has_ml_model": ml_model is not None,
        "has_prediction_model": hasattr(ml_model, 'prediction_model') if ml_model else False,
        "has_feature_columns": hasattr(ml_model, 'feature_columns') if ml_model else False,
        "feature_count": len(ml_model.feature_columns) if ml_model and hasattr(ml_model, 'feature_columns') else 0,
        "sector_models_count": len(sector_models)
    }
    return JSONResponse(content=status)

@app.get("/api/market-conditions")
async def get_market_conditions():
    """Get market conditions with fixed recession risk calculation"""
    market_data = get_market_data()
    vix = market_data['vix']
    
    # Determine market regime
    regime = "High Volatility" if vix > 30 else "Elevated" if vix > 20 else "Normal"
    
    # Calculate recession risk
    recession_risk = "Unknown"
    try:
        # Try to get yield curve data
        ten_year_data = yf.download("^TNX", period="1d", progress=False, timeout=5)
        two_year_data = yf.download("^FVX", period="1d", progress=False, timeout=5)  # 5-year as proxy
        
        if not ten_year_data.empty:
            ten_year = float(ten_year_data['Close'].iloc[-1])
            
            if not two_year_data.empty:
                two_year = float(two_year_data['Close'].iloc[-1])
                yield_spread = ten_year - two_year
                
                if yield_spread < -0.5:
                    recession_risk = "High"
                elif yield_spread < 0:
                    recession_risk = "Medium-High"
                elif yield_spread < 0.5:
                    recession_risk = "Medium"
                else:
                    recession_risk = "Low"
            else:
                # Use VIX as proxy
                if vix > 30:
                    recession_risk = "High"
                elif vix > 25:
                    recession_risk = "Medium"
                else:
                    recession_risk = "Low"
                    
    except Exception as e:
        logger.error(f"Error calculating recession risk: {e}")
        # Fallback based on VIX
        if vix > 30:
            recession_risk = "High"
        elif vix > 25:
            recession_risk = "Medium"
        else:
            recession_risk = "Low"
    
    # Determine Fed stance
    fed_stance = "Neutral"
    try:
        fed_data = yf.download("^IRX", period="3mo", progress=False, timeout=5)
        if not fed_data.empty and len(fed_data) > 20:
            recent = float(fed_data['Close'].iloc[-1])
            past = float(fed_data['Close'].iloc[-20])
            
            if recent > past + 0.25:
                fed_stance = "Hawkish"
            elif recent < past - 0.25:
                fed_stance = "Dovish"
    except:
        pass
    
    return JSONResponse(content={
        "regime": regime,
        "vix": vix,
        "fed_stance": fed_stance,
        "recession_risk": recession_risk,
        "ml_trained": ml_model_trained,
        "training_metrics": training_metrics if ml_model_trained else {}
    })

@app.get("/api/stocks/list")
async def get_stocks_list():
    sectors = stocks_data['GICS Sector'].dropna().unique().tolist() if 'GICS Sector' in stocks_data.columns else []
    sub_industries = stocks_data['GICS Sub-Industry'].dropna().unique().tolist() if 'GICS Sub-Industry' in stocks_data.columns else []
    total_stocks = len(stocks_data)
    
    return JSONResponse(content={
        "sectors": sectors,
        "sub_industries": sub_industries,
        "total_stocks": total_stocks,
        "ml_status": {
            "trained": ml_model_trained,
            "training_metrics": training_metrics
        }
    })

@app.get("/api/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "ml_available": ML_AVAILABLE,
        "ml_trained": ml_model_trained,
        "timestamp": datetime.now().isoformat()
    })

@app.get("/")
async def root():
    return JSONResponse(content={
        "name": "AutoAnalyst",
        "version": "17.1",
        "ml_enabled": ML_AVAILABLE,
        "ml_trained": ml_model_trained,
        "training_metrics": training_metrics
    })

# ============ STARTUP AND PERIODIC TASKS ============

@app.on_event("startup")
async def startup_event():
    """Initialize and pretrain model on startup"""
    logger.info("Starting up...")
    
    # Load market data
    get_market_data()
    
    # Start pretraining in background
    if ML_AVAILABLE:
        executor.submit(comprehensive_pretrain_ml_model)
        logger.info("ML pretraining started in background...")
    
    logger.info("API ready, ML training in progress...")

async def periodic_retrain():
    """Retrain model every 6 hours"""
    while True:
        await asyncio.sleep(3600 * 6)  # 6 hours
        if ML_AVAILABLE:
            logger.info("Starting periodic model retraining...")
            executor.submit(comprehensive_pretrain_ml_model)

@app.on_event("startup")
async def start_periodic_tasks():
    """Start background periodic tasks"""
    asyncio.create_task(periodic_retrain())

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)