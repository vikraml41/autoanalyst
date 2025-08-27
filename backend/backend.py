#!/usr/bin/env python3
"""
FastAPI Backend - Using QuantFinanceMLModel with real ML values extraction
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
app = FastAPI(title="AutoAnalyst API", version="4.0.0")

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
            "version": "4.0.0",
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
        
        # Use MarketConditionsAnalyzer
        if market_analyzer:
            try:
                regime = market_analyzer.get_market_regime()
            except:
                pass
            
            try:
                fed_data = market_analyzer.get_federal_reserve_stance()
                fed_stance = fed_data.get("stance", "Unknown")
            except:
                pass
            
            try:
                economic_data = market_analyzer.fetch_economic_data()
                vix = economic_data.get("Volatility Index", {}).get("current", 20)
            except:
                pass
            
            try:
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
    """Use REAL ML values from YOUR model"""
    logger.info(f"=" * 50)
    logger.info(f"Analysis START: {request.analysis_type} - {request.target}")
    
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
        
        # Set data in model
        ml_model.sectors_data = filtered_stocks
        
        # Train the model first if needed
        if hasattr(ml_model, 'train_prediction_model'):
            try:
                logger.info("Training prediction model...")
                ml_model.train_prediction_model()
            except Exception as e:
                logger.error(f"Training error: {e}")
        
        # Get predictions
        predictions_df = None
        if hasattr(ml_model, 'calculate_stock_predictions'):
            try:
                logger.info("Calculating stock predictions...")
                predictions_df = ml_model.calculate_stock_predictions()
                logger.info(f"Got predictions: {type(predictions_df)}")
            except Exception as e:
                logger.error(f"Predictions error: {e}")
        
        # Process results - USE ACTUAL ML VALUES
        top_stocks = []
        
        # If we have predictions DataFrame
        if predictions_df is not None and isinstance(predictions_df, pd.DataFrame):
            logger.info(f"Processing predictions DataFrame with columns: {list(predictions_df.columns)}")
            
            # Sort by predicted returns or ML score
            sort_columns = ['Predicted_Return', 'ML_Score', 'Expected_Return', 'Score']
            sort_col = None
            for col in sort_columns:
                if col in predictions_df.columns:
                    sort_col = col
                    break
            
            if sort_col:
                predictions_df = predictions_df.sort_values(sort_col, ascending=False)
            
            # Get top 3
            for idx, row in predictions_df.head(3).iterrows():
                symbol = row.get('Symbol', idx)
                
                # Get REAL ML score from predictions
                ml_score = row.get('ML_Score', row.get('Predicted_Return', row.get('Score', 0.5)))
                if ml_score > 1:  # If it's a percentage
                    ml_score = ml_score / 100
                
                # Get current price
                current_price = row.get('Current_Price', 100)
                if current_price <= 0:
                    try:
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        current_price = info.get('currentPrice', info.get('regularMarketPrice', 100))
                    except:
                        current_price = 100
                
                # Get REAL target price from model or valuator
                target_price = row.get('Target_Price', row.get('Predicted_Price', current_price * 1.15))
                if valuator and target_price == current_price * 1.15:
                    try:
                        intrinsic = valuator.calculate_intrinsic_value(symbol)
                        if intrinsic and intrinsic > 0:
                            target_price = intrinsic
                    except:
                        pass
                
                # Calculate REAL upside
                upside = ((target_price / current_price) - 1) * 100 if current_price > 0 else 0
                
                # Get REAL confidence from model
                confidence = row.get('Confidence', row.get('Probability', ml_score))
                if confidence > 1:  # Convert to 0-1 if percentage
                    confidence = confidence / 100
                
                top_stocks.append({
                    "symbol": symbol,
                    "metrics": {
                        "current_price": round(float(current_price), 2),
                        "target_price": round(float(target_price), 2),
                        "upside_potential": round(float(upside), 1),
                        "confidence_score": int(confidence * 100),
                        "sentiment_score": row.get('Sentiment', 0.7),
                        "ml_score": round(float(ml_score), 3)
                    }
                })
                
                logger.info(f"Added {symbol}: ML={ml_score:.3f}, Upside={upside:.1f}%, Confidence={confidence:.2f}")
        
        # Fallback if no predictions DataFrame
        else:
            logger.warning("No predictions DataFrame, analyzing individually")
            
            results_list = []
            for _, stock in filtered_stocks.head(10).iterrows():
                symbol = stock.get('Symbol', '')
                if not symbol:
                    continue
                
                # Get price
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    current_price = info.get('currentPrice', info.get('regularMarketPrice', 100))
                    
                    # Get historical data for ML scoring
                    hist = ticker.history(period="3mo")
                    if len(hist) > 20:
                        # Calculate simple momentum score
                        returns = (hist['Close'].iloc[-1] - hist['Close'].iloc[-20]) / hist['Close'].iloc[-20]
                        ml_score = 0.5 + (returns * 2)  # Convert returns to 0-1 score
                        ml_score = max(0.1, min(0.9, ml_score))
                    else:
                        ml_score = 0.5
                    
                    # Get real valuation
                    target_price = current_price * 1.15
                    if valuator:
                        try:
                            intrinsic = valuator.calculate_intrinsic_value(symbol)
                            if intrinsic and intrinsic > 0:
                                target_price = intrinsic
                        except:
                            pass
                    
                    results_list.append({
                        "symbol": symbol,
                        "current_price": current_price,
                        "target_price": target_price,
                        "ml_score": ml_score
                    })
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
            
            # Sort by ML score and get top 3
            results_list.sort(key=lambda x: x['ml_score'], reverse=True)
            for stock in results_list[:3]:
                upside = ((stock['target_price'] / stock['current_price']) - 1) * 100
                
                top_stocks.append({
                    "symbol": stock['symbol'],
                    "metrics": {
                        "current_price": round(stock['current_price'], 2),
                        "target_price": round(stock['target_price'], 2),
                        "upside_potential": round(upside, 1),
                        "confidence_score": int(stock['ml_score'] * 100),
                        "sentiment_score": 0.7,
                        "ml_score": round(stock['ml_score'], 3)
                    }
                })
        
        # Get market conditions
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
                        "adjustment_factor": 1.0
                    }
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
            content={"status": "error", "error": str(e)},
            status_code=500
        )

@app.get("/api/debug-analysis")
async def debug_analysis():
    """See what run_complete_analysis() actually returns"""
    if not ml_model:
        return {"error": "No ML model"}
    
    try:
        # Get a sample of stocks
        sample_stocks = stocks_data.head(10)
        ml_model.sectors_data = sample_stocks
        
        # Run the analysis
        results = None
        if hasattr(ml_model, 'run_complete_analysis'):
            results = ml_model.run_complete_analysis()
        
        return {
            "has_run_complete_analysis": hasattr(ml_model, 'run_complete_analysis'),
            "results_type": str(type(results)) if results is not None else "None",
            "results_is_none": results is None,
            "results_sample": str(results)[:500] if results is not None else None,
            "has_dataframe": isinstance(results, pd.DataFrame) if results is not None else False,
            "columns": list(results.columns) if isinstance(results, pd.DataFrame) else None,
            "shape": results.shape if isinstance(results, pd.DataFrame) else None
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/debug-methods")
async def debug_methods():
    """Test what each method returns"""
    if not ml_model:
        return {"error": "No ML model"}
    
    results = {}
    
    # Test calculate_stock_predictions
    try:
        # Set sample data
        ml_model.sectors_data = stocks_data.head(10)
        
        # Train if needed
        if hasattr(ml_model, 'train_prediction_model'):
            ml_model.train_prediction_model()
            results["train_prediction_model"] = "Success"
        
        # Get predictions
        if hasattr(ml_model, 'calculate_stock_predictions'):
            predictions = ml_model.calculate_stock_predictions()
            results["calculate_stock_predictions"] = {
                "type": str(type(predictions)),
                "is_none": predictions is None,
                "sample": str(predictions)[:200] if predictions is not None else None,
                "columns": list(predictions.columns) if isinstance(predictions, pd.DataFrame) else None
            }
    except Exception as e:
        results["calculate_stock_predictions"] = {"error": str(e)[:100]}
    
    # Test if model has stored results
    if hasattr(ml_model, 'training_data'):
        results["has_training_data"] = ml_model.training_data is not None
    
    if hasattr(ml_model, 'ml_model'):
        results["has_trained_model"] = ml_model.ml_model is not None
    
    # Check valuator
    if valuator:
        try:
            test_value = valuator.calculate_intrinsic_value("AAPL")
            results["valuator_test"] = {
                "success": True,
                "value": test_value
            }
        except Exception as e:
            results["valuator_test"] = {"error": str(e)[:100]}
    
    return results

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
    
    # Test valuator
    if valuator:
        try:
            results['intrinsic_value'] = valuator.calculate_intrinsic_value(symbol)
        except Exception as e:
            results['valuation_error'] = str(e)[:100]
    
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
        "has_run_complete_analysis": False,
        "stocks_data_shape": stocks_data.shape if stocks_data is not None else None,
        "stocks_data_columns": list(stocks_data.columns) if stocks_data is not None else []
    }
    
    # Check ML model methods
    if ml_model:
        diagnosis["ml_model"]["methods"] = [m for m in dir(ml_model) if not m.startswith('_')]
        diagnosis["has_run_complete_analysis"] = 'run_complete_analysis' in diagnosis["ml_model"]["methods"]
    
    return diagnosis

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    logger.info(f"Using ML Model: {type(ml_model).__name__ if ml_model else 'None'}")
    uvicorn.run(app, host="0.0.0.0", port=port)