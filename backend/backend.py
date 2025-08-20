#!/usr/bin/env python3
"""
FastAPI Backend with FULL ML Support via Docker
"""

import os
import sys
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import asyncio
import logging

# Import your ACTUAL ML model
try:
    from quant_model import QuantFinanceMLModel, MarketConditionsAnalyzer, EnhancedValuation
    ML_AVAILABLE = True
    print("✅ ML Models loaded successfully!")
except ImportError as e:
    print(f"⚠️ ML Models not available: {e}")
    ML_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AutoAnalyst API - Full ML Version", version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://autoanalyst.onrender.com",
        "https://*.onrender.com",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML models if available
if ML_AVAILABLE:
    ml_model = QuantFinanceMLModel()
    market_analyzer = MarketConditionsAnalyzer()
    valuator = EnhancedValuation()
    logger.info("ML models initialized successfully")
else:
    # Fallback mock models
    class MockModel:
        def predict_stocks(self, sector):
            return {"prediction": "mock", "confidence": 0.75}
    
    ml_model = MockModel()
    market_analyzer = None
    valuator = None

# Load your CSV data
def load_stock_data():
    try:
        # Try to load from data folder
        csv_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        if csv_files:
            df = pd.read_csv(f'data/{csv_files[0]}')
            logger.info(f"Loaded {len(df)} stocks from {csv_files[0]}")
            return df
    except:
        pass
    
    # Fallback to sample data
    sample_stocks = {
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA'],
        'GICS Sector': ['Technology', 'Technology', 'Technology', 'Consumer Discretionary', 
                        'Technology', 'Technology', 'Consumer Discretionary'],
        'GICS Sub-Industry': ['Hardware', 'Software', 'Internet', 'E-Commerce', 
                              'Semiconductors', 'Social Media', 'Automobiles']
    }
    return pd.DataFrame(sample_stocks)

stocks_data = load_stock_data()

@app.get("/")
async def root():
    return {
        "name": "AutoAnalyst API",
        "version": "2.0.0",
        "ml_enabled": ML_AVAILABLE,
        "status": "running",
        "models": {
            "ml_model": "QuantFinanceMLModel" if ML_AVAILABLE else "Mock",
            "market_analyzer": "MarketConditionsAnalyzer" if ML_AVAILABLE else "Mock",
            "valuator": "EnhancedValuation" if ML_AVAILABLE else "Mock"
        }
    }

@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "ml_available": ML_AVAILABLE,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/market-conditions")
async def get_market_conditions():
    try:
        if ML_AVAILABLE and market_analyzer:
            # Use real ML model
            regime = market_analyzer.get_market_regime()
            fed_data = market_analyzer.get_federal_reserve_stance()
            yield_curve = market_analyzer.analyze_yield_curve()
            economic_data = market_analyzer.fetch_economic_data()
            
            return {
                "regime": regime,
                "fed_stance": fed_data.get("stance", "Neutral"),
                "vix": economic_data.get("Volatility Index", {}).get("current", 20),
                "recession_risk": yield_curve.get("recession_risk", "Medium"),
                "yield_curve": yield_curve.get("curve_shape", "Normal"),
                "ml_powered": True
            }
        else:
            # Fallback to basic yfinance data
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")
            current_vix = float(vix_data['Close'].iloc[-1]) if len(vix_data) > 0 else 20
            
            return {
                "regime": "Neutral",
                "fed_stance": "Neutral",
                "vix": current_vix,
                "recession_risk": "Low" if current_vix < 20 else "Medium",
                "ml_powered": False
            }
    except Exception as e:
        logger.error(f"Market conditions error: {e}")
        return {"regime": "Unknown", "fed_stance": "Neutral", "vix": 20, "recession_risk": "Medium"}

@app.get("/api/stocks/list")
async def get_stocks_list():
    return {
        "sectors": stocks_data['GICS Sector'].unique().tolist(),
        "sub_industries": stocks_data['GICS Sub-Industry'].unique().tolist(),
        "total_stocks": len(stocks_data),
        "ml_enabled": ML_AVAILABLE
    }

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest):
    try:
        # Filter stocks based on request
        if request.analysis_type == "sector":
            filtered = stocks_data[stocks_data['GICS Sector'] == request.target]
        else:
            filtered = stocks_data[stocks_data['GICS Sub-Industry'] == request.target]
        
        if ML_AVAILABLE and ml_model:
            # USE YOUR REAL ML MODEL
            logger.info(f"Running ML analysis for {request.target}")
            
            # Call your actual model methods
            predictions = ml_model.predict_stocks(filtered['Symbol'].tolist())
            market_conditions = market_analyzer.analyze_sector_conditions(request.target)
            
            # Get top 3 predictions
            top_stocks = []
            for symbol in filtered['Symbol'].head(3):
                # Run your actual ML prediction
                ml_score = ml_model.calculate_ml_score(symbol)
                sentiment = ml_model.get_sentiment_score(symbol)
                target_price = valuator.calculate_intrinsic_value(symbol)
                
                top_stocks.append({
                    "symbol": symbol,
                    "metrics": {
                        "current_price": ml_model.get_current_price(symbol),
                        "target_price": target_price,
                        "upside_potential": ((target_price / ml_model.get_current_price(symbol)) - 1) * 100,
                        "confidence_score": ml_score * 100,
                        "sentiment_score": sentiment,
                        "ml_score": ml_score
                    }
                })
            
            return {
                "status": "completed",
                "ml_powered": True,
                "analysis_type": request.analysis_type,
                "target": request.target,
                "results": {
                    "top_stocks": top_stocks,
                    "market_conditions": market_conditions
                }
            }
        else:
            # Fallback without ML
            logger.warning("ML not available, using basic analysis")
            # ... basic analysis code ...
            
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    logger.info(f"ML Available: {ML_AVAILABLE}")
    uvicorn.run(app, host="0.0.0.0", port=port)