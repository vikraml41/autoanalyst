#!/usr/bin/env python3
"""
FastAPI Backend with FULL ML Support and CSV Loading
"""

import os
import glob
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import logging

# Import your FULL ML model
from quant_model import QuantFinanceMLModel, MarketConditionsAnalyzer, EnhancedValuation

app = FastAPI(title="AutoAnalyst API - Full Version", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize ML models
ml_model = QuantFinanceMLModel()
market_analyzer = MarketConditionsAnalyzer()
valuator = EnhancedValuation()

# Global storage for CSV data
stocks_data = None

def load_csv_files():
    """Load all CSV files from data directory"""
    global stocks_data
    
    data_dir = "/app/data" if os.path.exists("/app/data") else "./data"
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Find all CSV files
    csv_files = glob.glob(f"{data_dir}/*.csv")
    
    if csv_files:
        # Load and combine all CSV files
        dfs = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                logger.info(f"Loaded {len(df)} stocks from {file}")
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if dfs:
            stocks_data = pd.concat(dfs, ignore_index=True)
            logger.info(f"Total stocks loaded: {len(stocks_data)}")
            return True
    
    # If no CSV files, create sample data
    logger.warning("No CSV files found, creating sample data")
    stocks_data = pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
        'GICS Sector': ['Technology'] * 5,
        'GICS Sub-Industry': ['Hardware', 'Software', 'Internet', 'E-Commerce', 'Semiconductors']
    })
    return False

# Load CSV files on startup
csv_loaded = load_csv_files()

@app.on_event("startup")
async def startup_event():
    """Load data on startup"""
    load_csv_files()
    logger.info("AutoAnalyst API started with full ML support")
    logger.info(f"CSV data loaded: {csv_loaded}")
    logger.info(f"Total stocks available: {len(stocks_data) if stocks_data is not None else 0}")

@app.get("/")
async def root():
    return {
        "name": "AutoAnalyst API",
        "version": "2.0.0",
        "ml_enabled": True,
        "csv_loaded": csv_loaded,
        "total_stocks": len(stocks_data) if stocks_data is not None else 0,
        "status": "running"
    }

@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload new CSV file"""
    try:
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Save to data directory
        data_dir = "/app/data" if os.path.exists("/app/data") else "./data"
        file_path = f"{data_dir}/{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Reload all data
        load_csv_files()
        
        return {
            "status": "success",
            "filename": file.filename,
            "rows": len(df),
            "total_stocks": len(stocks_data)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/stocks/list")
async def get_stocks_list():
    if stocks_data is None or len(stocks_data) == 0:
        return {
            "sectors": [],
            "sub_industries": [],
            "total_stocks": 0,
            "error": "No data loaded"
        }
    
    return {
        "sectors": stocks_data['GICS Sector'].unique().tolist() if 'GICS Sector' in stocks_data.columns else [],
        "sub_industries": stocks_data['GICS Sub-Industry'].unique().tolist() if 'GICS Sub-Industry' in stocks_data.columns else [],
        "total_stocks": len(stocks_data),
        "csv_files_loaded": csv_loaded
    }

@app.get("/api/market-conditions")
async def get_market_conditions():
    """Get market conditions using your ML model"""
    try:
        regime = market_analyzer.get_market_regime()
        fed_data = market_analyzer.get_federal_reserve_stance()
        yield_curve = market_analyzer.analyze_yield_curve()
        economic_data = market_analyzer.fetch_economic_data()
        
        return {
            "regime": regime,
            "fed_stance": fed_data.get("stance"),
            "vix": economic_data.get("Volatility Index", {}).get("current", 20),
            "recession_risk": yield_curve.get("recession_risk"),
            "ml_powered": True
        }
    except Exception as e:
        logger.error(f"Market conditions error: {e}")
        return {
            "regime": "Neutral",
            "fed_stance": "Neutral",
            "vix": 20,
            "recession_risk": "Low",
            "ml_powered": False
        }

class AnalysisRequest(BaseModel):
    analysis_type: str
    target: str

@app.post("/api/analysis")
async def run_analysis(request: AnalysisRequest):
    """Run full ML analysis using your model"""
    try:
        if stocks_data is None or len(stocks_data) == 0:
            raise HTTPException(status_code=400, detail="No stock data loaded")
        
        # Filter stocks based on request
        if request.analysis_type == "sector":
            filtered = stocks_data[stocks_data['GICS Sector'] == request.target]
        else:
            filtered = stocks_data[stocks_data['GICS Sub-Industry'] == request.target]
        
        if len(filtered) == 0:
            raise HTTPException(status_code=404, detail=f"No stocks found for {request.target}")
        
        # Run your ACTUAL ML model
        results = []
        for _, stock in filtered.iterrows():
            symbol = stock['Symbol']
            
            # Use your real ML model methods
            ml_result = ml_model.analyze_stock(symbol)
            sentiment = ml_model.get_sentiment_score(symbol)
            valuation = valuator.calculate_intrinsic_value(symbol)
            
            results.append({
                "symbol": symbol,
                "ml_score": ml_result['score'],
                "target_price": valuation['target'],
                "sentiment": sentiment
            })
        
        # Sort by ML score and get top 3
        results.sort(key=lambda x: x['ml_score'], reverse=True)
        top_3 = results[:3]
        
        # Format response
        top_stocks = []
        for stock in top_3:
            top_stocks.append({
                "symbol": stock['symbol'],
                "metrics": {
                    "current_price": ml_model.get_current_price(stock['symbol']),
                    "target_price": stock['target_price'],
                    "upside_potential": ((stock['target_price'] / ml_model.get_current_price(stock['symbol'])) - 1) * 100,
                    "confidence_score": int(stock['ml_score'] * 100),
                    "sentiment_score": stock['sentiment'],
                    "ml_score": stock['ml_score']
                }
            })
        
        return {
            "status": "completed",
            "analysis_type": request.analysis_type,
            "target": request.target,
            "results": {
                "top_stocks": top_stocks,
                "market_conditions": market_analyzer.analyze_sector_conditions(request.target)
            },
            "ml_powered": True
        }
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)