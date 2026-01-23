"""
FastAPI Server for Stock Analysis
Wraps the backend.py stock analyzer and provides REST API endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import os
from backend import StockAnalyzer, DataFetchError, ALPHA_VANTAGE_KEY, MASSIVE_API_KEY, EdgarDataFetcher, fetch_stock_data, DCFValuation, RevenueForecaster, ComparableCompanyAnalysis
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Analyzer API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StockAnalysisRequest(BaseModel):
    ticker: str


@app.post('/api/stock-analysis')
async def analyze_stock(request: StockAnalysisRequest):
    """
    Analyze a stock using DCF, Revenue Forecasting, and Comparable Companies
    """
    try:
        ticker = request.ticker.strip().upper()

        if not ticker:
            raise HTTPException(status_code=400, detail='Ticker symbol is required')

        logger.info(f"Starting analysis for {ticker}")

        # Create analyzer and run complete analysis
        analyzer = StockAnalyzer(ticker)
        results = analyzer.analyze()

        if not results:
            raise HTTPException(status_code=500, detail='Analysis failed - unable to retrieve data')

        logger.info(f"Analysis completed successfully for {ticker}")

        # Ensure JSON serializable (convert numpy types)
        import json
        try:
            json.dumps(results)
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            # Try to sanitize the results
            def sanitize(obj):
                import numpy as np
                if isinstance(obj, dict):
                    return {k: sanitize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize(v) for v in obj]
                elif isinstance(obj, (np.integer, np.int64)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64)):
                    return float(obj) if not np.isnan(obj) else None
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, float) and (obj != obj):  # NaN check
                    return None
                return obj
            results = sanitize(results)

        return results

    except HTTPException:
        raise
    except DataFetchError as e:
        logger.error(f"Data fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Data fetch error: {str(e)}")
    except Exception as e:
        logger.error(f"Error analyzing stock: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get('/api/health')
async def health_check():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'service': 'stock-analyzer',
        'alpha_vantage_configured': bool(ALPHA_VANTAGE_KEY),
        'api_key_length': len(ALPHA_VANTAGE_KEY) if ALPHA_VANTAGE_KEY else 0
    }


@app.get('/api/market-conditions')
async def market_conditions():
    """Get current market conditions (placeholder for compatibility)"""
    return {
        'regime': 'Growth',
        'fed_stance': 'Neutral',
        'vix': 15.5,
        'recession_risk': 'Low'
    }


@app.get('/api/test-alpha-vantage/{ticker}')
async def test_alpha_vantage(ticker: str):
    """Debug endpoint to test Alpha Vantage API"""
    import requests

    ticker = ticker.upper()
    base = "https://www.alphavantage.co/query"

    url = f"{base}?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"

    try:
        response = requests.get(url, timeout=15)
        return {
            'status_code': response.status_code,
            'api_key_length': len(ALPHA_VANTAGE_KEY) if ALPHA_VANTAGE_KEY else 0,
            'response': response.json()
        }
    except Exception as e:
        return {'error': str(e)}


@app.get('/api/test-edgar/{ticker}')
async def test_edgar(ticker: str):
    """Debug endpoint to test SEC EDGAR API"""
    import requests

    ticker = ticker.upper()
    results = {
        'ticker': ticker,
        'massive_key_configured': bool(MASSIVE_API_KEY),
    }

    try:
        # Test ticker mapping
        fetcher = EdgarDataFetcher(ticker)
        cik = fetcher._get_cik()
        results['cik'] = cik
        results['ticker_mapping_loaded'] = bool(EdgarDataFetcher._ticker_to_cik_cache)
        results['mapping_count'] = len(EdgarDataFetcher._ticker_to_cik_cache) if EdgarDataFetcher._ticker_to_cik_cache else 0

        if not cik:
            results['error'] = 'Could not find CIK for ticker'
            return results

        # Test company facts endpoint
        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        headers = {'User-Agent': 'AutoAnalyst contact@autoanalyst.app'}
        facts_response = requests.get(facts_url, headers=headers, timeout=30)
        results['facts_status'] = facts_response.status_code
        results['facts_size'] = len(facts_response.text) if facts_response.status_code == 200 else 0

        # Test fetching financials
        fetcher.fetch_quote_data()
        results['company_name'] = fetcher.info.get('longName')

        financials = fetcher.fetch_financials()
        results['financials_rows'] = list(financials.index) if not financials.empty else []
        results['financials_cols'] = len(financials.columns) if not financials.empty else 0

        balance_sheet = fetcher.fetch_balance_sheet()
        results['balance_sheet_rows'] = list(balance_sheet.index) if not balance_sheet.empty else []

        cash_flow = fetcher.fetch_cash_flow()
        results['cash_flow_rows'] = list(cash_flow.index) if not cash_flow.empty else []

        # Test price fetching
        price = fetcher.fetch_current_price()
        results['price'] = price
        results['price_source'] = 'yahoo' if price and not MASSIVE_API_KEY else ('massive' if price else 'none')

        results['success'] = True

    except Exception as e:
        results['error'] = str(e)
        results['success'] = False

    return results


@app.get('/api/test-analysis/{ticker}')
async def test_analysis(ticker: str):
    """Debug endpoint to test analysis step by step"""
    ticker = ticker.upper()
    results = {'ticker': ticker, 'steps': {}}

    try:
        # Step 1: DCF (this fetches stock data)
        try:
            dcf = DCFValuation(ticker)
            dcf_result = dcf.perform_full_dcf_analysis()
            results['steps']['dcf'] = 'success'
            results['dcf_value'] = dcf_result.get('valuation', {}).get('intrinsic_value_per_share')
            results['shares_outstanding'] = dcf.info.get('sharesOutstanding')
        except Exception as e:
            results['steps']['dcf'] = f'failed: {str(e)}'
            import traceback
            results['dcf_traceback'] = traceback.format_exc()[-500:]

        # Step 2: Revenue Forecast (only if DCF succeeded)
        if 'dcf' in results['steps'] and results['steps']['dcf'] == 'success':
            try:
                rev = RevenueForecaster(ticker, dcf.stock, dcf.info)
                rev_result = rev.perform_full_revenue_analysis()
                results['steps']['revenue'] = 'success'
                results['revenue_cagr'] = rev_result.get('historical_cagr')
            except Exception as e:
                results['steps']['revenue'] = f'failed: {str(e)}'

        # Step 3: Comparable Companies (only if DCF succeeded)
        if 'dcf' in results['steps'] and results['steps']['dcf'] == 'success':
            try:
                comps = ComparableCompanyAnalysis(ticker, dcf.stock, dcf.info, dcf.current_price)
                comps_result = comps.perform_full_comps_analysis()
                results['steps']['comps'] = 'success'
                results['peers'] = [p.get('ticker') for p in comps_result.get('peers', [])]
            except Exception as e:
                results['steps']['comps'] = f'failed: {str(e)}'

        results['success'] = True

    except Exception as e:
        results['error'] = str(e)
        results['success'] = False

    return results


@app.get('/api/test-fetch/{ticker}')
async def test_fetch(ticker: str):
    """Debug endpoint to test fetch_stock_data function"""
    ticker = ticker.upper()
    results = {'ticker': ticker}

    try:
        stock, info, price = fetch_stock_data(ticker)

        results['success'] = True
        results['price'] = price
        results['data_source'] = type(stock).__name__
        results['info_keys'] = list(info.keys())
        results['company_name'] = info.get('longName')
        results['sector'] = info.get('sector')

        # Check financials
        results['financials_empty'] = stock.financials.empty
        results['financials_rows'] = list(stock.financials.index) if not stock.financials.empty else []
        results['financials_cols'] = len(stock.financials.columns) if not stock.financials.empty else 0

        # Check balance sheet
        results['balance_sheet_empty'] = stock.balance_sheet.empty
        results['balance_sheet_rows'] = list(stock.balance_sheet.index) if not stock.balance_sheet.empty else []

        # Check cash flow
        results['cash_flow_empty'] = stock.cash_flow.empty
        results['cash_flow_rows'] = list(stock.cash_flow.index) if not stock.cash_flow.empty else []

        # Sample data if available
        if not stock.financials.empty and 'Total Revenue' in stock.financials.index:
            results['sample_revenue'] = stock.financials.loc['Total Revenue'].iloc[0]
            results['financials_columns'] = list(stock.financials.columns)[:5]  # Show first 5 date columns

    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
        results['error_type'] = type(e).__name__

    return results


if __name__ == '__main__':
    import uvicorn

    # Get port from environment variable (for Render.com) or default to 8000
    port = int(os.environ.get('PORT', 8000))

    logger.info(f"Starting FastAPI server on port {port}")
    uvicorn.run("api_server:app", host='0.0.0.0', port=port, log_level="info", reload=False)
