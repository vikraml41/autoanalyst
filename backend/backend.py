"""
Advanced Stock Analyzer - Individual Stock Analysis
Combines DCF Valuation, Revenue Forecasting, and Comparable Company Analysis
With ML-powered synthesis for hedge fund-style insights
"""

import sys
import warnings
import os
import pandas as pd
import numpy as np
import requests
import re
import json
from bs4 import BeautifulSoup
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import logging
import time

warnings.filterwarnings('ignore')

# API Keys from environment variables
FMP_API_KEY = os.environ.get('FMP_API_KEY', 'demo')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ YAHOO FINANCE SCRAPER ============

class DataFetchError(Exception):
    """Raised when stock data cannot be fetched"""
    pass


class YahooFinanceScraper:
    """Direct scraper for Yahoo Finance using their JSON API endpoints"""

    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.info = {}
        self.financials = pd.DataFrame()
        self.quarterly_financials = pd.DataFrame()  # For compatibility with RevenueForecaster
        self.balance_sheet = pd.DataFrame()
        self.cash_flow = pd.DataFrame()
        self.current_price = None

    def _fetch_json(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        """Fetch JSON from Yahoo Finance API"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching JSON (attempt {attempt + 1}): {url}")
                response = self.session.get(url, timeout=20)
                logger.info(f"Response status: {response.status_code}, length: {len(response.text)}")

                if response.status_code == 200:
                    try:
                        return response.json()
                    except json.JSONDecodeError as je:
                        logger.error(f"JSON decode error: {je}. Response text: {response.text[:500]}")
                elif response.status_code == 429:
                    logger.warning(f"Rate limited (429). Waiting before retry...")
                    time.sleep(5)
                else:
                    logger.warning(f"Got status {response.status_code}. Response: {response.text[:300]}")
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
            except requests.exceptions.ConnectionError as ce:
                logger.error(f"Connection error on attempt {attempt + 1}: {ce}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1} for {url}: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        return None

    def _fetch_page(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Fetch HTML page"""
        for attempt in range(max_retries):
            try:
                logger.info(f"Fetching page (attempt {attempt + 1}): {url}")
                response = self.session.get(url, timeout=20)
                logger.info(f"Response status: {response.status_code}, length: {len(response.text)}")

                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:
                    logger.warning(f"Rate limited (429). Waiting before retry...")
                    time.sleep(5)
                else:
                    logger.warning(f"Got status {response.status_code}. Response preview: {response.text[:200]}")
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
            except requests.exceptions.ConnectionError as ce:
                logger.error(f"Connection error on attempt {attempt + 1}: {ce}")
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1} for {url}: {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        return None

    def _get_crumb(self) -> Optional[str]:
        """Get Yahoo Finance crumb for API authentication"""
        try:
            logger.info("Fetching Yahoo Finance crumb...")
            # First visit the main page to get cookies
            main_resp = self.session.get("https://finance.yahoo.com", timeout=15)
            logger.info(f"Main page status: {main_resp.status_code}, cookies: {len(self.session.cookies)}")

            # Get crumb
            crumb_url = "https://query1.finance.yahoo.com/v1/test/getcrumb"
            response = self.session.get(crumb_url, timeout=15)
            logger.info(f"Crumb response status: {response.status_code}")

            if response.status_code == 200:
                crumb = response.text
                logger.info(f"Got crumb: {crumb[:10]}..." if len(crumb) > 10 else f"Got crumb: {crumb}")
                return crumb
            else:
                logger.warning(f"Failed to get crumb. Status: {response.status_code}")
        except Exception as e:
            logger.error(f"Error getting crumb: {type(e).__name__}: {e}")
        return None

    def fetch_quote_data(self) -> Dict:
        """Fetch current price and basic info using Yahoo Finance chart API"""
        logger.info(f"=== Starting fetch_quote_data for {self.ticker} ===")

        # Try Method 1: Chart API (primary)
        try:
            data = self._try_chart_api()
            if data:
                return data
        except Exception as e:
            logger.warning(f"Chart API failed: {e}")

        # Try Method 2: Quote page scraping (fallback)
        try:
            data = self._try_quote_page_scraping()
            if data:
                return data
        except Exception as e:
            logger.warning(f"Quote page scraping failed: {e}")

        # Try Method 3: quoteSummary API (second fallback)
        try:
            data = self._try_quote_summary_api()
            if data:
                return data
        except Exception as e:
            logger.warning(f"Quote summary API failed: {e}")

        raise DataFetchError(f"All methods failed to fetch data for {self.ticker}. Yahoo Finance may be blocking this server's IP.")

    def _try_chart_api(self) -> Optional[Dict]:
        """Try Yahoo Finance chart API"""
        logger.info(f"Trying chart API for {self.ticker}...")

        # Get crumb for authentication
        crumb = self._get_crumb()

        # Use the chart API
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{self.ticker}?interval=1d&range=5d"
        if crumb:
            url += f"&crumb={crumb}"

        data = self._fetch_json(url)
        if not data:
            return None

        result = data.get('chart', {}).get('result')
        if not result:
            error = data.get('chart', {}).get('error', {})
            logger.warning(f"Chart API error: {error}")
            return None

        meta = result[0].get('meta', {})
        price = meta.get('regularMarketPrice')

        if not price:
            logger.warning(f"No price in chart API response")
            return None

        self.current_price = price
        logger.info(f"Chart API: Found price for {self.ticker}: ${price}")

        # Build info dict from meta
        self.info = {
            'currentPrice': price,
            'longName': meta.get('longName') or meta.get('shortName') or self.ticker,
            'shortName': meta.get('shortName'),
            'symbol': meta.get('symbol'),
            'exchangeName': meta.get('exchangeName'),
            'currency': meta.get('currency'),
            'regularMarketPrice': price,
            'previousClose': meta.get('previousClose'),
            'chartPreviousClose': meta.get('chartPreviousClose'),
        }

        # Try to get additional info from the quote page
        self._enrich_from_page()

        return self.info

    def _try_quote_page_scraping(self) -> Optional[Dict]:
        """Try scraping price directly from Yahoo Finance quote page"""
        logger.info(f"Trying quote page scraping for {self.ticker}...")

        url = f"https://finance.yahoo.com/quote/{self.ticker}"
        html = self._fetch_page(url)
        if not html:
            return None

        try:
            # Look for price in the page JSON data
            import re

            # Try multiple patterns to find the price
            patterns = [
                r'"regularMarketPrice":\s*\{\s*"raw":\s*([\d.]+)',
                r'data-value="([\d.]+)"[^>]*data-field="regularMarketPrice"',
                r'"regularMarketPrice":\s*([\d.]+)',
                r'class="Fw\(b\)[^"]*"[^>]*>([\d,.]+)</fin-streamer>',
            ]

            price = None
            for pattern in patterns:
                match = re.search(pattern, html)
                if match:
                    price_str = match.group(1).replace(',', '')
                    price = float(price_str)
                    logger.info(f"Quote page scraping: Found price {price} with pattern")
                    break

            if not price:
                logger.warning("Could not extract price from quote page")
                return None

            self.current_price = price

            # Extract other info
            name_match = re.search(r'"longName":\s*"([^"]+)"', html)
            sector_match = re.search(r'"sector":\s*"([^"]+)"', html)
            industry_match = re.search(r'"industry":\s*"([^"]+)"', html)
            market_cap_match = re.search(r'"marketCap":\s*\{\s*"raw":\s*([\d.]+)', html)

            self.info = {
                'currentPrice': price,
                'regularMarketPrice': price,
                'longName': name_match.group(1) if name_match else self.ticker,
                'sector': sector_match.group(1) if sector_match else None,
                'industry': industry_match.group(1) if industry_match else None,
                'marketCap': float(market_cap_match.group(1)) if market_cap_match else None,
            }

            logger.info(f"Quote page scraping: Successfully extracted data for {self.ticker}")
            return self.info

        except Exception as e:
            logger.error(f"Error parsing quote page: {e}")
            return None

    def _try_quote_summary_api(self) -> Optional[Dict]:
        """Try Yahoo Finance quoteSummary API"""
        logger.info(f"Trying quoteSummary API for {self.ticker}...")

        crumb = self._get_crumb()
        modules = "price,summaryDetail,defaultKeyStatistics"
        url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{self.ticker}?modules={modules}"
        if crumb:
            url += f"&crumb={crumb}"

        data = self._fetch_json(url)
        if not data:
            return None

        result = data.get('quoteSummary', {}).get('result')
        if not result or len(result) == 0:
            logger.warning("No data in quoteSummary response")
            return None

        quote_data = result[0]
        price_info = quote_data.get('price', {})
        summary_detail = quote_data.get('summaryDetail', {})

        price = price_info.get('regularMarketPrice', {}).get('raw')
        if not price:
            logger.warning("No price in quoteSummary response")
            return None

        self.current_price = price
        logger.info(f"quoteSummary API: Found price for {self.ticker}: ${price}")

        self.info = {
            'currentPrice': price,
            'regularMarketPrice': price,
            'longName': price_info.get('longName') or price_info.get('shortName') or self.ticker,
            'shortName': price_info.get('shortName'),
            'marketCap': price_info.get('marketCap', {}).get('raw'),
            'trailingPE': summary_detail.get('trailingPE', {}).get('raw'),
        }

        self._enrich_from_page()
        return self.info

    def _enrich_from_page(self):
        """Try to get sector, industry, market cap from the quote page"""
        try:
            url = f"https://finance.yahoo.com/quote/{self.ticker}"
            html = self._fetch_page(url)
            if not html:
                return

            # Extract data from embedded JSON in the page
            # Look for sector/industry in the page content
            sector_match = re.search(r'"sector":\s*"([^"]+)"', html)
            if sector_match:
                self.info['sector'] = sector_match.group(1)

            industry_match = re.search(r'"industry":\s*"([^"]+)"', html)
            if industry_match:
                self.info['industry'] = industry_match.group(1)

            market_cap_match = re.search(r'"marketCap":\s*\{\s*"raw":\s*([\d.]+)', html)
            if market_cap_match:
                self.info['marketCap'] = float(market_cap_match.group(1))

            shares_match = re.search(r'"sharesOutstanding":\s*\{\s*"raw":\s*([\d.]+)', html)
            if shares_match:
                self.info['sharesOutstanding'] = float(shares_match.group(1))

            pe_match = re.search(r'"trailingPE":\s*\{\s*"raw":\s*([\d.]+)', html)
            if pe_match:
                self.info['trailingPE'] = float(pe_match.group(1))

            ev_match = re.search(r'"enterpriseValue":\s*\{\s*"raw":\s*([\d.]+)', html)
            if ev_match:
                self.info['enterpriseValue'] = float(ev_match.group(1))

            summary_match = re.search(r'"longBusinessSummary":\s*"([^"]{100,500})', html)
            if summary_match:
                self.info['longBusinessSummary'] = summary_match.group(1).replace('\\n', ' ')

        except Exception as e:
            logger.warning(f"Could not enrich data from page: {e}")

    def fetch_financials(self) -> pd.DataFrame:
        """Fetch income statement data from the page"""
        return self._fetch_financial_data('financials', 'incomeStatementHistory')

    def fetch_balance_sheet(self) -> pd.DataFrame:
        """Fetch balance sheet data"""
        return self._fetch_financial_data('balance-sheet', 'balanceSheetHistory')

    def fetch_cash_flow(self) -> pd.DataFrame:
        """Fetch cash flow data"""
        return self._fetch_financial_data('cash-flow', 'cashflowStatementHistory')

    def _fetch_financial_data(self, page_type: str, json_key: str) -> pd.DataFrame:
        """Fetch financial data from Yahoo Finance page"""
        url = f"https://finance.yahoo.com/quote/{self.ticker}/{page_type}"
        logger.info(f"Scraping {page_type} from {url}")

        html = self._fetch_page(url)
        if not html:
            return pd.DataFrame()

        try:
            # Look for the financial data in the page's JSON
            pattern = rf'"{json_key}":\s*\{{\s*"{json_key}":\s*(\[.*?\])\s*\}}'
            match = re.search(pattern, html, re.DOTALL)
            if match:
                statements = json.loads(match.group(1))
                return self._json_to_dataframe(statements)

            # Alternative pattern
            pattern2 = rf'"{json_key}":\s*(\[.*?\])\s*[,\}}]'
            match2 = re.search(pattern2, html, re.DOTALL)
            if match2:
                statements = json.loads(match2.group(1))
                return self._json_to_dataframe(statements)

        except Exception as e:
            logger.warning(f"Error parsing {page_type}: {e}")

        return pd.DataFrame()

    def _json_to_dataframe(self, statements: List[Dict]) -> pd.DataFrame:
        """Convert Yahoo Finance JSON statements to DataFrame"""
        if not statements:
            return pd.DataFrame()

        data = {}
        for stmt in statements:
            date = stmt.get('endDate', {}).get('fmt', 'Unknown')
            for key, value in stmt.items():
                if isinstance(value, dict) and 'raw' in value:
                    if key not in data:
                        data[key] = {}
                    data[key][date] = value['raw']

        if data:
            df = pd.DataFrame(data).T
            return df
        return pd.DataFrame()

    def fetch_all_data(self) -> Tuple['YahooFinanceScraper', Dict, float]:
        """Fetch all data and return in format compatible with existing code"""
        self.fetch_quote_data()
        self.financials = self.fetch_financials()
        self.balance_sheet = self.fetch_balance_sheet()
        self.cash_flow = self.fetch_cash_flow()

        return self, self.info, self.current_price


# ============ FINANCIAL MODELING PREP API ============

class FMPDataFetcher:
    """Fetch stock data from Financial Modeling Prep API - cloud-friendly alternative"""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.api_key = FMP_API_KEY
        self.info = {}
        self.financials = pd.DataFrame()
        self.quarterly_financials = pd.DataFrame()  # For compatibility with RevenueForecaster
        self.balance_sheet = pd.DataFrame()
        self.cash_flow = pd.DataFrame()
        self.current_price = None

    def _fetch_json(self, endpoint: str) -> Optional[Dict]:
        """Fetch JSON from FMP API"""
        # Handle endpoints that already have query params
        separator = "&" if "?" in endpoint else "?"
        url = f"{self.BASE_URL}/{endpoint}{separator}apikey={self.api_key}"
        try:
            logger.info(f"FMP API: Fetching {url}")
            response = requests.get(url, timeout=15)
            logger.info(f"FMP API: Response status {response.status_code}, length: {len(response.text)}")

            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data
                elif isinstance(data, dict):
                    return data
                else:
                    logger.warning(f"FMP API: Empty response for endpoint")
            elif response.status_code == 401:
                logger.error(f"FMP API: Invalid API key (401)")
            elif response.status_code == 403:
                logger.error(f"FMP API: Access forbidden (403) - check API key permissions")
            elif response.status_code == 404:
                logger.warning(f"FMP API: Endpoint not found (404)")
            else:
                logger.warning(f"FMP API: Got status {response.status_code}. Response: {response.text[:200]}")
        except Exception as e:
            logger.error(f"FMP API error: {e}")
        return None

    def fetch_quote_data(self) -> Dict:
        """Fetch current price and company info"""
        logger.info(f"=== FMP API: Fetching quote data for {self.ticker} ===")

        # Get real-time quote
        quote_data = self._fetch_json(f"quote/{self.ticker}")
        if not quote_data or len(quote_data) == 0:
            raise DataFetchError(f"FMP API: No quote data for {self.ticker}")

        quote = quote_data[0]
        price = quote.get('price')

        if not price:
            raise DataFetchError(f"FMP API: No price for {self.ticker}")

        self.current_price = price
        logger.info(f"FMP API: Found price for {self.ticker}: ${price}")

        # Get company profile for additional info
        profile_data = self._fetch_json(f"profile/{self.ticker}")
        profile = profile_data[0] if profile_data and len(profile_data) > 0 else {}

        self.info = {
            'currentPrice': price,
            'regularMarketPrice': price,
            'longName': profile.get('companyName') or quote.get('name') or self.ticker,
            'shortName': quote.get('name'),
            'sector': profile.get('sector'),
            'industry': profile.get('industry'),
            'marketCap': quote.get('marketCap'),
            'sharesOutstanding': quote.get('sharesOutstanding'),
            'trailingPE': quote.get('pe'),
            'priceToSalesTrailing12Months': profile.get('priceToSalesRatio'),
            'priceToBook': profile.get('priceToBook'),
            'enterpriseValue': profile.get('enterpriseValue'),
            'beta': profile.get('beta'),
            'previousClose': quote.get('previousClose'),
            'longBusinessSummary': profile.get('description', '')[:500] if profile.get('description') else None,
        }

        return self.info

    def fetch_financials(self) -> pd.DataFrame:
        """Fetch income statement data"""
        data = self._fetch_json(f"income-statement/{self.ticker}?limit=5")
        if not data:
            return pd.DataFrame()

        return self._to_dataframe(data, {
            'Total Revenue': 'revenue',
            'Net Income': 'netIncome',
            'Operating Income': 'operatingIncome',
            'EBIT': 'operatingIncome',
            'EBITDA': 'ebitda',
            'Interest Expense': 'interestExpense',
        })

    def fetch_balance_sheet(self) -> pd.DataFrame:
        """Fetch balance sheet data"""
        data = self._fetch_json(f"balance-sheet-statement/{self.ticker}?limit=5")
        if not data:
            return pd.DataFrame()

        return self._to_dataframe(data, {
            'Total Assets': 'totalAssets',
            'Total Liabilities': 'totalLiabilities',
            'Total Equity': 'totalStockholdersEquity',
            'Total Debt': 'totalDebt',
            'Cash And Cash Equivalents': 'cashAndCashEquivalents',
        })

    def fetch_cash_flow(self) -> pd.DataFrame:
        """Fetch cash flow data"""
        data = self._fetch_json(f"cash-flow-statement/{self.ticker}?limit=5")
        if not data:
            return pd.DataFrame()

        return self._to_dataframe(data, {
            'Operating Cash Flow': 'operatingCashFlow',
            'Capital Expenditure': 'capitalExpenditure',
            'Free Cash Flow': 'freeCashFlow',
            'Depreciation And Amortization': 'depreciationAndAmortization',
        })

    def _to_dataframe(self, data: List[Dict], field_mapping: Dict[str, str]) -> pd.DataFrame:
        """Convert FMP API data to DataFrame format compatible with existing code"""
        result = {}
        for row_name, api_field in field_mapping.items():
            row_data = {}
            for item in data:
                date = item.get('date', 'Unknown')
                value = item.get(api_field)
                if value is not None:
                    row_data[date] = value
            if row_data:
                result[row_name] = row_data

        if result:
            return pd.DataFrame(result).T
        return pd.DataFrame()

    def fetch_quarterly_financials(self) -> pd.DataFrame:
        """Fetch quarterly income statement data"""
        data = self._fetch_json(f"income-statement/{self.ticker}?period=quarter&limit=8")
        if not data:
            return pd.DataFrame()

        return self._to_dataframe(data, {
            'Total Revenue': 'revenue',
            'Net Income': 'netIncome',
            'Operating Income': 'operatingIncome',
        })

    def fetch_all_data(self) -> Tuple['FMPDataFetcher', Dict, float]:
        """Fetch all data and return in format compatible with existing code"""
        self.fetch_quote_data()
        self.financials = self.fetch_financials()
        self.quarterly_financials = self.fetch_quarterly_financials()
        self.balance_sheet = self.fetch_balance_sheet()
        self.cash_flow = self.fetch_cash_flow()

        logger.info(f"FMP API: Fetched all data - financials rows: {len(self.financials)}, cash_flow rows: {len(self.cash_flow)}")
        return self, self.info, self.current_price


def fetch_stock_data(ticker: str, max_retries: int = 3) -> Tuple[Any, Dict, float]:
    """
    Fetch stock data using Financial Modeling Prep API.
    Returns (data_object, info_dict, current_price).
    Raises DataFetchError if data cannot be retrieved.
    """
    logger.info(f"=== Fetching data for {ticker} using FMP API ===")

    if not FMP_API_KEY or FMP_API_KEY == 'demo':
        raise DataFetchError("FMP_API_KEY environment variable not set. Please add your Financial Modeling Prep API key.")

    try:
        fetcher = FMPDataFetcher(ticker)
        return fetcher.fetch_all_data()
    except DataFetchError:
        raise
    except Exception as e:
        raise DataFetchError(f"Failed to fetch data for {ticker}: {e}")


class DCFValuation:
    """
    Discounted Cash Flow Valuation following the complete 6-step methodology
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.stock = None
        self.info = {}
        self.current_price = None
        self.financials = pd.DataFrame()
        self.balance_sheet = pd.DataFrame()
        self.cash_flow = pd.DataFrame()
        self.analysis_results = {}
        self.data_error = None

    def step1_understand_business(self) -> Dict:
        """Step 1: Gather company information and understand the business"""
        try:
            logger.info(f"DCF Step 1: Understanding {self.ticker} business...")

            # Fetch stock data - will raise DataFetchError if it fails
            self.stock, self.info, self.current_price = fetch_stock_data(self.ticker)

            # Get financial data from scraper
            self.financials = self.stock.financials
            self.balance_sheet = self.stock.balance_sheet
            self.cash_flow = self.stock.cash_flow

            # No fallbacks - use actual data or None
            business_info = {
                'company_name': self.info.get('longName') or self.info.get('shortName') or self.ticker,
                'sector': self.info.get('sector'),
                'industry': self.info.get('industry'),
                'market_cap': self.info.get('marketCap'),
                'current_price': self.current_price,
                'shares_outstanding': self.info.get('sharesOutstanding'),
                'description': (self.info.get('longBusinessSummary') or '')[:200] if self.info.get('longBusinessSummary') else None
            }

            logger.info(f"  Company: {business_info['company_name']}")
            logger.info(f"  Sector: {business_info['sector']}")
            logger.info(f"  Current Price: ${self.current_price}")

            return business_info

        except DataFetchError as e:
            self.data_error = str(e)
            logger.error(f"Data fetch error in DCF Step 1: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in DCF Step 1: {e}")
            raise DataFetchError(f"Failed to analyze {self.ticker}: {e}")

    def step2_forecast_cash_flows(self, forecast_years: int = 5) -> Dict:
        """Step 2: Forecast future cash flows using multiple methods"""
        try:
            logger.info(f"DCF Step 2: Forecasting {forecast_years} years of cash flows...")

            # Get historical cash flows
            if self.cash_flow.empty:
                logger.warning("No cash flow data available")
                return {}

            # Method 1: Free Cash Flow to Firm (FCFF) - Unlevered
            fcff_forecasts = self._forecast_fcff(forecast_years)

            # Method 2: Free Cash Flow to Equity (FCFE) - Levered
            fcfe_forecasts = self._forecast_fcfe(forecast_years)

            # Method 3: Simple FCF (Operating Cash Flow - CapEx)
            simple_fcf_forecasts = self._forecast_simple_fcf(forecast_years)

            results = {
                'fcff': fcff_forecasts,
                'fcfe': fcfe_forecasts,
                'simple_fcf': simple_fcf_forecasts,
                'forecast_years': forecast_years
            }

            logger.info(f"  Forecasted cash flows for {forecast_years} years")

            return results

        except Exception as e:
            logger.error(f"Error in DCF Step 2: {e}")
            return {}

    def _forecast_fcff(self, years: int) -> List[float]:
        """Calculate FCFF: EBIT × (1 - Tax Rate) + D&A - CapEx - ΔNWC"""
        try:
            # Get historical data
            if 'EBIT' in self.financials.index:
                ebit = self.financials.loc['EBIT'].iloc[0]
            elif 'Operating Income' in self.financials.index:
                ebit = self.financials.loc['Operating Income'].iloc[0]
            else:
                # Approximate EBIT
                revenue = self.financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in self.financials.index else 0
                ebit = revenue * 0.15  # Conservative 15% operating margin

            # Tax rate
            tax_rate = self.info.get('effectiveTaxRate', 0.21)

            # D&A
            if 'Depreciation And Amortization' in self.cash_flow.index:
                da = abs(self.cash_flow.loc['Depreciation And Amortization'].iloc[0])
            else:
                da = ebit * 0.05  # Estimate as 5% of EBIT

            # CapEx
            if 'Capital Expenditure' in self.cash_flow.index:
                capex = abs(self.cash_flow.loc['Capital Expenditure'].iloc[0])
            else:
                capex = da * 1.2  # Estimate CapEx as 120% of D&A

            # Change in NWC (simplified)
            nwc_change = ebit * 0.02  # Estimate as 2% of EBIT

            # Calculate current FCFF
            current_fcff = ebit * (1 - tax_rate) + da - capex - nwc_change

            # Forecast with growth rate
            growth_rate = self.info.get('revenueGrowth', 0.05)  # Default 5%
            growth_rate = max(min(growth_rate, 0.30), -0.10)  # Cap between -10% and 30%

            # Declining growth rate over time (more conservative)
            forecasts = []
            for year in range(1, years + 1):
                # Growth declines by 20% each year
                adjusted_growth = growth_rate * (0.8 ** (year - 1))
                fcff = current_fcff * ((1 + adjusted_growth) ** year)
                forecasts.append(fcff)

            return forecasts

        except Exception as e:
            logger.error(f"Error forecasting FCFF: {e}")
            return [0] * years

    def _forecast_fcfe(self, years: int) -> List[float]:
        """Calculate FCFE: Net Income + D&A - CapEx - ΔNWC + Net Borrowing"""
        try:
            # Get net income
            if 'Net Income' in self.financials.index:
                net_income = self.financials.loc['Net Income'].iloc[0]
            else:
                net_income = 0

            # D&A
            if 'Depreciation And Amortization' in self.cash_flow.index:
                da = abs(self.cash_flow.loc['Depreciation And Amortization'].iloc[0])
            else:
                da = net_income * 0.05

            # CapEx
            if 'Capital Expenditure' in self.cash_flow.index:
                capex = abs(self.cash_flow.loc['Capital Expenditure'].iloc[0])
            else:
                capex = da * 1.2

            # NWC change
            nwc_change = net_income * 0.02

            # Net borrowing (simplified - assume stable)
            net_borrowing = 0

            current_fcfe = net_income + da - capex - nwc_change + net_borrowing

            # Forecast with growth
            growth_rate = self.info.get('earningsGrowth', 0.05)
            growth_rate = max(min(growth_rate, 0.30), -0.10)

            forecasts = []
            for year in range(1, years + 1):
                adjusted_growth = growth_rate * (0.8 ** (year - 1))
                fcfe = current_fcfe * ((1 + adjusted_growth) ** year)
                forecasts.append(fcfe)

            return forecasts

        except Exception as e:
            logger.error(f"Error forecasting FCFE: {e}")
            return [0] * years

    def _forecast_simple_fcf(self, years: int) -> List[float]:
        """Calculate Simple FCF: Operating Cash Flow - CapEx"""
        try:
            # Get operating cash flow
            if 'Operating Cash Flow' in self.cash_flow.index:
                ocf = self.cash_flow.loc['Operating Cash Flow'].iloc[0]
            elif 'Total Cash From Operating Activities' in self.cash_flow.index:
                ocf = self.cash_flow.loc['Total Cash From Operating Activities'].iloc[0]
            else:
                ocf = 0

            # CapEx
            if 'Capital Expenditure' in self.cash_flow.index:
                capex = abs(self.cash_flow.loc['Capital Expenditure'].iloc[0])
            else:
                capex = ocf * 0.15

            current_fcf = ocf - capex

            # Forecast
            growth_rate = self.info.get('revenueGrowth', 0.05)
            growth_rate = max(min(growth_rate, 0.30), -0.10)

            forecasts = []
            for year in range(1, years + 1):
                adjusted_growth = growth_rate * (0.8 ** (year - 1))
                fcf = current_fcf * ((1 + adjusted_growth) ** year)
                forecasts.append(fcf)

            return forecasts

        except Exception as e:
            logger.error(f"Error forecasting Simple FCF: {e}")
            return [0] * years

    def step3_estimate_discount_rate(self) -> Dict:
        """Step 3: Calculate WACC (Weighted Average Cost of Capital)"""
        try:
            logger.info("DCF Step 3: Estimating discount rate (WACC)...")

            # Risk-free rate (10-year Treasury)
            rf = self._get_risk_free_rate()

            # Equity risk premium
            equity_risk_premium = 0.055  # ~5.5% historical average

            # Beta
            beta = self.info.get('beta', 1.0)
            if beta is None or beta <= 0:
                beta = 1.0

            # Cost of Equity using CAPM: rf + β × (rm - rf)
            cost_of_equity = rf + beta * equity_risk_premium

            # Cost of Debt
            cost_of_debt = self._calculate_cost_of_debt()

            # Tax rate
            tax_rate = self.info.get('effectiveTaxRate', 0.21)

            # Market values for weights
            market_cap = self.info.get('marketCap', 0)
            total_debt = self.info.get('totalDebt', 0)

            if total_debt is None:
                total_debt = 0

            total_value = market_cap + total_debt

            if total_value > 0:
                weight_equity = market_cap / total_value
                weight_debt = total_debt / total_value
            else:
                weight_equity = 1.0
                weight_debt = 0.0

            # WACC = (E/V × Cost of Equity) + (D/V × Cost of Debt × (1 - Tax Rate))
            wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt * (1 - tax_rate))

            results = {
                'wacc': wacc,
                'cost_of_equity': cost_of_equity,
                'cost_of_debt': cost_of_debt,
                'risk_free_rate': rf,
                'beta': beta,
                'equity_risk_premium': equity_risk_premium,
                'weight_equity': weight_equity,
                'weight_debt': weight_debt,
                'tax_rate': tax_rate
            }

            logger.info(f"  WACC: {wacc:.2%}")
            logger.info(f"  Cost of Equity: {cost_of_equity:.2%}")
            logger.info(f"  Cost of Debt: {cost_of_debt:.2%}")

            return results

        except Exception as e:
            logger.error(f"Error in DCF Step 3: {e}")
            return {'wacc': 0.10}  # Default 10%

    def _get_risk_free_rate(self) -> float:
        """Get current 10-year Treasury rate"""
        try:
            # Scrape treasury rate from Yahoo Finance
            scraper = YahooFinanceScraper("^TNX")
            html = scraper._fetch_page("https://finance.yahoo.com/quote/%5ETNX")
            if html:
                import re
                match = re.search(r'data-value="([\d.]+)"[^>]*data-field="regularMarketPrice"', html)
                if match:
                    rate = float(match.group(1)) / 100  # Convert to decimal
                    logger.info(f"Risk-free rate: {rate:.4f}")
                    return rate
            return 0.045  # Default 4.5%
        except Exception as e:
            logger.warning(f"Could not fetch risk-free rate: {e}")
            return 0.045

    def _calculate_cost_of_debt(self) -> float:
        """Calculate cost of debt using synthetic rating method"""
        try:
            # Get interest expense and total debt
            if 'Interest Expense' in self.financials.index:
                interest_expense = abs(self.financials.loc['Interest Expense'].iloc[0])
            else:
                interest_expense = 0

            total_debt = self.info.get('totalDebt', 0)

            if total_debt and total_debt > 0 and interest_expense > 0:
                # Simple method: Interest Expense / Total Debt
                cost_of_debt = interest_expense / total_debt
            else:
                # Use default spread over risk-free
                rf = self._get_risk_free_rate()
                cost_of_debt = rf + 0.02  # 2% spread

            return min(cost_of_debt, 0.15)  # Cap at 15%

        except Exception as e:
            logger.error(f"Error calculating cost of debt: {e}")
            return 0.05

    def step4_estimate_terminal_value(self, final_year_cf: float, discount_rate: float) -> Dict:
        """Step 4: Calculate terminal value using both PGM and EMM"""
        try:
            logger.info("DCF Step 4: Estimating terminal value...")

            # Method 1: Perpetual Growth Method (Gordon Growth Model)
            perpetual_growth_rate = 0.025  # Conservative 2.5% long-term growth
            terminal_value_pgm = (final_year_cf * (1 + perpetual_growth_rate)) / (discount_rate - perpetual_growth_rate)

            # Method 2: Exit Multiple Method
            # Use EV/EBITDA multiple
            ebitda_multiple = self._get_industry_ebitda_multiple()

            # Estimate terminal year EBITDA
            if 'EBITDA' in self.financials.index:
                current_ebitda = self.financials.loc['EBITDA'].iloc[0]
                # Grow it to terminal year
                growth_rate = self.info.get('revenueGrowth', 0.05)
                terminal_ebitda = current_ebitda * ((1 + growth_rate * 0.5) ** 5)  # Reduced growth
            else:
                # Approximate from cash flow
                terminal_ebitda = final_year_cf * 2.5

            terminal_value_emm = terminal_ebitda * ebitda_multiple

            # Average both methods
            terminal_value = (terminal_value_pgm + terminal_value_emm) / 2

            results = {
                'terminal_value': terminal_value,
                'terminal_value_pgm': terminal_value_pgm,
                'terminal_value_emm': terminal_value_emm,
                'perpetual_growth_rate': perpetual_growth_rate,
                'ebitda_multiple': ebitda_multiple
            }

            logger.info(f"  Terminal Value (PGM): ${terminal_value_pgm:,.0f}")
            logger.info(f"  Terminal Value (EMM): ${terminal_value_emm:,.0f}")
            logger.info(f"  Terminal Value (Average): ${terminal_value:,.0f}")

            return results

        except Exception as e:
            logger.error(f"Error in DCF Step 4: {e}")
            return {'terminal_value': final_year_cf * 10}

    def _get_industry_ebitda_multiple(self) -> float:
        """Get industry-appropriate EV/EBITDA multiple"""
        sector = self.info.get('sector', 'Technology')

        # Industry average EV/EBITDA multiples
        multiples = {
            'Technology': 18.0,
            'Healthcare': 16.0,
            'Financial Services': 12.0,
            'Consumer Cyclical': 14.0,
            'Consumer Defensive': 13.0,
            'Industrials': 13.0,
            'Energy': 10.0,
            'Utilities': 11.0,
            'Real Estate': 17.0,
            'Basic Materials': 11.0,
            'Communication Services': 15.0
        }

        return multiples.get(sector, 14.0)

    def step5_calculate_present_value(self, cash_flows: List[float], terminal_value: float,
                                     discount_rate: float) -> Dict:
        """Step 5: Calculate present value and determine valuation"""
        try:
            logger.info("DCF Step 5: Calculating present value...")

            # Discount cash flows using mid-year convention
            pv_cash_flows = []
            for year, cf in enumerate(cash_flows, start=1):
                discount_factor = 1 / ((1 + discount_rate) ** (year - 0.5))
                pv_cf = cf * discount_factor
                pv_cash_flows.append(pv_cf)

            # Discount terminal value
            terminal_year = len(cash_flows)
            terminal_discount_factor = 1 / ((1 + discount_rate) ** terminal_year)
            pv_terminal_value = terminal_value * terminal_discount_factor

            # Enterprise Value = Sum of PV of cash flows + PV of terminal value
            enterprise_value = sum(pv_cash_flows) + pv_terminal_value

            # Bridge to Equity Value
            cash = self.info.get('totalCash', 0) or 0
            debt = self.info.get('totalDebt', 0) or 0

            equity_value = enterprise_value + cash - debt

            # Per-share value
            shares_outstanding = self.info.get('sharesOutstanding', 1)
            intrinsic_value_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0

            # Current market price (stored from step1)
            current_price = self.current_price

            # Valuation metrics
            if current_price > 0:
                premium_discount = (intrinsic_value_per_share / current_price) - 1
            else:
                premium_discount = 0

            # Apply margin of safety (20%)
            margin_of_safety = 0.20
            buy_price = intrinsic_value_per_share * (1 - margin_of_safety)

            results = {
                'enterprise_value': enterprise_value,
                'equity_value': equity_value,
                'intrinsic_value_per_share': intrinsic_value_per_share,
                'current_price': current_price,
                'premium_discount': premium_discount,
                'buy_price': buy_price,
                'margin_of_safety': margin_of_safety,
                'pv_cash_flows': sum(pv_cash_flows),
                'pv_terminal_value': pv_terminal_value,
                'terminal_value_percentage': pv_terminal_value / enterprise_value if enterprise_value > 0 else 0
            }

            logger.info(f"  Enterprise Value: ${enterprise_value:,.0f}")
            logger.info(f"  Equity Value: ${equity_value:,.0f}")
            logger.info(f"  Intrinsic Value per Share: ${intrinsic_value_per_share:.2f}")
            logger.info(f"  Current Price: ${current_price:.2f}")
            logger.info(f"  Premium/(Discount): {premium_discount:.1%}")
            logger.info(f"  Buy Price (with MoS): ${buy_price:.2f}")

            return results

        except Exception as e:
            logger.error(f"Error in DCF Step 5: {e}")
            return {}

    def step6_sensitivity_analysis(self, base_case: Dict, cash_flows: List[float]) -> Dict:
        """Step 6: Perform sensitivity analysis on key variables"""
        try:
            logger.info("DCF Step 6: Performing sensitivity analysis...")

            base_wacc = base_case.get('wacc', 0.10)
            base_growth = base_case.get('perpetual_growth_rate', 0.025)
            final_cf = cash_flows[-1] if cash_flows else 0

            # Test WACC variations (+/- 1%)
            wacc_range = [base_wacc - 0.02, base_wacc - 0.01, base_wacc, base_wacc + 0.01, base_wacc + 0.02]

            # Test growth rate variations (+/- 0.5%)
            growth_range = [base_growth - 0.01, base_growth - 0.005, base_growth, base_growth + 0.005, base_growth + 0.01]

            # Create sensitivity matrix
            sensitivity_matrix = []
            for wacc in wacc_range:
                row = []
                for growth in growth_range:
                    if wacc <= growth:
                        row.append(None)  # Invalid combination
                    else:
                        tv = (final_cf * (1 + growth)) / (wacc - growth)
                        pv_result = self.step5_calculate_present_value(cash_flows, tv, wacc)
                        row.append(pv_result.get('intrinsic_value_per_share', 0))
                sensitivity_matrix.append(row)

            results = {
                'wacc_range': wacc_range,
                'growth_range': growth_range,
                'sensitivity_matrix': sensitivity_matrix,
                'base_intrinsic_value': base_case.get('intrinsic_value_per_share', 0)
            }

            logger.info("  Sensitivity analysis complete")

            return results

        except Exception as e:
            logger.error(f"Error in DCF Step 6: {e}")
            return {}

    def perform_full_dcf_analysis(self) -> Dict:
        """Execute complete 6-step DCF analysis"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting DCF Analysis for {self.ticker}")
            logger.info(f"{'='*60}\n")

            # Step 1: Understand Business
            business_info = self.step1_understand_business()

            # Step 2: Forecast Cash Flows
            cash_flow_forecasts = self.step2_forecast_cash_flows(forecast_years=5)

            # Use FCFF as primary method
            primary_cf = cash_flow_forecasts.get('fcff', [])

            if not primary_cf or all(cf == 0 for cf in primary_cf):
                # Fallback to simple FCF
                primary_cf = cash_flow_forecasts.get('simple_fcf', [])

            # Step 3: Estimate Discount Rate
            discount_info = self.step3_estimate_discount_rate()
            wacc = discount_info.get('wacc', 0.10)

            # Step 4: Estimate Terminal Value
            final_year_cf = primary_cf[-1] if primary_cf else 0
            terminal_value_info = self.step4_estimate_terminal_value(final_year_cf, wacc)
            terminal_value = terminal_value_info.get('terminal_value', 0)

            # Step 5: Calculate Present Value
            valuation_results = self.step5_calculate_present_value(primary_cf, terminal_value, wacc)

            # Step 6: Sensitivity Analysis
            sensitivity_results = self.step6_sensitivity_analysis(
                {**discount_info, **terminal_value_info, **valuation_results},
                primary_cf
            )

            # Compile complete results
            complete_results = {
                'ticker': self.ticker,
                'business_info': business_info,
                'cash_flows': {
                    'forecasts': primary_cf,
                    'all_methods': cash_flow_forecasts
                },
                'discount_rate': discount_info,
                'terminal_value': terminal_value_info,
                'valuation': valuation_results,
                'sensitivity': sensitivity_results,
                'recommendation': self._generate_dcf_recommendation(valuation_results)
            }

            logger.info(f"\n{'='*60}")
            logger.info(f"DCF Analysis Complete for {self.ticker}")
            logger.info(f"{'='*60}\n")

            return complete_results

        except Exception as e:
            logger.error(f"Error in full DCF analysis: {e}")
            return {}

    def _generate_dcf_recommendation(self, valuation: Dict) -> str:
        """Generate DCF-based recommendation"""
        intrinsic = valuation.get('intrinsic_value_per_share', 0)
        current = valuation.get('current_price', 0)
        buy_price = valuation.get('buy_price', 0)

        if current == 0:
            return "Unable to determine - no price data"

        if buy_price > current:
            upside = ((intrinsic / current) - 1) * 100
            return f"STRONG BUY - Undervalued by {upside:.1f}%. Buy below ${buy_price:.2f}"
        elif intrinsic > current:
            return f"BUY - Fairly valued to slightly undervalued. Target: ${intrinsic:.2f}"
        elif intrinsic > current * 0.9:
            return f"HOLD - Trading near fair value. Fair value: ${intrinsic:.2f}"
        else:
            overvalued = ((current / intrinsic) - 1) * 100
            return f"SELL - Overvalued by {overvalued:.1f}%. Fair value: ${intrinsic:.2f}"


# ============ REVENUE FORECASTING MODULE ============

class RevenueForecaster:
    """
    Advanced revenue forecasting using multiple methodologies
    """

    def __init__(self, ticker: str, stock: Any = None, info: Dict = None):
        self.ticker = ticker
        # Reuse stock object if provided, otherwise fetch new
        if stock is not None:
            self.stock = stock
            self.info = info or {}
        else:
            self.stock, self.info, _ = fetch_stock_data(ticker)
        self.historical_data = pd.DataFrame()

    def gather_revenue_data(self) -> pd.DataFrame:
        """Gather historical revenue data"""
        try:
            logger.info(f"\nRevenue Forecasting: Gathering data for {self.ticker}...")

            financials = self.stock.financials
            quarterly_financials = self.stock.quarterly_financials

            revenue_data = []

            # Annual data
            if not financials.empty and 'Total Revenue' in financials.index:
                annual_revenue = financials.loc['Total Revenue']
                for date, value in annual_revenue.items():
                    revenue_data.append({
                        'date': date,
                        'revenue': value,
                        'period': 'Annual'
                    })

            # Quarterly data
            if not quarterly_financials.empty and 'Total Revenue' in quarterly_financials.index:
                quarterly_revenue = quarterly_financials.loc['Total Revenue']
                for date, value in quarterly_revenue.items():
                    revenue_data.append({
                        'date': date,
                        'revenue': value,
                        'period': 'Quarterly'
                    })

            self.historical_data = pd.DataFrame(revenue_data)
            self.historical_data = self.historical_data.sort_values('date').reset_index(drop=True)

            logger.info(f"  Gathered {len(self.historical_data)} revenue data points")

            return self.historical_data

        except Exception as e:
            logger.error(f"Error gathering revenue data: {e}")
            return pd.DataFrame()

    def analyze_growth_trends(self) -> Dict:
        """Analyze historical revenue growth patterns"""
        try:
            if self.historical_data.empty:
                return {}

            # Calculate growth rates
            annual_data = self.historical_data[self.historical_data['period'] == 'Annual'].copy()

            if len(annual_data) < 2:
                return {}

            annual_data['growth_rate'] = annual_data['revenue'].pct_change()

            # Growth metrics
            avg_growth = annual_data['growth_rate'].mean()
            recent_growth = annual_data['growth_rate'].iloc[-1] if len(annual_data) > 0 else 0
            growth_volatility = annual_data['growth_rate'].std()

            # Calculate CAGR
            if len(annual_data) >= 2:
                years = len(annual_data) - 1
                first_rev = annual_data['revenue'].iloc[0]
                last_rev = annual_data['revenue'].iloc[-1]
                if first_rev > 0 and last_rev > 0:
                    cagr = (last_rev / first_rev) ** (1/years) - 1
                else:
                    cagr = avg_growth if not np.isnan(avg_growth) else 0
            else:
                cagr = avg_growth if not np.isnan(avg_growth) else 0

            results = {
                'average_growth': avg_growth,
                'recent_growth': recent_growth,
                'cagr': cagr,
                'growth_volatility': growth_volatility,
                'trend': 'Accelerating' if recent_growth > avg_growth else 'Decelerating'
            }

            logger.info(f"  CAGR: {cagr:.2%}")
            logger.info(f"  Recent Growth: {recent_growth:.2%}")
            logger.info(f"  Trend: {results['trend']}")

            return results

        except Exception as e:
            logger.error(f"Error analyzing growth trends: {e}")
            return {}

    def forecast_revenue(self, years: int = 5) -> Dict:
        """Forecast future revenue using multiple methods"""
        try:
            logger.info(f"Revenue Forecasting: Projecting {years} years ahead...")

            if self.historical_data.empty:
                self.gather_revenue_data()

            growth_analysis = self.analyze_growth_trends()

            # Get latest annual revenue
            annual_data = self.historical_data[self.historical_data['period'] == 'Annual']
            if annual_data.empty:
                return {}

            current_revenue = annual_data['revenue'].iloc[-1]

            # Method 1: Linear Regression
            linear_forecast = self._linear_regression_forecast(annual_data, years, current_revenue)

            # Method 2: Growth Rate Projection (Conservative)
            growth_forecast = self._growth_rate_forecast(growth_analysis, current_revenue, years)

            # Method 3: Industry Benchmark Adjustment
            industry_forecast = self._industry_adjusted_forecast(current_revenue, years)

            # Ensemble forecast (weighted average)
            ensemble_forecast = []
            for i in range(years):
                weighted_avg = (
                    linear_forecast[i] * 0.3 +
                    growth_forecast[i] * 0.4 +
                    industry_forecast[i] * 0.3
                )
                ensemble_forecast.append(weighted_avg)

            results = {
                'current_revenue': current_revenue,
                'forecasted_revenue': ensemble_forecast,
                'linear_regression': linear_forecast,
                'growth_projection': growth_forecast,
                'industry_adjusted': industry_forecast,
                'growth_analysis': growth_analysis,
                'forecast_years': years
            }

            logger.info(f"  Current Revenue: ${current_revenue:,.0f}")
            logger.info(f"  Year 1 Forecast: ${ensemble_forecast[0]:,.0f}")
            logger.info(f"  Year {years} Forecast: ${ensemble_forecast[-1]:,.0f}")

            return results

        except Exception as e:
            logger.error(f"Error forecasting revenue: {e}")
            return {}

    def _linear_regression_forecast(self, data: pd.DataFrame, years: int, current: float) -> List[float]:
        """Linear regression forecast"""
        try:
            if len(data) < 3:
                return [current * 1.05 ** i for i in range(1, years + 1)]

            X = np.arange(len(data)).reshape(-1, 1)
            y = data['revenue'].values

            model = LinearRegression()
            model.fit(X, y)

            # Predict future
            future_X = np.arange(len(data), len(data) + years).reshape(-1, 1)
            predictions = model.predict(future_X)

            return [max(p, current * 0.8) for p in predictions]  # Floor at 80% of current

        except Exception as e:
            logger.error(f"Linear regression error: {e}")
            return [current * 1.05 ** i for i in range(1, years + 1)]

    def _growth_rate_forecast(self, growth_analysis: Dict, current: float, years: int) -> List[float]:
        """Growth rate projection with declining growth"""
        try:
            cagr = growth_analysis.get('cagr', 0.05)
            recent_growth = growth_analysis.get('recent_growth', 0.05)

            # Use average of CAGR and recent growth
            base_growth = (cagr + recent_growth) / 2

            # Cap growth at reasonable levels
            base_growth = max(min(base_growth, 0.30), -0.05)

            forecasts = []
            for year in range(1, years + 1):
                # Growth declines over time (regression to mean)
                adjusted_growth = base_growth * (0.85 ** (year - 1))
                revenue = current * ((1 + adjusted_growth) ** year)
                forecasts.append(revenue)

            return forecasts

        except Exception as e:
            logger.error(f"Growth rate forecast error: {e}")
            return [current * 1.05 ** i for i in range(1, years + 1)]

    def _industry_adjusted_forecast(self, current: float, years: int) -> List[float]:
        """Industry benchmark-adjusted forecast"""
        try:
            info = self.stock.info
            sector = info.get('sector', 'Technology')

            # Industry growth rates (conservative estimates)
            industry_growth = {
                'Technology': 0.08,
                'Healthcare': 0.06,
                'Financial Services': 0.04,
                'Consumer Cyclical': 0.05,
                'Consumer Defensive': 0.03,
                'Industrials': 0.04,
                'Energy': 0.03,
                'Utilities': 0.02,
                'Real Estate': 0.04,
                'Basic Materials': 0.03,
                'Communication Services': 0.05
            }

            growth_rate = industry_growth.get(sector, 0.05)

            forecasts = []
            for year in range(1, years + 1):
                revenue = current * ((1 + growth_rate) ** year)
                forecasts.append(revenue)

            return forecasts

        except Exception as e:
            logger.error(f"Industry forecast error: {e}")
            return [current * 1.05 ** i for i in range(1, years + 1)]

    def perform_full_revenue_analysis(self) -> Dict:
        """Execute complete revenue forecasting analysis"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Revenue Forecast for {self.ticker}")
            logger.info(f"{'='*60}\n")

            # Gather data
            self.gather_revenue_data()

            # Analyze trends
            growth_analysis = self.analyze_growth_trends()

            # Forecast
            forecast_results = self.forecast_revenue(years=5)

            # Generate recommendation
            recommendation = self._generate_revenue_recommendation(forecast_results, growth_analysis)

            results = {
                'ticker': self.ticker,
                'historical_data': self.historical_data.to_dict('records'),
                'growth_analysis': growth_analysis,
                'forecast': forecast_results,
                'recommendation': recommendation
            }

            logger.info(f"\n{'='*60}")
            logger.info(f"Revenue Forecast Complete for {self.ticker}")
            logger.info(f"{'='*60}\n")

            return results

        except Exception as e:
            logger.error(f"Error in full revenue analysis: {e}")
            return {}

    def _generate_revenue_recommendation(self, forecast: Dict, growth: Dict) -> str:
        """Generate revenue-based recommendation"""
        try:
            cagr = growth.get('cagr', 0)
            trend = growth.get('trend', 'Unknown')

            if cagr > 0.15:
                return f"Strong Growth - {cagr:.1%} CAGR with {trend} trend. Revenue model supports bullish outlook."
            elif cagr > 0.08:
                return f"Moderate Growth - {cagr:.1%} CAGR with {trend} trend. Solid revenue trajectory."
            elif cagr > 0.03:
                return f"Slow Growth - {cagr:.1%} CAGR with {trend} trend. Stable but limited upside."
            elif cagr > 0:
                return f"Minimal Growth - {cagr:.1%} CAGR with {trend} trend. Mature company characteristics."
            else:
                return f"Declining Revenue - {cagr:.1%} CAGR. Significant concerns about business model."

        except:
            return "Unable to generate recommendation"


# ============ COMPARABLE COMPANY ANALYSIS MODULE ============

class ComparableCompanyAnalysis:
    """
    Comparable Company Analysis (Trading Comps) following 5-step methodology
    """

    def __init__(self, ticker: str, stock: Any = None, info: Dict = None, current_price: float = None):
        self.ticker = ticker
        # Reuse stock object if provided, otherwise fetch new
        if stock is not None:
            self.stock = stock
            self.info = info or {}
            self.current_price = current_price
        else:
            self.stock, self.info, self.current_price = fetch_stock_data(ticker)
        self.peer_group = []
        self.peer_data = {}

    def step1_compile_peer_group(self) -> List[str]:
        """Step 1: Identify and compile peer group of comparable companies"""
        try:
            logger.info(f"\nComps Step 1: Compiling peer group for {self.ticker}...")

            info = self.stock.info
            sector = info.get('sector', '')
            industry = info.get('industry', '')

            # Try to get recommended symbols from yfinance
            try:
                recommendations = self.stock.recommendations
                if recommendations is not None and not recommendations.empty:
                    # Look for analyst coverage
                    pass
            except:
                pass

            # Get industry peers (simplified - in production, use proper peer screening)
            peers = self._find_industry_peers(sector, industry)

            # Add target company
            self.peer_group = [self.ticker] + peers

            logger.info(f"  Identified {len(self.peer_group)} companies in peer group")
            logger.info(f"  Peers: {', '.join(peers[:5])}" + ("..." if len(peers) > 5 else ""))

            return self.peer_group

        except Exception as e:
            logger.error(f"Error in Comps Step 1: {e}")
            return [self.ticker]

    def _find_industry_peers(self, sector: str, industry: str) -> List[str]:
        """Find peer companies in same sector/industry"""
        # Predefined peer groups for major companies (simplified)
        known_peers = {
            'AAPL': ['MSFT', 'GOOGL', 'META', 'NVDA'],
            'MSFT': ['AAPL', 'GOOGL', 'ORCL', 'ADBE'],
            'GOOGL': ['META', 'AAPL', 'MSFT', 'AMZN'],
            'TSLA': ['F', 'GM', 'RIVN', 'LCID'],
            'AMZN': ['WMT', 'GOOGL', 'EBAY', 'SHOP'],
            'NVDA': ['AMD', 'INTC', 'QCOM', 'AVGO'],
            'JPM': ['BAC', 'WFC', 'C', 'GS'],
            'JNJ': ['PFE', 'UNH', 'ABBV', 'MRK'],
            'V': ['MA', 'AXP', 'PYPL', 'SQ'],
            'WMT': ['TGT', 'COST', 'AMZN', 'HD']
        }

        if self.ticker in known_peers:
            return known_peers[self.ticker]

        # Default: return some similar market cap companies in sector
        # In production, would query database or API for proper peer screening
        sector_leaders = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA'],
            'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST']
        }

        return sector_leaders.get(sector, ['SPY'])[:4]  # Return up to 4 peers

    def step2_industry_research(self) -> Dict:
        """Step 2: Conduct industry research and understand market trends"""
        try:
            logger.info("Comps Step 2: Conducting industry research...")

            info = self.stock.info

            research = {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'target_market_cap': info.get('marketCap', 0),
                'industry_characteristics': self._get_industry_characteristics(info.get('sector', ''))
            }

            logger.info(f"  Sector: {research['sector']}")
            logger.info(f"  Industry: {research['industry']}")

            return research

        except Exception as e:
            logger.error(f"Error in Comps Step 2: {e}")
            return {}

    def _get_industry_characteristics(self, sector: str) -> Dict:
        """Get key characteristics for industry valuation"""
        characteristics = {
            'Technology': {
                'key_multiples': ['P/E', 'EV/EBITDA', 'P/S', 'PEG'],
                'growth_focus': True,
                'typical_pe_range': (20, 35)
            },
            'Healthcare': {
                'key_multiples': ['P/E', 'EV/EBITDA', 'P/B'],
                'growth_focus': True,
                'typical_pe_range': (15, 25)
            },
            'Financial Services': {
                'key_multiples': ['P/E', 'P/B', 'P/TBV'],
                'growth_focus': False,
                'typical_pe_range': (10, 15)
            },
            'Consumer Cyclical': {
                'key_multiples': ['P/E', 'EV/EBITDA', 'P/S'],
                'growth_focus': True,
                'typical_pe_range': (15, 25)
            }
        }

        return characteristics.get(sector, {
            'key_multiples': ['P/E', 'EV/EBITDA'],
            'growth_focus': False,
            'typical_pe_range': (12, 20)
        })

    def step3_input_financial_data(self) -> Dict:
        """Step 3: Collect and normalize financial data for all peers"""
        try:
            logger.info("Comps Step 3: Collecting financial data for peer group...")

            for symbol in self.peer_group:
                try:
                    # Use our scraper for peer data
                    peer_scraper, peer_info, peer_price = fetch_stock_data(symbol)

                    self.peer_data[symbol] = {
                        'market_cap': peer_info.get('marketCap', 0),
                        'enterprise_value': peer_info.get('enterpriseValue', 0),
                        'trailing_pe': peer_info.get('trailingPE', None),
                        'forward_pe': peer_info.get('forwardPE', None),
                        'price_to_book': peer_info.get('priceToBook', None),
                        'price_to_sales': peer_info.get('priceToSalesTrailing12Months', None),
                        'ev_to_revenue': peer_info.get('enterpriseToRevenue', None),
                        'ev_to_ebitda': peer_info.get('enterpriseToEbitda', None),
                        'peg_ratio': peer_info.get('pegRatio', None),
                        'revenue': peer_info.get('totalRevenue', 0),
                        'ebitda': peer_info.get('ebitda', 0),
                        'net_income': peer_info.get('netIncomeToCommon', 0),
                        'revenue_growth': peer_info.get('revenueGrowth', None),
                        'earnings_growth': peer_info.get('earningsGrowth', None),
                        'current_price': peer_price
                    }

                except Exception as e:
                    logger.warning(f"  Could not fetch data for {symbol}: {e}")
                    continue

            logger.info(f"  Collected data for {len(self.peer_data)} companies")

            return self.peer_data

        except Exception as e:
            logger.error(f"Error in Comps Step 3: {e}")
            return {}

    def step4_calculate_peer_multiples(self) -> Dict:
        """Step 4: Calculate and compare valuation multiples across peer group"""
        try:
            logger.info("Comps Step 4: Calculating peer group multiples...")

            if not self.peer_data:
                self.step3_input_financial_data()

            # Collect all multiples
            multiples_df = pd.DataFrame(self.peer_data).T

            # Calculate statistics for each multiple
            multiple_stats = {}

            key_multiples = ['trailing_pe', 'forward_pe', 'price_to_book', 'price_to_sales',
                           'ev_to_revenue', 'ev_to_ebitda', 'peg_ratio']

            for multiple in key_multiples:
                if multiple in multiples_df.columns:
                    values = multiples_df[multiple].dropna()

                    if len(values) > 0:
                        multiple_stats[multiple] = {
                            'min': float(values.min()),
                            'percentile_25': float(values.quantile(0.25)),
                            'median': float(values.median()),
                            'mean': float(values.mean()),
                            'percentile_75': float(values.quantile(0.75)),
                            'max': float(values.max()),
                            'target_value': float(multiples_df.loc[self.ticker, multiple]) if self.ticker in multiples_df.index else None
                        }

            logger.info(f"  Calculated statistics for {len(multiple_stats)} valuation multiples")

            # Display key multiples
            if 'trailing_pe' in multiple_stats:
                target_pe = multiple_stats['trailing_pe']['target_value']
                target_pe_str = f"{target_pe:.2f}" if target_pe is not None else "N/A"
                logger.info(f"  P/E - Median: {multiple_stats['trailing_pe']['median']:.2f}, Target: {target_pe_str}")
            if 'ev_to_ebitda' in multiple_stats:
                target_ev = multiple_stats['ev_to_ebitda']['target_value']
                target_ev_str = f"{target_ev:.2f}" if target_ev is not None else "N/A"
                logger.info(f"  EV/EBITDA - Median: {multiple_stats['ev_to_ebitda']['median']:.2f}, Target: {target_ev_str}")

            return {
                'multiple_statistics': multiple_stats,
                'peer_data': self.peer_data,
                'peer_count': len(self.peer_data)
            }

        except Exception as e:
            logger.error(f"Error in Comps Step 4: {e}")
            return {}

    def step5_apply_multiple_to_target(self, multiple_stats: Dict) -> Dict:
        """Step 5: Apply peer multiples to target company to derive valuation"""
        try:
            logger.info("Comps Step 5: Applying multiples to derive target valuation...")

            target_data = self.peer_data.get(self.ticker, {})

            if not target_data:
                return {}

            valuations = {}

            # P/E based valuation
            if 'trailing_pe' in multiple_stats and target_data.get('net_income'):
                median_pe = multiple_stats['trailing_pe']['median']
                net_income = target_data['net_income']
                shares = self.stock.info.get('sharesOutstanding', 1)

                eps = net_income / shares if shares > 0 else 0
                implied_price_pe = eps * median_pe

                valuations['pe_valuation'] = {
                    'median_multiple': median_pe,
                    'implied_price': implied_price_pe,
                    'current_price': target_data['current_price'],
                    'upside_downside': (implied_price_pe / target_data['current_price'] - 1) if target_data['current_price'] > 0 else 0
                }

            # EV/EBITDA based valuation
            if 'ev_to_ebitda' in multiple_stats and target_data.get('ebitda'):
                median_ev_ebitda = multiple_stats['ev_to_ebitda']['median']
                ebitda = target_data['ebitda']

                implied_ev = ebitda * median_ev_ebitda

                # Convert to equity value
                net_debt = self.stock.info.get('totalDebt', 0) - self.stock.info.get('totalCash', 0)
                implied_equity_value = implied_ev - net_debt
                shares = self.stock.info.get('sharesOutstanding', 1)
                implied_price_ev = implied_equity_value / shares if shares > 0 else 0

                valuations['ev_ebitda_valuation'] = {
                    'median_multiple': median_ev_ebitda,
                    'implied_price': implied_price_ev,
                    'current_price': target_data['current_price'],
                    'upside_downside': (implied_price_ev / target_data['current_price'] - 1) if target_data['current_price'] > 0 else 0
                }

            # P/S based valuation
            if 'price_to_sales' in multiple_stats and target_data.get('revenue'):
                median_ps = multiple_stats['price_to_sales']['median']
                revenue = target_data['revenue']
                shares = self.stock.info.get('sharesOutstanding', 1)

                revenue_per_share = revenue / shares if shares > 0 else 0
                implied_price_ps = revenue_per_share * median_ps

                valuations['ps_valuation'] = {
                    'median_multiple': median_ps,
                    'implied_price': implied_price_ps,
                    'current_price': target_data['current_price'],
                    'upside_downside': (implied_price_ps / target_data['current_price'] - 1) if target_data['current_price'] > 0 else 0
                }

            # Calculate average implied valuation
            all_implied_prices = [v['implied_price'] for v in valuations.values() if v.get('implied_price', 0) > 0]

            if all_implied_prices:
                avg_implied_price = np.mean(all_implied_prices)
                current_price = target_data['current_price']

                overall_assessment = {
                    'average_implied_price': avg_implied_price,
                    'current_price': current_price,
                    'overall_upside_downside': (avg_implied_price / current_price - 1) if current_price > 0 else 0
                }
            else:
                overall_assessment = {}

            results = {
                'individual_valuations': valuations,
                'overall_assessment': overall_assessment
            }

            logger.info(f"  Average Implied Price: ${overall_assessment.get('average_implied_price', 0):.2f}")
            logger.info(f"  Current Price: ${overall_assessment.get('current_price', 0):.2f}")
            logger.info(f"  Implied Upside/(Downside): {overall_assessment.get('overall_upside_downside', 0):.1%}")

            return results

        except Exception as e:
            logger.error(f"Error in Comps Step 5: {e}")
            return {}

    def perform_full_comps_analysis(self) -> Dict:
        """Execute complete 5-step comparable company analysis"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Comparable Company Analysis for {self.ticker}")
            logger.info(f"{'='*60}\n")

            # Step 1: Compile Peer Group
            peer_group = self.step1_compile_peer_group()

            # Step 2: Industry Research
            industry_research = self.step2_industry_research()

            # Step 3: Input Financial Data
            financial_data = self.step3_input_financial_data()

            # Step 4: Calculate Peer Multiples
            multiples_analysis = self.step4_calculate_peer_multiples()

            # Step 5: Apply Multiples to Target
            valuation_results = self.step5_apply_multiple_to_target(multiples_analysis.get('multiple_statistics', {}))

            # Generate recommendation
            recommendation = self._generate_comps_recommendation(valuation_results)

            results = {
                'ticker': self.ticker,
                'peer_group': peer_group,
                'industry_research': industry_research,
                'multiples_analysis': multiples_analysis,
                'valuation': valuation_results,
                'recommendation': recommendation
            }

            logger.info(f"\n{'='*60}")
            logger.info(f"Comparable Company Analysis Complete for {self.ticker}")
            logger.info(f"{'='*60}\n")

            return results

        except Exception as e:
            logger.error(f"Error in full comps analysis: {e}")
            return {}

    def _generate_comps_recommendation(self, valuation: Dict) -> str:
        """Generate comps-based recommendation"""
        try:
            overall = valuation.get('overall_assessment', {})
            upside = overall.get('overall_upside_downside', 0)

            if upside > 0.20:
                return f"STRONG BUY - Trading at {upside:.1%} discount to peer group median. Significant upside potential."
            elif upside > 0.10:
                return f"BUY - Trading at {upside:.1%} discount to peers. Moderately undervalued."
            elif upside > -0.05:
                return f"HOLD - Trading in line with peer group. Fairly valued relative to comparables."
            elif upside > -0.15:
                return f"UNDERPERFORM - Trading at {abs(upside):.1%} premium to peers. Moderately overvalued."
            else:
                return f"SELL - Trading at {abs(upside):.1%} premium to peer group. Significantly overvalued."

        except:
            return "Unable to generate recommendation"


# ============ ML SYNTHESIS MODEL ============

class MLSynthesisModel:
    """
    Machine Learning model that synthesizes DCF, Revenue Forecast, and Comps analyses
    Provides hedge fund analyst-style insights and final recommendation
    """

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self._initialize_model()

    def _initialize_model(self):
        """Initialize ensemble model"""
        try:
            # Use Gradient Boosting for nuanced decision making
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )

            # Pre-train with synthetic data representing various scenarios
            self._pretrain_model()

        except Exception as e:
            logger.error(f"Error initializing ML model: {e}")

    def _pretrain_model(self):
        """Pre-train model with synthetic scenarios"""
        try:
            # Create synthetic training data representing different valuation scenarios
            n_samples = 1000

            # Features: [DCF upside, Revenue growth, Comps upside, DCF confidence, Revenue confidence, Comps confidence]
            X_train = np.random.randn(n_samples, 6)

            # Target: Overall recommendation score (-1 to 1, where 1 is strong buy, -1 is strong sell)
            y_train = (
                X_train[:, 0] * 0.40 +  # DCF upside (40% weight)
                X_train[:, 1] * 0.25 +  # Revenue growth (25% weight)
                X_train[:, 2] * 0.20 +  # Comps upside (20% weight)
                X_train[:, 3] * 0.05 +  # DCF confidence (5% weight)
                X_train[:, 4] * 0.05 +  # Revenue confidence (5% weight)
                X_train[:, 5] * 0.05 +  # Comps confidence (5% weight)
                np.random.randn(n_samples) * 0.1  # Noise
            )

            self.scaler.fit(X_train)
            X_scaled = self.scaler.transform(X_train)

            self.model.fit(X_scaled, y_train)

            logger.info("ML synthesis model pre-trained successfully")

        except Exception as e:
            logger.error(f"Error pre-training model: {e}")

    def synthesize_analyses(self, dcf_results: Dict, revenue_results: Dict, comps_results: Dict) -> Dict:
        """Synthesize all three analyses into comprehensive insights"""
        try:
            logger.info(f"\n{'='*60}")
            logger.info("ML Synthesis: Combining all analyses...")
            logger.info(f"{'='*60}\n")

            # Extract key metrics from each analysis
            dcf_metrics = self._extract_dcf_metrics(dcf_results)
            revenue_metrics = self._extract_revenue_metrics(revenue_results)
            comps_metrics = self._extract_comps_metrics(comps_results)

            # Prepare features for ML model
            features = np.array([
                dcf_metrics['upside'],
                revenue_metrics['growth_score'],
                comps_metrics['upside'],
                dcf_metrics['confidence'],
                revenue_metrics['confidence'],
                comps_metrics['confidence']
            ]).reshape(1, -1)

            # Replace NaN values with 0
            features = np.nan_to_num(features, nan=0.0)

            # Get ML prediction
            features_scaled = self.scaler.transform(features)
            ml_score = self.model.predict(features_scaled)[0]

            # Calculate weighted target price
            target_price = self._calculate_consensus_target(dcf_results, comps_results)

            # Generate comprehensive analysis
            comprehensive_analysis = self._generate_hedge_fund_analysis(
                dcf_results, revenue_results, comps_results, ml_score
            )

            # Final recommendation
            final_recommendation = self._generate_final_recommendation(
                ml_score, target_price, dcf_results, revenue_results, comps_results
            )

            results = {
                'ml_score': ml_score,
                'target_price': target_price,
                'comprehensive_analysis': comprehensive_analysis,
                'final_recommendation': final_recommendation,
                'model_weights': {
                    'dcf': 0.40,
                    'revenue': 0.25,
                    'comps': 0.20,
                    'confidence': 0.15
                },
                'individual_scores': {
                    'dcf': dcf_metrics,
                    'revenue': revenue_metrics,
                    'comps': comps_metrics
                }
            }

            logger.info(f"ML Score: {ml_score:.3f} (-1=Sell, 0=Hold, 1=Buy)")
            logger.info(f"Consensus Target Price: ${target_price:.2f}")

            return results

        except Exception as e:
            logger.error(f"Error in ML synthesis: {e}")
            return {}

    def _extract_dcf_metrics(self, dcf_results: Dict) -> Dict:
        """Extract key metrics from DCF analysis"""
        try:
            valuation = dcf_results.get('valuation', {})

            intrinsic = valuation.get('intrinsic_value_per_share', 0)
            current = valuation.get('current_price', 1)

            upside = (intrinsic / current - 1) if current > 0 else 0

            # Confidence based on terminal value percentage (lower is better)
            tv_pct = valuation.get('terminal_value_percentage', 0.7)
            confidence = 1.0 - (tv_pct - 0.5) if tv_pct > 0.5 else 1.0
            confidence = max(0.3, min(1.0, confidence))

            return {
                'upside': upside,
                'confidence': confidence,
                'intrinsic_value': intrinsic,
                'current_price': current
            }

        except Exception as e:
            logger.error(f"Error extracting DCF metrics: {e}")
            return {'upside': 0, 'confidence': 0.5}

    def _extract_revenue_metrics(self, revenue_results: Dict) -> Dict:
        """Extract key metrics from revenue forecast"""
        try:
            growth_analysis = revenue_results.get('growth_analysis', {})

            cagr = growth_analysis.get('cagr', 0)
            volatility = growth_analysis.get('growth_volatility', 0.1)

            # Handle NaN values
            if np.isnan(cagr):
                cagr = 0
            if np.isnan(volatility):
                volatility = 0.1

            # Normalize growth to -1 to 1 scale
            growth_score = np.tanh(cagr * 5)  # tanh keeps it bounded

            # Confidence inversely related to volatility
            confidence = 1.0 / (1.0 + volatility * 10)
            confidence = max(0.3, min(1.0, confidence))

            return {
                'growth_score': growth_score,
                'confidence': confidence,
                'cagr': cagr,
                'volatility': volatility
            }

        except Exception as e:
            logger.error(f"Error extracting revenue metrics: {e}")
            return {'growth_score': 0, 'confidence': 0.5}

    def _extract_comps_metrics(self, comps_results: Dict) -> Dict:
        """Extract key metrics from comps analysis"""
        try:
            valuation = comps_results.get('valuation', {})
            overall = valuation.get('overall_assessment', {})

            upside = overall.get('overall_upside_downside', 0)

            # Confidence based on number of peers and valuation methods
            peer_count = comps_results.get('peer_group', [])
            valuations_count = len(valuation.get('individual_valuations', {}))

            confidence = min(1.0, (len(peer_count) * valuations_count) / 12)  # Max at 3 peers x 4 methods
            confidence = max(0.3, confidence)

            return {
                'upside': upside,
                'confidence': confidence,
                'implied_price': overall.get('average_implied_price', 0)
            }

        except Exception as e:
            logger.error(f"Error extracting comps metrics: {e}")
            return {'upside': 0, 'confidence': 0.5}

    def _calculate_consensus_target(self, dcf_results: Dict, comps_results: Dict) -> float:
        """Calculate consensus target price from DCF and Comps"""
        try:
            dcf_target = dcf_results.get('valuation', {}).get('intrinsic_value_per_share', 0)
            comps_target = comps_results.get('valuation', {}).get('overall_assessment', {}).get('average_implied_price', 0)

            if dcf_target > 0 and comps_target > 0:
                # Weight DCF more heavily (60/40)
                consensus = dcf_target * 0.60 + comps_target * 0.40
            elif dcf_target > 0:
                consensus = dcf_target
            elif comps_target > 0:
                consensus = comps_target
            else:
                consensus = 0

            return consensus

        except:
            return 0

    def _generate_hedge_fund_analysis(self, dcf: Dict, revenue: Dict, comps: Dict, ml_score: float) -> str:
        """Generate comprehensive hedge fund analyst-style writeup"""
        try:
            ticker = dcf.get('ticker', 'N/A')
            company_name = dcf.get('business_info', {}).get('company_name', ticker)

            analysis = f"""
INVESTMENT THESIS - {company_name} ({ticker})

VALUATION ANALYSIS:
Our comprehensive three-model valuation framework indicates {'a compelling investment opportunity' if ml_score > 0.3 else 'fair value' if ml_score > -0.3 else 'significant overvaluation'}.

1. DISCOUNTED CASH FLOW (DCF) ANALYSIS:
   - Intrinsic Value: ${dcf.get('valuation', {}).get('intrinsic_value_per_share', 0):.2f}
   - Current Price: ${dcf.get('valuation', {}).get('current_price', 0):.2f}
   - Implied Return: {dcf.get('valuation', {}).get('premium_discount', 0):.1%}
   - WACC: {dcf.get('discount_rate', {}).get('wacc', 0):.2%}
   - Terminal Value represents {dcf.get('valuation', {}).get('terminal_value_percentage', 0):.1%} of enterprise value
   - Assessment: {dcf.get('recommendation', 'N/A')}

2. REVENUE FORECASTING MODEL:
   - Historical CAGR: {revenue.get('growth_analysis', {}).get('cagr', 0):.1%}
   - Revenue Trend: {revenue.get('growth_analysis', {}).get('trend', 'N/A')}
   - Growth Volatility: {revenue.get('growth_analysis', {}).get('growth_volatility', 0):.2%}
   - 5-Year Projection: Ensemble model projects {'strong' if revenue.get('growth_analysis', {}).get('cagr', 0) > 0.10 else 'moderate' if revenue.get('growth_analysis', {}).get('cagr', 0) > 0.05 else 'low'} growth
   - Assessment: {revenue.get('recommendation', 'N/A')}

3. COMPARABLE COMPANY ANALYSIS:
   - Peer Group: {len(comps.get('peer_group', []))} comparable companies
   - Relative Valuation: {'Discount to peers' if comps.get('valuation', {}).get('overall_assessment', {}).get('overall_upside_downside', 0) > 0 else 'Premium to peers'}
   - Implied Upside/(Downside): {comps.get('valuation', {}).get('overall_assessment', {}).get('overall_upside_downside', 0):.1%}
   - Assessment: {comps.get('recommendation', 'N/A')}

MACHINE LEARNING SYNTHESIS:
Our proprietary ML model, which weighs DCF (40%), Revenue Growth (25%), and Comps (20%) with confidence adjustments (15%),
generates a score of {ml_score:.3f} on a scale of -1 (Strong Sell) to +1 (Strong Buy).

KEY INVESTMENT CONSIDERATIONS:
- {'Intrinsic value substantially exceeds market price' if ml_score > 0.3 else 'Trading near fair value' if ml_score > -0.3 else 'Market price exceeds intrinsic value'}
- Revenue growth trajectory is {'accelerating' if revenue.get('growth_analysis', {}).get('trend', '') == 'Accelerating' else 'decelerating'}
- Valuation relative to peers is {'attractive' if comps.get('valuation', {}).get('overall_assessment', {}).get('overall_upside_downside', 0) > 0 else 'stretched'}
- Risk-adjusted return profile is {'favorable' if ml_score > 0.2 else 'neutral' if ml_score > -0.2 else 'unfavorable'}
"""

            return analysis.strip()

        except Exception as e:
            logger.error(f"Error generating hedge fund analysis: {e}")
            return "Unable to generate comprehensive analysis"

    def _generate_final_recommendation(self, ml_score: float, target_price: float,
                                      dcf: Dict, revenue: Dict, comps: Dict) -> Dict:
        """Generate final buy/hold/sell recommendation with price target"""
        try:
            current_price = dcf.get('valuation', {}).get('current_price', 0)

            if current_price == 0:
                return {'action': 'NO RECOMMENDATION', 'rationale': 'Insufficient data'}

            upside = (target_price / current_price - 1) if current_price > 0 else 0

            # Determine action based on ML score and upside
            if ml_score > 0.4 and upside > 0.20:
                action = "STRONG BUY"
                rationale = f"All three models indicate significant undervaluation. Target price of ${target_price:.2f} implies {upside:.1%} upside. High conviction opportunity."
            elif ml_score > 0.2 and upside > 0.10:
                action = "BUY"
                rationale = f"Models suggest stock is undervalued. Target price of ${target_price:.2f} implies {upside:.1%} upside. Favorable risk/reward."
            elif ml_score > -0.2 and abs(upside) < 0.10:
                action = "HOLD"
                rationale = f"Stock trading near fair value. Target price of ${target_price:.2f} implies limited upside/downside. Maintain position."
            elif ml_score > -0.4 and upside < -0.10:
                action = "REDUCE"
                rationale = f"Models suggest stock is overvalued. Current price exceeds target of ${target_price:.2f}. Consider reducing exposure."
            else:
                action = "SELL"
                rationale = f"All models indicate significant overvaluation. Current price substantially exceeds target of ${target_price:.2f}. High conviction sell."

            recommendation = {
                'action': action,
                'target_price': target_price,
                'current_price': current_price,
                'upside_potential': upside,
                'ml_score': ml_score,
                'rationale': rationale,
                'price_range': {
                    'bull_case': target_price * 1.15,
                    'base_case': target_price,
                    'bear_case': target_price * 0.85
                }
            }

            return recommendation

        except Exception as e:
            logger.error(f"Error generating final recommendation: {e}")
            return {'action': 'ERROR', 'rationale': str(e)}


# ============ MAIN STOCK ANALYZER ============

class StockAnalyzer:
    """
    Main class orchestrating all three analyses and ML synthesis
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        # Initialize DCF first to fetch data
        self.dcf_analyzer = DCFValuation(self.ticker)
        # Will share stock object after DCF fetches it
        self.revenue_forecaster = None
        self.comps_analyzer = None
        self.ml_synthesizer = MLSynthesisModel()

    def analyze(self) -> Dict:
        """Perform complete stock analysis"""
        try:
            logger.info(f"\n{'#'*80}")
            logger.info(f"#{'':^78}#")
            logger.info(f"#{'COMPREHENSIVE STOCK ANALYSIS':^78}#")
            logger.info(f"#{'':^78}#")
            logger.info(f"#{'Ticker: ' + self.ticker:^78}#")
            logger.info(f"#{'':^78}#")
            logger.info(f"{'#'*80}\n")

            start_time = datetime.now()

            # Perform DCF analysis first (this fetches the stock data)
            dcf_results = self.dcf_analyzer.perform_full_dcf_analysis()

            # Now initialize other analyzers with the shared stock object
            self.revenue_forecaster = RevenueForecaster(
                self.ticker,
                stock=self.dcf_analyzer.stock,
                info=self.dcf_analyzer.info
            )
            self.comps_analyzer = ComparableCompanyAnalysis(
                self.ticker,
                stock=self.dcf_analyzer.stock,
                info=self.dcf_analyzer.info,
                current_price=self.dcf_analyzer.current_price
            )

            # Perform remaining analyses
            revenue_results = self.revenue_forecaster.perform_full_revenue_analysis()
            comps_results = self.comps_analyzer.perform_full_comps_analysis()

            # ML Synthesis
            synthesis = self.ml_synthesizer.synthesize_analyses(
                dcf_results, revenue_results, comps_results
            )

            # Compile final output
            final_output = {
                'ticker': self.ticker,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'analysis_duration': str(datetime.now() - start_time),
                'dcf_analysis': dcf_results,
                'revenue_forecast': revenue_results,
                'comparable_companies': comps_results,
                'ml_synthesis': synthesis,
                'executive_summary': self._create_executive_summary(
                    dcf_results, revenue_results, comps_results, synthesis
                )
            }

            # Print executive summary
            self._print_executive_summary(final_output)

            logger.info(f"\n{'#'*80}")
            logger.info(f"#{'ANALYSIS COMPLETE':^78}#")
            logger.info(f"{'#'*80}\n")

            return final_output

        except Exception as e:
            logger.error(f"Error in stock analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def _create_executive_summary(self, dcf: Dict, revenue: Dict, comps: Dict, synthesis: Dict) -> Dict:
        """Create executive summary of all analyses"""
        return {
            'company_name': dcf.get('business_info', {}).get('company_name', self.ticker),
            'sector': dcf.get('business_info', {}).get('sector', 'N/A'),
            'current_price': dcf.get('valuation', {}).get('current_price', 0),
            'dcf_intrinsic_value': dcf.get('valuation', {}).get('intrinsic_value_per_share', 0),
            'comps_implied_value': comps.get('valuation', {}).get('overall_assessment', {}).get('average_implied_price', 0),
            'consensus_target': synthesis.get('target_price', 0),
            'recommendation': synthesis.get('final_recommendation', {}),
            'key_metrics': {
                'dcf_upside': dcf.get('valuation', {}).get('premium_discount', 0),
                'revenue_cagr': revenue.get('growth_analysis', {}).get('cagr', 0),
                'comps_upside': comps.get('valuation', {}).get('overall_assessment', {}).get('overall_upside_downside', 0),
                'ml_score': synthesis.get('ml_score', 0)
            }
        }

    def _print_executive_summary(self, output: Dict):
        """Print formatted executive summary"""
        try:
            summary = output.get('executive_summary', {})
            recommendation = summary.get('recommendation', {})

            print("\n" + "="*80)
            print("EXECUTIVE SUMMARY".center(80))
            print("="*80)
            print(f"\nCompany: {summary.get('company_name', 'N/A')}")
            print(f"Ticker: {self.ticker}")
            print(f"Sector: {summary.get('sector', 'N/A')}")
            print(f"Analysis Date: {output.get('analysis_date', 'N/A')}")
            print(f"\nCurrent Price: ${summary.get('current_price', 0):.2f}")
            print(f"\nVALUATION SUMMARY:")
            print(f"  DCF Intrinsic Value:      ${summary.get('dcf_intrinsic_value', 0):.2f}")
            print(f"  Comps Implied Value:      ${summary.get('comps_implied_value', 0):.2f}")
            print(f"  Consensus Target Price:   ${summary.get('consensus_target', 0):.2f}")
            print(f"\nKEY METRICS:")
            metrics = summary.get('key_metrics', {})
            print(f"  DCF Upside/(Downside):    {metrics.get('dcf_upside', 0):>7.1%}")
            print(f"  Revenue CAGR:             {metrics.get('revenue_cagr', 0):>7.1%}")
            print(f"  Comps Relative Value:     {metrics.get('comps_upside', 0):>7.1%}")
            print(f"  ML Confidence Score:      {metrics.get('ml_score', 0):>7.3f}")
            print(f"\nFINAL RECOMMENDATION: {recommendation.get('action', 'N/A')}")
            print(f"Target Price: ${recommendation.get('target_price', 0):.2f}")
            print(f"Upside Potential: {recommendation.get('upside_potential', 0):.1%}")
            print(f"\nRationale: {recommendation.get('rationale', 'N/A')}")
            print(f"\nPrice Range:")
            price_range = recommendation.get('price_range', {})
            print(f"  Bull Case:  ${price_range.get('bull_case', 0):.2f}")
            print(f"  Base Case:  ${price_range.get('base_case', 0):.2f}")
            print(f"  Bear Case:  ${price_range.get('bear_case', 0):.2f}")
            print("\n" + "="*80)

        except Exception as e:
            logger.error(f"Error printing executive summary: {e}")


# ============ MAIN EXECUTION ============

def main(ticker: str = "AAPL"):
    """Main function to run stock analysis"""
    try:
        analyzer = StockAnalyzer(ticker)
        results = analyzer.analyze()
        return results

    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import sys

    # Get ticker from command line or use default
    if len(sys.argv) > 1:
        ticker = sys.argv[1]
    else:
        ticker = "AAPL"

    print(f"\nAnalyzing {ticker}...\n")
    results = main(ticker)

    if results:
        print("\n✓ Analysis complete! Results saved in output.")
