import React, { useState, useEffect } from 'react';

function App() {
  const [status, setStatus] = useState('Testing...');
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // EXACT backend URL - no variables
    const testUrl = 'https://autoanalyst-docker.onrender.com/api/health';
    
    console.log('Attempting to fetch:', testUrl);
    
    // Simple fetch with full logging
    fetch(testUrl, {
      method: 'GET',
      mode: 'cors',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      }
    })
    .then(res => {
      console.log('Response received:', res);
      console.log('Response status:', res.status);
      console.log('Response headers:', res.headers);
      setStatus(`Response: ${res.status}`);
      return res.text(); // Get as text first to see what's returned
    })
    .then(text => {
      console.log('Response text:', text);
      setResponse(text);
      try {
        const json = JSON.parse(text);
        console.log('Parsed JSON:', json);
      } catch (e) {
        console.log('Not valid JSON:', e);
      }
    })
    .catch(err => {
      console.error('Fetch error:', err);
      setStatus('ERROR');
      setError(err.toString());
    });
  }, []);

  return (
    <div style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h1>Connection Test</h1>
      <p>Backend URL: https://autoanalyst-docker.onrender.com/api/health</p>
      <p>Status: {status}</p>
      {error && <p style={{color: 'red'}}>Error: {error}</p>}
      {response && (
        <div>
          <h3>Response:</h3>
          <pre style={{background: '#f0f0f0', padding: '10px'}}>
            {response}
          </pre>
        </div>
      )}
      <hr />
      <h3>Manual Test Links (open in new tab):</h3>
      <ul>
        <li>
          <a href="https://autoanalyst-docker.onrender.com/api/health" 
             target="_blank" rel="noopener noreferrer">
            Test Backend Health
          </a>
        </li>
      </ul>
    </div>
  );
}

export default App;