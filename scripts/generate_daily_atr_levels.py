
name: Update Daily ATR Levels

on:
  schedule:
    - cron: '30 21 * * 1-5'  # 21:30 UTC = 4:30PM ET, Monday–Friday
  workflow_dispatch:  # optional manual trigger

jobs:
  update-atr:
    runs-on: ubuntu-latest

    steps:
    - name: Checkout repository
      uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install yfinance pandas

    - name: Run ATR Level Generator
      run: python scripts/generate_daily_atr_levels.py

    - name: Commit updated ATR levels
      run: |
        git config user.name "github-actions"
        git config user.email "actions@github.com"
        git add data/daily_atr_levels.json
        git commit -m "Update ATR levels automatically"
        git push
