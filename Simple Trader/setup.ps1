# OSRS Trading Analytics - Setup Script
# Run this script to set up the project automatically

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  OSRS Trading Analytics - Setup" -ForegroundColor Yellow
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "[1/5] Checking Python version..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  ✅ Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  ❌ Python not found! Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host ""
Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Cyan
if (Test-Path "venv") {
    Write-Host "  ℹ️  Virtual environment already exists" -ForegroundColor Yellow
} else {
    python -m venv venv
    Write-Host "  ✅ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host ""
Write-Host "[3/5] Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"
Write-Host "  ✅ Virtual environment activated" -ForegroundColor Green

# Install dependencies
Write-Host ""
Write-Host "[4/5] Installing dependencies..." -ForegroundColor Cyan
pip install -q -r requirements.txt
Write-Host "  ✅ Dependencies installed" -ForegroundColor Green

# Create data directory
Write-Host ""
Write-Host "[5/5] Creating data directory..." -ForegroundColor Cyan
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
    Write-Host "  ✅ Data directory created" -ForegroundColor Green
} else {
    Write-Host "  ℹ️  Data directory already exists" -ForegroundColor Yellow
}

# Final instructions
Write-Host ""
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host ""
Write-Host "⚠️  IMPORTANT: Before running, update your User-Agent!" -ForegroundColor Yellow
Write-Host ""
Write-Host "   1. Open core/config.py" -ForegroundColor White
Write-Host "   2. Find USER_AGENT variable" -ForegroundColor White
Write-Host "   3. Replace with your Discord handle" -ForegroundColor White
Write-Host ""
Write-Host "Example:" -ForegroundColor Cyan
Write-Host '   USER_AGENT = "osrs-trader - @YourDiscord#1234"' -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "   • Test API: python examples.py" -ForegroundColor White
Write-Host "   • Run dashboard: streamlit run app.py" -ForegroundColor White
Write-Host ""
