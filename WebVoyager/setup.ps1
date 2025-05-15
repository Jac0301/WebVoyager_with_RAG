# Activate conda environment (uncomment and modify if needed)
# conda activate webvoyager

# Uninstall existing OpenAI package if it exists
pip uninstall -y openai

# Install required packages in specific order
pip install openai>=1.1.1
pip install selenium==4.15.2
pip install pillow==10.1.0

# Create necessary directories if they don't exist
if (-not (Test-Path ".\data")) { New-Item -ItemType Directory -Path ".\data" }
if (-not (Test-Path ".\results")) { New-Item -ItemType Directory -Path ".\results" }
if (-not (Test-Path ".\downloads")) { New-Item -ItemType Directory -Path ".\downloads" }

# Check if Chrome is installed
$chrome = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\chrome.exe" -ErrorAction SilentlyContinue
if (-not $chrome) {
    Write-Host "Chrome browser is not installed. Please install Chrome to continue."
    exit 1
}

Write-Host "Setup completed successfully!" 