$ErrorActionPreference = "Stop"

# Define color function
function Write-ColorOutput($message, $color) {
    Write-Host $message -ForegroundColor $color
}

# Helper function to check if a command exists
function Test-Command($command) {
    $exists = $null -ne (Get-Command $command -ErrorAction SilentlyContinue)
    return $exists
}

# Check if Node.js is installed
if (-not (Test-Command "node")) {
    Write-ColorOutput "Node.js no está instalado. Por favor instala Node.js antes de continuar." "Red"
    exit 1
}

# Check if Python is installed
if (-not (Test-Command "python")) {
    Write-ColorOutput "Python no está instalado. Por favor instala Python antes de continuar." "Red"
    exit 1
}

# Start backend
Write-ColorOutput "`n🚀 Iniciando servidor backend..." "Cyan"
Start-Process -FilePath "powershell" -ArgumentList "-Command", "cd $PSScriptRoot\backend; python -m uvicorn src.main:app --reload --port 8000"

# Wait a moment for the backend to start
Start-Sleep -Seconds 3

# Start frontend
Write-ColorOutput "`n🚀 Iniciando servidor frontend..." "Green"
Start-Process -FilePath "powershell" -ArgumentList "-Command", "cd $PSScriptRoot\frontend; npm start"

Write-ColorOutput "`n✅ Aplicación iniciada correctamente!" "Yellow"
Write-ColorOutput "- Backend: http://localhost:8000" "Yellow"
Write-ColorOutput "- Frontend: http://localhost:3000" "Yellow"
Write-ColorOutput "- API Docs: http://localhost:8000/docs" "Yellow"
Write-ColorOutput "`nPresiona CTRL+C en sus respectivas ventanas para detener los servidores." "Yellow"

# Keep the script running
Write-ColorOutput "`nPresiona CTRL+C para cerrar este script (los servidores seguirán ejecutándose en sus propias ventanas)" "Magenta"
try {
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
catch {
    Write-ColorOutput "`nScript cerrado. Los servidores siguen ejecutándose en ventanas separadas." "Magenta"
}
