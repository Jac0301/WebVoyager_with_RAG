# Set environment variable for Python to use UTF-8 for IO
$env:PYTHONIOENCODING='utf-8'

# Navigate to the script directory (optional, assumes running from WebVoyager folder)
# cd (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Create results directory if it doesn't exist
if (-not (Test-Path ".\results")) {
    New-Item -ItemType Directory -Path ".\results"
}

# Run the main Python script with arguments
Write-Host "Starting WebVoyager run..."
try {
    python run.py `
        --test_file "./data/tasks_test.jsonl" `
        --max_iter 10 `
        --save_accessibility_tree `
        --output_dir "./results"
    Write-Host "WebVoyager run finished."
} catch {
    Write-Error "An error occurred during the Python script execution: $($_.Exception.Message)"
    # Optionally, add more detailed error logging or handling here
} finally {
    # Clean up environment variable if needed (optional)
    # Remove-Item Env:\PYTHONIOENCODING
}
