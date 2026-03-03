<#
 .SYNOPSIS
 Script file to pre-calculate consel results using a local consel installation.

 .DESCRIPTION
 This script is used to generate the CSV files that the benchmarks use as consel's reference output.
 The CSV files contain only the base statistics consel always generates, without those that can be disabled,
 since we only require the AU value. Furhter, the script doesn't need to be run for existing inputs, since we
 track them in git. Use this to add new datasets to the benchmarks.

 .PARAM SiteLH
 Path to the puzzle-style (raxml) per-site log-likelihood file that serves as input to consel.

 .PARAM Samples
 How many consel runs to generate
#>

param (
    [Parameter(Mandatory)]
    [string] $SiteLH,

    [Parameter(Mandatory = $false)]
    [int] $Samples = 50
)

Import-Module $PSScriptRoot/consel

$DataDir = Split-Path -Parent $SiteLH
$DirectoryName = Split-Path -LeafBase $SiteLH
$DirectoryPath = [System.IO.Path]::Combine($DataDir, $DirectoryName)

if (-not (Test-Path $DirectoryPath)) {
    New-Item -ItemType Directory $DirectoryPath
}

Write-Host "Generating Run 0 to $($Samples - 1)..."
0..($Samples - 1) | ForEach-Object {
    $Prefix = [System.IO.Path]::Combine($DirectoryPath, "run$_")
    if (-not (Test-Path "$Prefix.csv")) {
        # Run consel
        Invoke-Makermt -Sitelh $SiteLH -Output $Prefix
        Invoke-Consel -Rmt "$Prefix.rmt" -Output $Prefix

        # Convert to CSV file
        Import-Pv "$Prefix.pv" | Export-Csv "$Prefix.csv" -UseQuotes Always -NoTypeInformation

        # Cleanup consel output
        Remove-Item "$Prefix.*" -Exclude *.csv
    } else {
        Write-Host "Skipping run $_!"
    }
}