<#
 .SYNOPSIS
 Install consel from github if it doesn't exist yet.

 .DESCRIPTION
 The benchmark scripts compare consel and cocos. For this we need to download and compile consel. For ease of use,
 we compile and execute it in WSL, which assumes the WSL default distribution is equipped to compile C through
 Makefiles.
#>

$ConselDir = [System.IO.Path]::Combine($PSScriptRoot, "consel")
if (-not (Test-Path $ConselDir)) {
    Write-Host "Installing consel..."
    git clone https://github.com/shimo-lab/consel $ConselDir

    Push-Location ([System.IO.Path]::Combine($ConselDir, "src"))
    wsl make
    wsl make install

    Pop-Location
}
$ConselBinary = [System.IO.Path]::Combine($ConselDir, "bin", "consel")

if (-not (Test-Path $ConselBinary)) {
    Write-Error "Consel installation failed."
    exit 1
}