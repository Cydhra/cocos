<#
 .SYNOPSIS
 Run consel and cocos multiple times on the same inputs and plot the standard deviation of the results.

 .DESCRIPTION
 The results of the AU test depend on the bootstrap results and thus on the random seed. To verify that cocos
 does not introduce additional variance into the result, we compare cocos and consel under different seed for the
 same inputs.

 .PARAM InputFiles
 A list of RAxML-NG *.siteLH files to run against both tools. The files must be located in the data/ directory
 of the bench crate.

 .PARAM Repeats
 How often to run consel and cocos against each input file.
#>
param(
    [Parameter(Mandatory = $false)]
    [string[]] $InputFiles = @("flat-phylo.siteLH", "medium.siteLH"),

    [Parameter(Mandatory = $false)]
    [int] $Repeats = 10
)

$SITE_LH_RECORD_SPLIT = "    "

# Make scratch space
$ScratchDir = [System.IO.Path]::Combine($PSScriptRoot, "scratch")
New-Item -ItemType Directory $ScratchDir -ErrorAction SilentlyContinue > $null

# Install and load consel
& "$PSScriptRoot/installconsel.ps1"
Import-Module "$PSScriptRoot/modconsel"

# Make sure we optimize cocos
$env:RUSTFLAGS="-C target-cpu=native"

# For each input file, create resamplings and then run those against consel and cocos
foreach ($InputName in $InputFiles) {
    $InputPath = [System.IO.Path]::Combine($PSScriptRoot, "..", "data", $InputName)
    if (-not (Test-Path $InputPath)) {
        Write-Error "Input file $InputName not found at $InputPath."
        exit 1
    }

    $ConselResults = @{}
    $CocosResults = @{}

    foreach ($rep in 1..$Repeats) {
        # Prepare Paths for use in WSL
        $ScratchFile = [System.IO.Path]::Combine($ScratchDir, "temp")

        # Run consel
        (Invoke-Consel -InputPath $InputPath -OutputPath $ScratchFile -Seed $rep) | `
            ForEach-Object {
                Write-Host $_
                exit 1
                if (-not ($ConselResults.Contains($_.item))) {
                    $ConselResults[$item] = @($_.au)
                } else {
                    $ConselResults[$item] += $_.au
                }
            }

        # Run cocos
        $Results = (cargo --quiet run --release --bin cocos -- -i $InputPath -o - -s $rep) | `
            Select-Object -Skip 1 | `
            ForEach-Object {$Index = 0}{
                $Columns = $_ -split "`t"
                $au = [float]$Columns[1]

                if (-not ($CocosResults.Contains($Index))) {
                    $CocosResults[$Index] = @($au)
                } else {
                    $CocosResults[$Index] += $au
                }

                $Index++
            }
    }

    # calculate standard deviation of all trees of consel au vales
    $ConselDeviations = $ConselResults.GetEnumerator() | ForEach-Object {
        $Values = $_.Value

        ($Values | Measure-Object -StandardDeviation).StandardDeviation
    }

    # calculate standard deviations of all trees of cocos au values
    $CocosDeviations = $CocosResults.GetEnumerator() | ForEach-Object {
        $Values = $_.Value

        ($Values | Measure-Object -StandardDeviation).StandardDeviation
    }

    Write-Host "Deviations Consel:"
    $ConselDeviations | Measure-Object -AllStats

    Write-Host "Deviations Cocos:"
    $CocosDeviations | Measure-Object -AllStats
}

