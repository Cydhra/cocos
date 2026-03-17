<#
.SYNOPSIS
Converts a path to a linux-compatible path.

.DESCRIPTION
Converts a path to a WSL-compatible path on windows, and leaves it as is on Linux.
This prepares the path to be used in a linux environment natively accessible by the platform,
but it cannot be used to prepare paths on windows to be used natively in a linux environment remotely.

.PARAMETER Path
A native path
#>
function ConvertTo-LinuxPath {
    param (
        [Parameter(Mandatory = $true)]
        [string] $Path
    )

    if (-not $IsLinux -and -not $IsWindows) {
        throw "This command supports only Windows and Linux"
    }

    if ($IsLinux) {
        return $Path
    }

    return (wsl wslpath -a "'$path'")
}

<#
.SYNOPSIS
Execute a provided command line on linux.

.DESCRIPTION
Runs a provided executable natively on linux if the current platform is Linux,
or runs it in WSL if the platform is Windows.
Paths provided as parameters need to be converted before calling this function using `ConvertTo-LinuxPath`

.PARAMETER Path
Linux-compatible path to the executable.

.PARAMETER CommandLine
The full command line provided to the executable. Beware that powershell may interpret some arguments differently
than linux shells, so they have to be escaped to be properly passed by the linux shell.
#>
function Invoke-OnLinux {
    param (
        [Parameter(Mandatory = $true)]
        [string] $Path,

        [Parameter(Mandatory = $false, ValueFromRemainingArguments)]
        [string[]] $CommandLine
    )

    if (-not $IsLinux -and -not $IsWindows) {
        throw "This command supports only Windows and Linux"
    }

    if ($IsLinux) {
        return & $Path $CommandLine
    } else {
        return wsl $Path $CommandLine
    }
}

function Get-MakermtPath {
    return ConvertTo-LinuxPath -Path ([System.IO.Path]::Combine($PSScriptRoot, "bin", "makermt"))
}

function Get-ConselPath {
    return ConvertTo-LinuxPath -Path ([System.IO.Path]::Combine($PSScriptRoot, "bin", "consel"))
}

function Get-CatpvPath {
    return ConvertTo-LinuxPath -Path ([System.IO.Path]::Combine($PSScriptRoot, "bin", "catpv"))
}


<#
 .SYNOPSIS
 Invokes the consel program makermt.

 .DESCRIPTION
 Converts all provided paths into linux-compatible paths and then calls consel's makermt binary with the provided
 parameters.

 .PARAMETER sitelh
 Path of the sitelh file containing the per-site log-likelihoods used for bootstrapping. The "format" parameter
 controls which file format is expected for the sitelh file, and defaults to the tree-puzzle format, which is also
 used by raxml-ng. The extension ".sitelh" can be ommitted.

 .PARAMETER output
 Output file name pattern. Makermt will generate two output files, which will use the provided output pattern and
 the respective file ending. Do not append a file ending to the output pattern. One of the files will be the
 "output.rmt" file required for invoking consel.

 .PARAMETER format
 File format to use for the log-likelihoods file. Defaults to "puzzle", which is the format used by tree-puzzle and
 raxml-ng. Further supports "molphy", "paml", "paup", and "pyhml".

 .PARAMETER Rescaling
 If set, the rescaling approximation is used, which limits the bootstraping to single-scale bootstrap at scale
 factor 1.
#>
function Invoke-Makermt {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Sitelh,

        [Parameter(Mandatory = $true)]
        [string] $Output,

        [ValidateSet("molphy", "paml", "paup", "puzzle", "phyml", IgnoreCase = $true)]
        [Parameter(Mandatory = $false)]
        [string] $Format = "puzzle",

        [switch] $Rescaling
    )

    $LinuxSiteLh = ConvertTo-LinuxPath -Path $Sitelh
    $LinuxOutput = ConvertTo-LinuxPath -Path $Output

    $RemainingArgs = @()

    if ($Rescaling) {
        $RemainingArgs += "-f"
    }

    Invoke-OnLinux -Path (Get-MakermtPath) "--$Format" $LinuxSiteLh $LinuxOutput @RemainingArgs
}

<#
 .SYNOPSIS
 Invokes the consel program.

 .DESCRIPTION
 Converts all provided paths into linux-compatible paths and then calls the consel binary with the provided
 parameters.

 .PARAMETER rmt
 Path of the rmt file containing the bootstrap values created by makermt. The extension ".rmt" can be ommitted.

 .PARAMETER output
 Output file name pattern. Makermt will generate two output files, which will use the provided output pattern and
 the respective file ending. Do not append a file ending to the output pattern.

  .PARAMETER Rescaling
 If set, the rescaling approximation is used, which extrapolates multiscale bootstrap results from a single bootstrap run.
#>
function Invoke-Consel {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Rmt,

        [Parameter(Mandatory = $true)]
        [string] $Output,

        [switch] $Rescaling
    )

    $LinuxRmt = ConvertTo-LinuxPath -Path $Rmt
    $LinuxOutput = ConvertTo-LinuxPath -Path $Output

    $RemainingArgs = @()

    if ($Rescaling) {
        $RemainingArgs += "-f"
    }

    Invoke-OnLinux -Path (Get-ConselPath) $LinuxRmt $LinuxOutput "--no_sort" "--no_bp" "--no_pp" "--no_sh" @RemainingArgs
}

<#
 .SYNOPSIS
 Imports the contents of the consel pv file as PS objects.

 .DESCRIPTION
 Converts the provided path into a linux-compatible path and then calls the consel binary "catpv" with the provided
 pv path. Then, it converts the output of "catpv" into a properly formatted Powershell objects containing all
 properties of the PV table.

 .PARAMETER InputPv
 The path to the "*.pv" file generated by Invoke-Consel.

 .EXAMPLE
 # Convert the pv file into a CSV file
 Import-Pv -InputPv "gene1\ranks.pv" | Export-Csv -NoTypeInformation -UseQuotes -Path "gene1\ranks.csv"
#>
function Import-Pv {
    param(
        [Parameter(Mandatory)]
        [string] $InputPv
    )

    $LinuxPv = ConvertTo-LinuxPath -Path $InputPv
    Invoke-OnLinux -Path (Get-CatpvPath) $LinuxPv | `
        Where-Object { -not [string]::IsNullOrEmpty($_) } | `
        Select-Object -Skip 2 | `
        ForEach-Object {
            $cols = $_ -split '\s{1,}'
            [PSCustomObject]@{
                rank = [int]$cols[1]
                item = [int]$cols[2]
                obs = [float]$cols[3]
                au = [float]$cols[4]
                np = [float]$cols[5]
            }
        }
}

Export-ModuleMember -Function Invoke-Makermt
Export-ModuleMember -Function Invoke-Consel
Export-ModuleMember -Function Import-Pv