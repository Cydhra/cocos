function Invoke-Consel {
    param(
        [Parameter(Mandatory = $true)]
        [string] $InputPath,

        [Parameter(Mandatory = $true)]
        [string] $OutputPath,

        [Parameter(Mandatory = $false)]
        [int] $Seed = 0
    )

    function ConvertTo-LinuxPath {
        param (
            [string] $Path
        )

        (wsl wslpath -a "'$Path'")
    }

    $ConselDir = [System.IO.Path]::Combine($PSScriptRoot, "..", "consel")
    $MakermtBinaryWsl = ConvertTo-LinuxPath ([System.IO.Path]::Combine($ConselDir, "bin", "makermt"))
    $ConselBinaryWsl = ConvertTo-LinuxPath ([System.IO.Path]::Combine($ConselDir, "bin", "consel"))
    $CatPvBinaryWsl = ConvertTo-LinuxPath ([System.IO.Path]::Combine($ConselDir, "bin", "catpv"))

    $InputPathWsl = ConvertTo-LinuxPath $InputPath
    $OutputPathWsl = ConvertTo-LinuxPath $OutputPath

    if ($Seed -eq 0) {
        wsl $MakermtBinaryWsl --puzzle $InputPathWsl $OutputPathWsl | Out-Host
    } else {
        wsl $MakermtBinaryWsl --puzzle $InputPathWsl $OutputPathWsl -s $Seed | Out-Host
    }

    wsl $ConselBinaryWsl $OutputPathWsl.rmt $OutputPathWsl --no_bp --no_pp --no_sh | Out-Host
    $Result = (wsl $CatPvBinaryWsl $OutputPathWsl.pv) | `
        Where-Object { -not [string]::IsNullOrEmpty($_) } | `
        Select-Object -Skip 2 | `
        ForEach-Object {
            Write-Host $_
            $Columns = $_ -split '\s{1,}'
            [PSCustomObject]@{
                rank = [int]$Columns[1]
                item = [int]$Columns[2]
                obs = [float]$Columns[3]
                au = [float]$Columns[4]
                np = [float]$Columns[5]
                bp = [float]$Columns[7]
                pp = [float]$Columns[8]
                kh = [float]$Columns[9]
                sh = [float]$Columns[10]
                wkh = [float]$Columns[11]
                wsh = [float]$Columns[12]
            }
        }

    Return $Result
}



