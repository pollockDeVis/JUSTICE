$A = Get-ChildItem -recurse -filter *.py | Select-String -Pattern 'TODO' -List
$pattern = (".venv|.ipynb_checkpoints")
$A = $A | Where-Object {$_ -notmatch $pattern}
$A
Write-host ""
Write-host "----------------------"
Write-host -NoNewline ("Number of TODOs:::", $A.Count)