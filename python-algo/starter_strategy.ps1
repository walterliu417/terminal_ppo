$scriptPath = Split-Path -parent $PSCommandPath;
$algoPath = "$scriptPath\starter_strategy.py"

python $algoPath
