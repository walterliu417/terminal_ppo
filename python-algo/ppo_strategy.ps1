$scriptPath = Split-Path -parent $PSCommandPath;
$algoPath = "$scriptPath\ppo_strategy.py"

python $algoPath
