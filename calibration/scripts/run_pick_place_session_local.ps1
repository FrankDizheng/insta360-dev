param(
    [string]$BridgeUrl = "http://10.13.167.212:8765",
    [string]$Handeye = "",
    [string]$Tcp = "",
    [string]$OutputDir = "",
    [string]$ScanPose = "-19.4,10.7,4.6,63.0,7.1,1.4,56.6",
    [double]$StandoffMm = 80,
    [double]$GraspZMm = 300,
    [double]$GraspRetryStepMm = 15,
    [int]$GraspRetryCount = 2,
    [double]$LiftZMm = 400,
    [double]$PlaceZMm = 295,
    [double]$ZSafeMm = 400,
    [double]$GripperWidth = 0.06,
    [double]$GripperForce = 1.0,
    [double]$PickLocalXOffsetMm = 0,
    [double]$PickLocalYOffsetMm = 0,
    [double]$PickLocalZOffsetMm = 0,
    [int]$Speed = 10,
    [int]$FlushFrames = 2,
    [switch]$ShowHelp,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
$sessionScript = Join-Path $repoRoot "insta360-dev\calibration\scripts\pick_place_session.py"

if (-not (Test-Path $pythonExe)) {
    throw "Local venv not found: $pythonExe`nRun .\setup_local_env.ps1 first."
}

if (-not $Handeye) {
    $Handeye = Join-Path $repoRoot "insta360-dev\calibration\results\session1\handeye_result.json"
}

if (-not $Tcp) {
    $Tcp = Join-Path $repoRoot "insta360-dev\calibration\results\session1\gripper_tcp_left_front_tip_samples_004_006.json"
}

if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "session_local"
}

$argsList = @(
    $sessionScript
    "--bridge-url", $BridgeUrl
    "--handeye", $Handeye
    "--tcp", $Tcp
    "--output-dir", $OutputDir
    "--scan-pose=$ScanPose"
    "--standoff-mm", $StandoffMm
    "--grasp-z-mm", $GraspZMm
    "--grasp-retry-step-mm", $GraspRetryStepMm
    "--grasp-retry-count", $GraspRetryCount
    "--lift-z-mm", $LiftZMm
    "--place-z-mm", $PlaceZMm
    "--z-safe-mm", $ZSafeMm
    "--gripper-width", $GripperWidth
    "--gripper-force", $GripperForce
    "--pick-local-x-offset-mm", $PickLocalXOffsetMm
    "--pick-local-y-offset-mm", $PickLocalYOffsetMm
    "--pick-local-z-offset-mm", $PickLocalZOffsetMm
    "--speed", $Speed
    "--flush-frames", $FlushFrames
)

if ($RemainingArgs) {
    $argsList += $RemainingArgs
}

if ($ShowHelp) {
    $argsList += "--help"
}

Write-Host "[local-run] python: $pythonExe"
Write-Host "[local-run] script: $sessionScript"
Write-Host "[local-run] bridge: $BridgeUrl"

& $pythonExe @argsList
