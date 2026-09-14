<#
独立120日试验任务。安装必须已有通过P0和全名单输入验收的active contract。
不覆盖SPX任务，不保存Windows密码；以当前交互账户运行，依赖开机、登录和联网。
#>
param(
    [ValidateSet('Install', 'Status', 'Run', 'Disable')]
    [string]$Mode = 'Status',
    [Parameter(Mandatory = $true)]
    [string]$RunDirectory
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$dataRoot = (Resolve-Path -LiteralPath $RunDirectory).Path
$allowedRoot = [System.IO.Path]::GetFullPath((Join-Path $projectRoot '.local-runs')) + [System.IO.Path]::DirectorySeparatorChar
if (-not $dataRoot.StartsWith($allowedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw '试验目录必须属于当前项目.local-runs。'
}
if ((Get-TimeZone).Id -ne 'China Standard Time') { throw '当前Windows时区不是北京时间，停止任务操作。' }
$taskName = 'FundRadar-Direction1d-Independent-Validation'
$entry = Join-Path $PSScriptRoot 'direction_1d_independent.py'
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$pythonWindowless = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($Mode -eq 'Install') {
    if ($existing) { throw '同名任务已存在，未覆盖。' }
    Push-Location -LiteralPath $projectRoot
    try {
        & $python -B $entry active-check --run-dir $dataRoot
        if ($LASTEXITCODE -ne 0) { throw '完整日历、模型、输入验收和启动契约尚未全部通过，未安装任务。' }
    } finally { Pop-Location }
    $arguments = '-B "' + $entry + '" tick --run-dir "' + $dataRoot + '"'
    $action = New-ScheduledTaskAction -Execute $pythonWindowless -Argument $arguments -WorkingDirectory $projectRoot
    $triggers = @(
        New-ScheduledTaskTrigger -Daily -At '18:30'
        New-ScheduledTaskTrigger -Daily -At '22:30'
        New-ScheduledTaskTrigger -Daily -At '07:30'
        New-ScheduledTaskTrigger -Daily -At '08:10'
        New-ScheduledTaskTrigger -Daily -At '08:20'
        New-ScheduledTaskTrigger -Daily -At '08:35'
        New-ScheduledTaskTrigger -AtLogOn -User ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name)
    )
    # 08:35和登录补触发仅按当前窗口处理；错过的答案留缺失，不回拨时钟。
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 9)
    $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
        -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $triggers -Settings $settings `
        -Principal $principal -Description '一日ACTIVITY12与原7项模型固定未来对照；120计划目标日及10日核对宽限，缺失不顺延。' | Out-Null
} elseif ($Mode -eq 'Run') {
    if (-not $existing) { throw '任务尚未安装。' }
    Start-ScheduledTask -TaskName $taskName
} elseif ($Mode -eq 'Disable') {
    if (-not $existing) { throw '任务尚未安装。' }
    Disable-ScheduledTask -TaskName $taskName | Out-Null
}
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {
    $info = Get-ScheduledTaskInfo -TaskName $taskName
    [pscustomobject]@{ TaskName = $taskName; State = [string]$task.State; LastTaskResult = $info.LastTaskResult;
        NextRunTime = $info.NextRunTime.ToString('o'); TriggerCount = $task.Triggers.Count } | ConvertTo-Json
} else {
    [pscustomobject]@{ TaskName = $taskName; State = 'NOT_INSTALLED' } | ConvertTo-Json
}
