<#
安装/查看一日隔夜数据计划任务。只启动本项目Python，不创建Codex会话或依赖应用界面。
使用当前用户的交互登录令牌，不读取或保存Windows密码；电脑需已登录、开机并联网。
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
    throw '采集目录必须属于当前项目.local-runs，避免对错误目录写入。'
}
if ((Get-TimeZone).Id -ne 'China Standard Time') {
    throw '计划时刻采用北京时间；当前Windows时区不匹配，未创建任务。'
}
$taskName = 'FundRadar-Direction1d-SPX-Overnight'
$taskPath = '\'
$pythonWindowless = Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
$entry = Join-Path $PSScriptRoot 'direction_1d_overnight_collect.py'
if (-not (Test-Path -LiteralPath $pythonWindowless) -or -not (Test-Path -LiteralPath (Join-Path $dataRoot 'contract.json'))) {
    throw '虚拟环境或留档契约未准备好。'
}
$existing = Get-ScheduledTask -TaskName $taskName -TaskPath $taskPath -ErrorAction SilentlyContinue
if ($Mode -eq 'Install') {
    if ($existing) { throw '同名任务已存在，请先查看，不覆盖已有任务。' }
    $arguments = '-B "' + $entry + '" tick --run-dir "' + $dataRoot + '"'
    $action = New-ScheduledTaskAction -Execute $pythonWindowless -Argument $arguments -WorkingDirectory $projectRoot
    $triggers = @(
        New-ScheduledTaskTrigger -Daily -At '07:30'
        New-ScheduledTaskTrigger -Daily -At '07:50'
        New-ScheduledTaskTrigger -Daily -At '07:58'
        New-ScheduledTaskTrigger -Daily -At '08:05'
        New-ScheduledTaskTrigger -AtLogOn -User ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name)
    )
    # 系统补触发时由Python按真实时间判定；不会补写错过的早间时刻。pythonw启动无可见控制台。
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 3)
    $principal = New-ScheduledTaskPrincipal -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
        -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -TaskPath $taskPath -Action $action -Trigger $triggers `
        -Settings $settings -Principal $principal -Description '基金雷达SPX隔夜信息本机留档：7:30/7:50/7:58采集，8:05检查；中国非交易日跳过，晚到不补为准时。' | Out-Null
} elseif ($Mode -eq 'Run') {
    if (-not $existing) { throw '计划任务尚未安装。' }
    Start-ScheduledTask -TaskName $taskName -TaskPath $taskPath
} elseif ($Mode -eq 'Disable') {
    if (-not $existing) { throw '计划任务不存在。' }
    Disable-ScheduledTask -TaskName $taskName -TaskPath $taskPath | Out-Null
}
$task = Get-ScheduledTask -TaskName $taskName -TaskPath $taskPath
$info = Get-ScheduledTaskInfo -TaskName $taskName -TaskPath $taskPath
[pscustomobject]@{
    TaskName = $task.TaskName
    State = [string]$task.State
    NextRunTime = $info.NextRunTime.ToString('o')
    LastRunTime = $info.LastRunTime.ToString('o')
    LastTaskResult = $info.LastTaskResult
    TriggerCount = $task.Triggers.Count
    Execute = $task.Actions.Execute
    Arguments = $task.Actions.Arguments
    LogonType = [string]$task.Principal.LogonType
    StartWhenAvailable = $task.Settings.StartWhenAvailable
} | ConvertTo-Json -Depth 4
