<#
三天研究的独立本地执行器。保留旧SPX及120日任务，不触发交易或业务库写入。
以当前登录账户运行；需要电脑开机、登录、联网，Docker和数据库可用。
pythonw隐藏控制台；互斥锁和IgnoreNew避免定时与人工运行互相覆盖。
#>
param([ValidateSet('Install','Status','Run','Disable')][string]$Mode='Status')
$ErrorActionPreference='Stop'
$projectRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$dataRoot=Join-Path $projectRoot '.local-runs\direction-1d-sprint-20260914'
$taskName='FundRadar-Direction1d-ThreeDayResearch-20260914'
$entry=Join-Path $PSScriptRoot 'direction_1d_sprint.py'
$python=Join-Path $projectRoot '.venv\Scripts\pythonw.exe'
$existing=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if($Mode -eq 'Install') {
    if($existing){throw '同名三天研究任务已经存在，未覆盖。'}
    if((Get-TimeZone).Id -ne 'China Standard Time'){throw '系统时区必须为北京时间。'}
    if(-not(Test-Path -LiteralPath (Join-Path $dataRoot 'round-01\result.json'))){throw '首轮训练尚未完成。'}
    $spec=(Get-Content -LiteralPath (Join-Path $dataRoot 'protocol.json') -Raw -Encoding UTF8 | ConvertFrom-Json).payload
    $deadline=[DateTimeOffset]::Parse($spec.deadline_at).LocalDateTime
    if($deadline -le (Get-Date)){throw '研究窗口已经结束。'}
    $action=New-ScheduledTaskAction -Execute $python -Argument ('-B "'+$entry+'" tick') -WorkingDirectory $projectRoot
    $trigger=New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
        -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration ($deadline-(Get-Date))
    # 三天截止后最后一次只生成检查点报告，Python入口不会继续调用供应商。
    $finalTrigger=New-ScheduledTaskTrigger -Once -At $deadline.AddMinutes(1)
    $logonTrigger=New-ScheduledTaskTrigger -AtLogOn -User ([Security.Principal.WindowsIdentity]::GetCurrent().Name)
    $settings=New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
    $principal=New-ScheduledTaskPrincipal -UserId ([Security.Principal.WindowsIdentity]::GetCurrent().Name) `
        -LogonType Interactive -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($trigger,$finalTrigger,$logonTrigger) `
        -Settings $settings -Principal $principal `
        -Description '三天一日模型研究：提前记录预测、到期核对；新增费用0元；截止后只生成报告。' | Out-Null
} elseif($Mode -eq 'Run') {
    if(-not $existing){throw '任务尚未安装。'}
    Start-ScheduledTask -TaskName $taskName
} elseif($Mode -eq 'Disable') {
    if($existing){Disable-ScheduledTask -TaskName $taskName | Out-Null}
}
$task=Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if($task){
    $info=Get-ScheduledTaskInfo -TaskName $taskName
    [pscustomobject]@{TaskName=$taskName;State=[string]$task.State;LastTaskResult=$info.LastTaskResult;
        NextRunTime=$info.NextRunTime.ToString('o');TriggerCount=$task.Triggers.Count}|ConvertTo-Json
} else { [pscustomobject]@{TaskName=$taskName;State='NOT_INSTALLED'}|ConvertTo-Json }
