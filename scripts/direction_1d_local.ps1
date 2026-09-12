<#
本地1日实验后台入口，不部署生产、不发送通知。
Start 只启动未占用端口；Stop 只停止此脚本记录且创建时刻匹配的进程。
依赖沿用两个工程已有.env；不在脚本、命令行或状态文件保存密码。
关闭浏览器不影响后台，电脑关机或服务退出不会补造当时预测。
#>
param(
    [ValidateSet('Start','Status','Stop')][string]$Action = 'Status',
    [string]$JavaRoot = 'C:\ideaProject\workSpace12',
    [int]$PythonPort = 8000,
    [int]$JavaPort = 8080,
    # 本地验收可短时缩短轮询；这是作业收取频率，同一目标日仍只留一份原始预测。
    [ValidatePattern('^PT[1-9][0-9]*[SMH]$')][string]$CheckInterval = 'PT30M',
    [switch]$DisableExperiment
)
$ErrorActionPreference = 'Stop'
$pythonRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$javaProject = (Resolve-Path -LiteralPath $JavaRoot).Path
$runtimeDir = Join-Path $pythonRoot '.local-runs\direction-1d-runtime'
[IO.Directory]::CreateDirectory($runtimeDir) | Out-Null
$stateFile = Join-Path $runtimeDir 'local-processes.json'

function Read-State {
    if (Test-Path -LiteralPath $stateFile) { return Get-Content -LiteralPath $stateFile -Raw -Encoding UTF8 | ConvertFrom-Json }
    return @()
}
function Test-OwnedProcess($entry) {
    $process = Get-Process -Id $entry.pid -ErrorAction SilentlyContinue
    return $process -and $process.StartTime.ToUniversalTime().Ticks -eq ([DateTimeOffset]$entry.startedAt).UtcDateTime.Ticks
}
if ($Action -eq 'Status') {
    foreach ($entry in @(Read-State)) {
        [pscustomobject]@{ Service=$entry.service; PID=$entry.pid; Port=$entry.port; Running=[bool](Test-OwnedProcess $entry); StartedAt=$entry.startedAt }
    }
    exit 0
}
if ($Action -eq 'Stop') {
    foreach ($entry in @(Read-State)) {
        if (Test-OwnedProcess $entry) { Stop-Process -Id $entry.pid -ErrorAction Stop }
    }
    exit 0
}
foreach ($port in @($PythonPort,$JavaPort)) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { throw "端口 $port 已占用；保留现有进程，请先核对归属。" }
}
$pythonExe = Join-Path $pythonRoot '.venv\Scripts\python.exe'
$javaExe = Join-Path $javaProject '.tools\jdk17\jdk-17.0.20.1+1\bin\java.exe'
$builtJar = Join-Path $javaProject 'target\fund-core-0.1.0-SNAPSHOT.jar'
foreach ($path in @($pythonExe,$javaExe,$builtJar)) { if (!(Test-Path -LiteralPath $path)) { throw "缺少运行产物：$path" } }
# 使用独立副本运行，避免Windows占用target产物阻塞以后的编译。
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$runJar = Join-Path $runtimeDir "fund-core-$stamp.jar"
Copy-Item -LiteralPath $builtJar -Destination $runJar
$experiment = if ($DisableExperiment) { 'false' } else { 'true' }
$processes = @()
$launches = @(
    @{ service='python'; exe=$pythonExe; args=@('-B','-m','uvicorn','app.main:app','--host','127.0.0.1','--port',"$PythonPort"); cwd=$pythonRoot; port=$PythonPort; url="http://127.0.0.1:$PythonPort/health" },
    @{ service='java'; exe=$javaExe; args=@('-Dfile.encoding=UTF-8','-jar',('"'+$runJar+'"'),"--server.port=$JavaPort","--server.address=127.0.0.1","--ai.service.base-url=http://127.0.0.1:$PythonPort","--direction1d.enabled=$experiment","--direction1d.fixed-delay=$CheckInterval"); cwd=$javaProject; port=$JavaPort; url="http://127.0.0.1:$JavaPort/actuator/health" }
)
foreach ($item in $launches) {
    $launched = Start-Process -FilePath $item.exe -ArgumentList $item.args -WorkingDirectory $item.cwd -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtimeDir "$($item.service)-$stamp.out.log") `
        -RedirectStandardError (Join-Path $runtimeDir "$($item.service)-$stamp.err.log")
    $ready = $false
    for ($attempt=0; $attempt -lt 30; $attempt++) {
        try {
            if ($item.service -eq 'python') {
                Push-Location $pythonRoot
                try {
                    # 服务令牌由Python在内存读取；不回显、不经过命令行参数。
                    & $pythonExe -B -c 'import sys,httpx; from app.core.config import get_settings; r=httpx.get(sys.argv[1],headers={"X-Service-Token":get_settings().ai_service_token.get_secret_value()},timeout=2); sys.exit(0 if r.status_code==200 else 1)' "http://127.0.0.1:$PythonPort/internal/v1/health" 2>$null
                    $ready = $LASTEXITCODE -eq 0
                } finally { Pop-Location }
            } else { $response = Invoke-WebRequest -Uri $item.url -TimeoutSec 2; $ready=$response.StatusCode -eq 200 }
            if ($ready) { break }
        } catch { }
        Start-Sleep -Milliseconds 1000
    }
    $listener = Get-NetTCPConnection -LocalPort $item.port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    $runningPid = if ($listener) { $listener.OwningProcess } else { $launched.Id }
    $running = Get-Process -Id $runningPid -ErrorAction SilentlyContinue
    if ($running) {
        $processes += [pscustomobject]@{ service=$item.service; pid=$runningPid; port=$item.port; startedAt=$running.StartTime.ToUniversalTime().ToString('o'); healthy=$ready }
        ConvertTo-Json -InputObject @($processes) | Set-Content -LiteralPath $stateFile -Encoding UTF8
    }
    if (!$ready) { throw "$($item.service) 健康检查未通过；已保存进程和日志，未启动后续服务。" }
}
$processes | Select-Object service,pid,port,healthy
