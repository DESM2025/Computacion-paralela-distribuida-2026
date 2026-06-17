# start-4workers.ps1  —  nodo0 + nodo1..nodo4  (4 workers)
#
# Requiere haber ejecutado setup-discos.ps1 antes.
#
# Uso:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\start-4workers.ps1

$BASE  = Join-Path $PSScriptRoot "qemu-cluster-demo"
$QEMU  = "$BASE\qemu\qemu-system-x86_64.exe"
$DISKS = "$BASE\disks"

$COMMON = @("-m", "256M", "-smp", "1")

Stop-Process -Name qemu-system-x86_64 -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# nodo0 — coordinador
Start-Process -FilePath $QEMU -ArgumentList ($COMMON + @(
    "-name",   "nodo0",
    "-drive",  "file=$DISKS\nodo0.qcow2,format=qcow2,if=virtio",
    "-netdev", "user,id=net0,hostfwd=tcp:127.0.0.1:2220-:22",
    "-device", "virtio-net-pci,netdev=net0"
))

# nodo1..nodo4
$workers = @(
    @{n=1; ssh=2221; task=9001},
    @{n=2; ssh=2222; task=9002},
    @{n=3; ssh=2223; task=9003},
    @{n=4; ssh=2224; task=9004}
)

foreach ($w in $workers) {
    Start-Process -FilePath $QEMU -ArgumentList ($COMMON + @(
        "-name",   "nodo$($w.n)",
        "-drive",  "file=$DISKS\nodo$($w.n).qcow2,format=qcow2,if=virtio",
        "-netdev", "user,id=net0,hostfwd=tcp:127.0.0.1:$($w.ssh)-:22,hostfwd=tcp:127.0.0.1:$($w.task)-:9000",
        "-device", "virtio-net-pci,netdev=net0"
    ))
}

Write-Host "Iniciando cluster con 4 workers (nodo1..nodo4)..."
Write-Host "SSH nodo0: ssh root@127.0.0.1 -p 2220"
Write-Host "Espera ~30 segundos para que arranquen las VMs."
