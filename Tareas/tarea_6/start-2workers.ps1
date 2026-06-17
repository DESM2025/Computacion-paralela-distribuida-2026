# start-2workers.ps1  —  nodo0 + nodo1 + nodo2  (2 workers)
#
# Uso:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\start-2workers.ps1

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

# nodo1
Start-Process -FilePath $QEMU -ArgumentList ($COMMON + @(
    "-name",   "nodo1",
    "-drive",  "file=$DISKS\nodo1.qcow2,format=qcow2,if=virtio",
    "-netdev", "user,id=net0,hostfwd=tcp:127.0.0.1:2221-:22,hostfwd=tcp:127.0.0.1:9001-:9000",
    "-device", "virtio-net-pci,netdev=net0"
))

# nodo2
Start-Process -FilePath $QEMU -ArgumentList ($COMMON + @(
    "-name",   "nodo2",
    "-drive",  "file=$DISKS\nodo2.qcow2,format=qcow2,if=virtio",
    "-netdev", "user,id=net0,hostfwd=tcp:127.0.0.1:2222-:22,hostfwd=tcp:127.0.0.1:9002-:9000",
    "-device", "virtio-net-pci,netdev=net0"
))

Write-Host "Iniciando cluster con 2 workers (nodo1, nodo2)..."
Write-Host "SSH nodo0: ssh root@127.0.0.1 -p 2220"
Write-Host "Espera ~30 segundos para que arranquen las VMs."
