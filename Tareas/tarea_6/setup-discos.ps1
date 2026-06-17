# setup-discos.ps1
# Crea los discos para nodo3, nodo4, nodo5 y nodo6
# copiando el disco de nodo1 (que ya tiene Python y Alpine configurados).
#
# Ejecutar UNA SOLA VEZ desde Windows, con las VMs apagadas.
#
# Uso:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup-discos.ps1

$BASE  = Join-Path $PSScriptRoot "qemu-cluster-demo"
$DISKS = "$BASE\disks"
$QEMU  = "$BASE\qemu\qemu-img.exe"

Write-Host "Creando discos adicionales en: $DISKS"
Write-Host ""

foreach ($n in 3..6) {
    $src = "$DISKS\nodo1.qcow2"
    $dst = "$DISKS\nodo$n.qcow2"

    if (Test-Path $dst) {
        Write-Host "nodo$n.qcow2 ya existe, omitiendo."
        continue
    }

    Write-Host "Creando nodo$n.qcow2 (copia de nodo1)..."
    Copy-Item -Path $src -Destination $dst
    Write-Host "  OK -> $dst"
}

Write-Host ""
Write-Host "Listo. Discos disponibles:"
Get-ChildItem "$DISKS\*.qcow2" | ForEach-Object { Write-Host "  $_" }
