# eubi's GUI with micro-glancer in the Inspect page, for local testing (nothing is
# installed into eubi's venv): micro-glancer's python folder goes on PYTHONPATH.
# Build micro-glancer first (in this folder): .venv\Scripts\maturin develop --release
#
# usage:   .\eubi-dev.ps1 [PATH]
#          $env:EUBI_VIEWER = "classic"; .\eubi-dev.ps1     # the classic viewer
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$eubi = Resolve-Path (Join-Path $here "..\..")
$env:PYTHONPATH = (Join-Path $here "python")
& (Join-Path $eubi "venv\Scripts\eubi-gui.exe") @args
