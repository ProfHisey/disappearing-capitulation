# commit_v92.ps1 - one-shot commit and push of the v9.2 paper artifacts.
# Run from anywhere:  powershell -ExecutionPolicy Bypass -File commit_v92.ps1
# It operates on the repo folder this script sits in (E:\Finance\Capitulation).

# Native git writes progress to stderr; do not let that abort the script.
$ErrorActionPreference = "Continue"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
Set-Location -LiteralPath $PSScriptRoot

function Step($msg) { Write-Host ""; Write-Host "== $msg" -ForegroundColor Cyan }
function Fail($msg) { Write-Host ""; Write-Host "STOPPED: $msg" -ForegroundColor Red; Read-Host "Press Enter to close"; exit 1 }

# --- 0. Word must not be holding the draft open ---------------------------
$lock = Join-Path $PSScriptRoot "paper\~`$pitulation_draft_v9_2.docx"
if (Test-Path -LiteralPath $lock) {
  Write-Host "Word still has capitulation_draft_v9_2.docx open (lock file present)." -ForegroundColor Yellow
  $ans = Read-Host "Close it in Word, then press Enter to continue, or type s to skip this check"
  if ($ans -ne "s" -and (Test-Path -LiteralPath $lock)) {
    Fail "lock file is still there. Close the document in Word and re-run."
  }
}

# --- 0b. git must be willing to touch a repo on E:\ ------------------------
# E:\ does not record POSIX ownership, so git refuses the repo until it is
# named as a safe.directory. Detect that and fix it once, with consent.
$repoSlash = $PSScriptRoot -replace '\\', '/'
git -C $PSScriptRoot rev-parse --is-inside-work-tree 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
  $probe = (git -C $PSScriptRoot status 2>&1) -join " "
  if ($probe -match "dubious ownership") {
    Write-Host ""
    Write-Host "git will not touch this repo yet: E:\ does not record ownership," -ForegroundColor Yellow
    Write-Host "so git treats the folder as untrusted. The one-time fix is:" -ForegroundColor Yellow
    Write-Host "  git config --global --add safe.directory $repoSlash"
    $ok = Read-Host "Add that exception now? (y/n)"
    if ($ok -eq "y") {
      git config --global --add safe.directory $repoSlash
      if ($LASTEXITCODE -ne 0) { Fail "could not write the git config exception." }
      Write-Host "Added." -ForegroundColor Green
    } else {
      Fail "run the git config line above, then re-run this script."
    }
  } else {
    Fail ("git cannot read this repo: " + $probe)
  }
}

# --- 1. the files this commit is for --------------------------------------
$files = @(
  "paper\make_draft_v9_2.js",
  "paper\capitulation_draft_v9_2.docx",
  "paper\figs\fig1.png",
  "paper\figs\fig2.png",
  "paper\figs\fig3.png",
  "paper\figs\fig4.png",
  "paper\figs\fig5.png",
  "paper\figs\fig6.png",
  ".gitignore"
)
$missing = $files | Where-Object { -not (Test-Path -LiteralPath (Join-Path $PSScriptRoot $_)) }
if ($missing) { Fail ("these files are not on disk: " + ($missing -join ", ")) }

# --- 2. pull first (the standing ritual) ----------------------------------
Step "git pull --ff-only"
git pull --ff-only
if ($LASTEXITCODE -ne 0) {
  Fail "pull failed or the branch has diverged. Resolve by hand (git status, git log --oneline --graph -10) before committing."
}

# --- 3. stage, show, commit ----------------------------------------------
Step "git add"
git add -- $files
if ($LASTEXITCODE -ne 0) { Fail "git add failed." }

Step "what is staged"
git status --short

$staged = git diff --cached --name-only
if (-not $staged) {
  Write-Host ""
  Write-Host "Nothing to commit - these files are already in the repo at this content." -ForegroundColor Yellow
  Read-Host "Press Enter to close"
  exit 0
}

$msg = "v9.2: figures embedded, Fig 4 into section 3, section 8 softened per 37d; builder reconstructed and byte-verified"
Step "git commit"
git commit -m $msg
if ($LASTEXITCODE -ne 0) { Fail "commit failed." }

# --- 4. push --------------------------------------------------------------
Step "git push"
git push
if ($LASTEXITCODE -ne 0) { Fail "push failed (credentials or network). The commit is safe locally - just re-run git push." }

Step "done"
git log --oneline -3
Write-Host ""
Write-Host "v9.2 artifacts are committed and pushed." -ForegroundColor Green
Read-Host "Press Enter to close"
