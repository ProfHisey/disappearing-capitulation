# commit_selection_paper.ps1 - commit the "Selection Under Skewness" pilot work.
# Run:  powershell -ExecutionPolicy Bypass -File commit_selection_paper.ps1
#
# FIRST ACTION IS A COPYRIGHT GUARD. E:\Finance\Capitulation\literature holds
# publisher PDFs (Elsevier, Wiley, AEA, Cambridge, NBER). The repo is public.
# Those files must never be committed. This script adds literature/ to
# .gitignore and REFUSES to proceed if any PDF from it is already staged or
# tracked.

$ErrorActionPreference = "Continue"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
Set-Location -LiteralPath $PSScriptRoot

function Step($m) { Write-Host ""; Write-Host "== $m" -ForegroundColor Cyan }
function Fail($m) { Write-Host ""; Write-Host "STOPPED: $m" -ForegroundColor Red; Read-Host "Enter to close"; exit 1 }

# --- 0. git must be willing to touch a repo on E:\ ------------------------
git -C $PSScriptRoot rev-parse --is-inside-work-tree 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
  $repoSlash = $PSScriptRoot -replace '\\', '/'
  $probe = (git -C $PSScriptRoot status 2>&1) -join " "
  if ($probe -match "dubious ownership") {
    Write-Host "git needs a one-time exception for this drive:" -ForegroundColor Yellow
    Write-Host "  git config --global --add safe.directory $repoSlash"
    if ((Read-Host "Add it now? (y/n)") -eq "y") { git config --global --add safe.directory $repoSlash }
    else { Fail "run the line above, then re-run." }
  } else { Fail ("git cannot read this repo: " + $probe) }
}

# --- 1. COPYRIGHT GUARD ---------------------------------------------------
Step "copyright guard: literature/ must never be committed"
$gi = Join-Path $PSScriptRoot ".gitignore"
$ignore = @("", "# journal PDFs - copyright, never commit to a public repo", "literature/", "*.pdf")
$cur = if (Test-Path $gi) { Get-Content $gi } else { @() }
$added = $false
foreach ($line in $ignore) {
  if ($line -and ($cur -notcontains $line)) { Add-Content $gi $line; $added = $true }
}
if ($added) { Write-Host "  .gitignore updated (literature/ and *.pdf)" -ForegroundColor Green }
else { Write-Host "  .gitignore already covers it" }

$tracked = git ls-files "literature/*" "*.pdf"
if ($tracked) {
  Write-Host "  ALREADY TRACKED:" -ForegroundColor Red
  $tracked | ForEach-Object { Write-Host "    $_" }
  Write-Host "  Remove from the index (keeps your local copies) with:" -ForegroundColor Yellow
  Write-Host '    git rm --cached -r literature/'
  Fail "publisher PDFs are tracked. Untrack them before committing."
}
Write-Host "  clean - no PDFs tracked" -ForegroundColor Green

# --- 2. pull --------------------------------------------------------------
Step "git pull --ff-only"
git pull --ff-only
if ($LASTEXITCODE -ne 0) { Fail "pull failed or the branch diverged. Resolve by hand." }

# --- 3. stage the pilot work ---------------------------------------------
Step "staging"
$paths = @("pilot\41_fee_vs_performance_probe.py", "pilot\41b_nanigian_replication.py",
           "pilot\41c_rebuild_panel_with_categories.py", "pilot\43_fee_gradient_over_time.py",
           "pilot\44_fee_size_double_sort.py", "pilot\45_menu_simulation.py",
           "pilot\45b_horizon_and_survivorship.py", "pilot\46_naive_vs_aware_fee_sort.py",
           "pilot\47_ties_and_skewness.py", "pilot\48_breakeven_persistence.py",
           "pilot\49_decay_curve_long_horizon.py", "pilot\50_long_horizon_robustness.py",
           "pilot\51_post2000_headline.py", "pilot\52_gradient_inference.py",
           "pilot\53_factor_adjusted_gradient.py", "pilot\54_gradient_robustness.py",
           "pilot\55_stability_and_taxonomy.py", "pilot\56_final_inference.py",
           ".gitignore")
$have = $paths | Where-Object { Test-Path (Join-Path $PSScriptRoot $_) }
git add -- $have
git add -- "pilot\output\s4*.csv" "pilot\output\s5*.csv" 2>$null

Step "what is staged"
git status --short
$staged = git diff --cached --name-only
if (-not $staged) { Write-Host "nothing to commit." -ForegroundColor Yellow; Read-Host "Enter"; exit 0 }

# one more guard, after staging
if ($staged | Where-Object { $_ -match '\.pdf$' -or $_ -match '^literature/' }) {
  Fail "a PDF or a literature/ file got staged. Unstage it before committing."
}

# --- 4. commit and push ---------------------------------------------------
$msg = "Selection Under Skewness: pilot stages 41-56. Menu-size gradient in " +
       "risk-adjusted returns; -40bp/yr from 3 to 20 options per sleeve on " +
       "index-free menus, post-2000, 4-factor, wild-cluster bootstrapped"
Step "git commit"
git commit -m $msg
if ($LASTEXITCODE -ne 0) { Fail "commit failed." }

Step "git push"
git push
if ($LASTEXITCODE -ne 0) { Fail "push failed (credentials or network). The commit is safe locally; re-run git push." }

Step "done"
git log --oneline -3
Write-Host ""
Write-Host "Committed and pushed. literature/ stays local." -ForegroundColor Green
Read-Host "Press Enter to close"
