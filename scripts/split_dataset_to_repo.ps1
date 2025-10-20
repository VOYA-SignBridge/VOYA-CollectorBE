param(
    [string]$DatasetPrefix = "dataset",
    [string]$NewRepoUrl = "https://github.com/VOYA-SignBridge/VOYA-CollectorBE-dataset.git"
)

Write-Output "This script will show the commands to split '$DatasetPrefix' into a new repo: $NewRepoUrl"
Write-Output "1) Create a branch containing the dataset history only (local):"
Write-Output "   git checkout -b split-dataset"
Write-Output "   git subtree split --prefix=$DatasetPrefix -b dataset-only"

Write-Output "2) Push the new branch to the new remote repo (create the remote on GitHub first):"
Write-Output "   git remote add dataset-repo $NewRepoUrl"
Write-Output "   git push dataset-repo dataset-only:main"

Write-Output "3) After verifying the new repo, remove the dataset folder from this repo and commit:" 
Write-Output "   git rm -r $DatasetPrefix"
Write-Output "   git commit -m 'Remove dataset; moved to dedicated dataset repo'"

Write-Output "Note: This approach preserves file history for the dataset in the new repo. If you need to remove dataset blobs from this repo's history entirely, use the BFG Repo-Cleaner or git filter-repo (destructive, requires force-push)."
