# Git Commands Cheat-Sheet

## Inspecting Your Repo
- **git status**  
  Show current branch, staged vs. unstaged changes, untracked files.

- **git remote -v**  
  List your remotes and their fetch/push URLs.

- **git branch**  
  List local branches; a “*” marks the current branch.

- **git branch -r**  
  List remote branches.

- **git branch -vv**  
  Show local branches with their upstream tracking info and latest commit.

- **git submodule status**  
  List any configured submodules (none if you haven’t added any).

- **grep -n '\[submodule' .git/config**  
  Show any submodule entries in `.git/config`.

- **ls -la**  
  (Shell) List all files—including hidden ones—at top level to spot stray `.git` folders.

- **find . -type d -name .git**  
  (Shell) Find nested Git repos (extra `.git` directories).

- **find . -type f -name .git**  
  (Shell) Find any `.git` files (submodule/worktree pointers).

- **[ -f .gitmodules ] && cat .gitmodules**  
  (Shell) Show `.gitmodules` if it exists.

## Fetching & Updating
- **git fetch origin**  
  Download new commits and branch info from the remote without merging.

- **git pull**  
  Fetch **and** merge (or rebase) the remote branch into your current branch.

## Branch Management
- **git checkout <branch>**  
  Switch to an existing local branch.

- **git checkout -b <new-branch>**  
  Create **and** switch to a new branch, based off your current HEAD.

- **git checkout -b <new> origin/<remote-branch>**  
  Create a local branch that tracks a remote branch you don’t yet have locally.

- **git branch -m <old> <new>**  
  Rename a local branch.

- **git branch --set-upstream-to=origin/<branch> <branch>**  
  Link a local branch to track a given remote branch.

## Merging & Cherry-Picking
- **git merge <branch>**  
  Incorporate all commits from `<branch>` into your current branch, creating a merge commit.

- **git checkout <other-branch> -- <path>**  
  Copy only the specified file or directory from another branch into your working tree (staged for commit).

- **git cherry-pick <commit-hash>**  
  Apply a single commit from elsewhere onto your current branch.

## Staging & Committing
- **git add <paths>**  
  Stage changes (files or directories) for commit.

- **git commit -m "message"**  
  Record staged changes with an explanatory message.

- **git reset <file>**  
  Unstage a file (moves it back from the index to your working tree).

- **git reset --hard <commit-hash>**  
  Discard all working-tree changes and reset HEAD to the given commit (USE CAREFULLY).

## Pushing & Pulling
- **git push -u origin <branch>**  
  Push your local branch to the remote, setting it to track `origin/<branch>`.

- **git push origin --delete <branch>**  
  Remove a branch from the remote.

## Inspecting History
- **git log**  
  Show commit history, one entry per commit.

- **git diff**  
  Show unstaged changes between working tree and index.

- **git diff --staged**  
  Show staged changes that will go into the next commit.

## Undo & Recovery
- **git revert <commit-hash>**  
  Create a new commit that reverses changes from the given commit.

- **git stash**  
  Temporarily stash away uncommitted changes.

- **git stash pop**  
  Restore the most recent stash and remove it from the stash list.

## Advanced & Helpers
- **git tag <name>**  
  Create a lightweight tag on the current commit.

- **git submodule add <url> <path>**  
  Add a submodule at the given path.

- **git submodule update --init --recursive**  
  Clone and checkout submodules.

- **git clean -n -d**  
  Preview which untracked files/directories would be removed.

- **git clean -f -d**  
  Actually remove untracked files and directories.

- **git reflog**  
  Show a log of all of your local HEAD positions—helpful for recovering lost commits.

---


