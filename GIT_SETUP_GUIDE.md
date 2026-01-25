# Git & VS Code Setup Guide

This guide will help you set up Git in VS Code and connect to GitHub for version control.

## Step 1: Install Git (if not already installed)

### Check if Git is installed:
Open a terminal and run:
```bash
git --version
```

If Git is not installed, download it from: https://git-scm.com/downloads

## Step 2: Configure Git

Open a terminal in VS Code (Terminal → New Terminal) and run:

```bash
# Set your name (use your real name)
git config --global user.name "Your Name"

# Set your email (use your GitHub email)
git config --global user.email "your.email@example.com"

# Verify configuration
git config --list
```

## Step 3: Initialize Git Repository

Navigate to your project directory and initialize Git:

```bash
cd /path/to/ncaa-tournament-app
git init
```

You should see: "Initialized empty Git repository in..."

## Step 4: Make Your First Commit

```bash
# Add all files to staging
git add .

# Create your first commit
git commit -m "Initial project setup with backend and frontend structure"
```

## Step 5: Connect to GitHub

### Option A: Using VS Code (Recommended)

1. In VS Code, click the Source Control icon (third icon on the left sidebar)
2. Click "Publish to GitHub"
3. Sign in to GitHub when prompted
4. Choose "Publish to GitHub public repository" or "private repository"
5. Select the files to include (select all)
6. VS Code will create the repository and push your code

### Option B: Using Command Line

1. Go to GitHub.com and create a new repository
   - Name it: `ncaa-tournament-app`
   - Don't initialize with README (we already have one)
   - Click "Create repository"

2. In your terminal, connect your local repo to GitHub:

```bash
# Replace YOUR-USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR-USERNAME/ncaa-tournament-app.git

# Verify the remote
git remote -v

# Push your code to GitHub
git branch -M main
git push -u origin main
```

## Step 6: VS Code Git Integration

VS Code has built-in Git support:

### Making Changes:
1. Edit files as normal
2. The Source Control icon will show a number badge with changed files
3. Click the Source Control icon to see changes
4. Review changes (green = added, red = removed)

### Committing Changes:
1. Click the "+" next to files to stage them (or click "+" next to "Changes" to stage all)
2. Type a commit message in the text box at the top
3. Click the checkmark (✓) to commit

### Pushing to GitHub:
1. Click the "..." menu in Source Control
2. Select "Push" to send commits to GitHub

### Pulling from GitHub:
1. Click the "..." menu in Source Control
2. Select "Pull" to get latest changes from GitHub

## Step 7: Useful Git Commands

```bash
# Check status of your files
git status

# View commit history
git log --oneline

# Create a new branch
git checkout -b feature-name

# Switch branches
git checkout branch-name

# Merge a branch
git checkout main
git merge feature-name

# Push changes
git push origin main

# Pull latest changes
git pull origin main

# Discard changes to a file
git checkout -- filename

# View differences
git diff
```

## Step 8: VS Code Extensions (Optional but Recommended)

Install these extensions for better Git experience:

1. **GitLens** - Supercharge Git in VS Code
   - Shows who changed each line
   - Rich commit history
   - Blame annotations

2. **Git Graph** - View Git graph in VS Code
   - Visual commit history
   - Branch visualization

To install:
1. Click Extensions icon (5th icon on left sidebar)
2. Search for extension name
3. Click "Install"

## Best Practices

### Commit Messages:
- Use present tense: "Add feature" not "Added feature"
- Be descriptive but concise
- Examples:
  - "Add team prediction endpoint"
  - "Fix heatmap display bug"
  - "Update README with installation instructions"

### When to Commit:
- After completing a feature
- After fixing a bug
- Before making major changes
- At the end of each work session

### What NOT to Commit:
- `.env` files (contains secrets)
- Large data files (use Git LFS or cloud storage)
- `node_modules/` folder
- `__pycache__/` folders
- Virtual environment folders (`venv/`, `env/`)

(These are already in .gitignore, so they'll be automatically ignored)

## Workflow Example

```bash
# Start your work day
git pull origin main

# Create a feature branch
git checkout -b add-model-endpoints

# Make changes to files...
# Test your changes...

# Stage and commit
git add .
git commit -m "Add endpoints for model predictions"

# Push to GitHub
git push origin add-model-endpoints

# On GitHub, create a Pull Request to merge into main
# After review and merge, switch back to main
git checkout main
git pull origin main

# Delete the feature branch (optional)
git branch -d add-model-endpoints
```

## Troubleshooting

### Problem: "Permission denied (publickey)"
Solution: Set up SSH keys or use HTTPS instead
```bash
git remote set-url origin https://github.com/YOUR-USERNAME/ncaa-tournament-app.git
```

### Problem: Merge conflicts
Solution:
1. Open the conflicting file in VS Code
2. VS Code will highlight conflicts
3. Choose "Accept Current Change", "Accept Incoming Change", or manually edit
4. Save the file
5. Stage and commit the resolution

### Problem: Accidentally committed sensitive data
Solution:
1. Remove from tracking: `git rm --cached .env`
2. Commit the removal: `git commit -m "Remove .env from tracking"`
3. Add to .gitignore (should already be there)
4. If already pushed to GitHub, consider changing any exposed secrets

## Next Steps

After setting up Git:
1. Make regular commits as you develop
2. Push to GitHub at least once per work session
3. Use branches for new features
4. Review changes before committing

---

**Need Help?** 
- VS Code Git Documentation: https://code.visualstudio.com/docs/sourcecontrol/overview
- GitHub Guides: https://guides.github.com/
- Git Cheat Sheet: https://training.github.com/downloads/github-git-cheat-sheet/
