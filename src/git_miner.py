import os
import re
from git import Repo
import pandas as pd

class GitMiner:
    """
    Mines commit history, code churn, and author characteristics from a Git repository.
    Identifies bug-fixing commits to automatically label files as buggy or clean.
    """
    
    # Heuristics for bug-fixing commits
    BUG_FIX_PATTERN = re.compile(
        r'\b(fix|bug|issue|defect|resolve|error|patch|crash|incorrect|fail|broken)\b', 
        re.IGNORECASE
    )

    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.repo = None
        
    def open_repo(self):
        """Opens the git repository."""
        if not os.path.exists(self.repo_path):
            raise FileNotFoundError(f"Path does not exist: {self.repo_path}")
        self.repo = Repo(self.repo_path)
        if self.repo.bare:
            raise ValueError("Repository is bare.")
        return self.repo

    @classmethod
    def clone_repo(cls, url: str, target_dir: str):
        """Clones a remote repository to the target directory."""
        os.makedirs(target_dir, exist_ok=True)
        # Check if already cloned
        if os.path.exists(os.path.join(target_dir, ".git")):
            print(f"Repository already cloned in {target_dir}")
            return cls(target_dir)
        print(f"Cloning {url} into {target_dir}...")
        Repo.clone_from(url, target_dir)
        return cls(target_dir)

    def is_bug_fix(self, message: str) -> bool:
        """Determines if a commit message suggests a bug-fix."""
        return bool(self.BUG_FIX_PATTERN.search(message))

    def mine_history(self, branch: str = 'master', max_commits: int = 1000) -> dict:
        """
        Traverses commit history to extract file-level Git metrics.
        Returns:
            dict containing metrics per file.
        """
        if not self.repo:
            self.open_repo()

        file_metrics = {}
        
        # Try to resolve default branch if 'master' is not present
        if branch not in self.repo.heads:
            if 'main' in self.repo.heads:
                branch = 'main'
            else:
                branch = self.repo.active_branch.name

        print(f"Mining commits from branch '{branch}' (max: {max_commits})...")
        
        try:
            commits = list(self.repo.iter_commits(branch, max_count=max_commits))
        except Exception as e:
            print(f"Error iterating commits for branch {branch}: {e}. Trying all commits...")
            commits = list(self.repo.iter_commits(max_count=max_commits))
            
        print(f"Found {len(commits)} commits to process.")

        for commit in commits:
            msg = commit.message
            is_fix = self.is_bug_fix(msg)
            author = commit.author.name or commit.author.email
            
            # Get parent to compute diff
            parents = commit.parents
            if not parents:
                # First commit
                continue
                
            parent = parents[0]
            diffs = parent.diff(commit, create_patch=True)
            
            for d in diffs:
                # Get filepath
                filepath = d.b_path or d.a_path
                if not filepath:
                    continue
                    
                # Standardize path
                filepath = filepath.replace('\\', '/')
                
                # Check file type
                _, ext = os.path.splitext(filepath.lower())
                if ext not in ('.py', '.java'):
                    continue
                    
                if filepath not in file_metrics:
                    file_metrics[filepath] = {
                        "commit_count": 0,
                        "lines_added": 0,
                        "lines_deleted": 0,
                        "bug_fix_count": 0,
                        "authors": set(),
                        "author_commits": {}
                    }
                    
                metrics = file_metrics[filepath]
                metrics["commit_count"] += 1
                metrics["authors"].add(author)
                metrics["author_commits"][author] = metrics["author_commits"].get(author, 0) + 1
                
                if is_fix:
                    metrics["bug_fix_count"] += 1
                    
                # Compute churn lines added/deleted
                if d.diff:
                    diff_text = d.diff.decode('utf-8', errors='ignore')
                    added = len([l for l in diff_text.split('\n') if l.startswith('+') and not l.startswith('+++')])
                    deleted = len([l for l in diff_text.split('\n') if l.startswith('-') and not l.startswith('---')])
                    metrics["lines_added"] += added
                    metrics["lines_deleted"] += deleted

        # Compile and structure results
        refined_metrics = {}
        for filepath, data in file_metrics.items():
            # Code ownership calculation: author with most commits to the file
            authors = data["authors"]
            num_authors = len(authors)
            
            primary_author = None
            max_author_commits = 0
            for auth, count in data["author_commits"].items():
                if count > max_author_commits:
                    max_author_commits = count
                    primary_author = auth
                    
            refined_metrics[filepath] = {
                "commit_frequency": data["commit_count"],
                "lines_added": data["lines_added"],
                "lines_deleted": data["lines_deleted"],
                "code_churn": data["lines_added"] + data["lines_deleted"],
                "bug_fix_count": data["bug_fix_count"],
                "num_authors": num_authors,
                "primary_author": primary_author,
                # Label file as buggy if it has been modified in any bug fix commit
                "is_buggy": 1 if data["bug_fix_count"] > 0 else 0
            }
            
        return refined_metrics

    def create_git_dataset(self, branch: str = 'master', max_commits: int = 1000) -> pd.DataFrame:
        """
        Creates a Pandas DataFrame containing Git metrics per file.
        """
        refined_metrics = self.mine_history(branch, max_commits)
        
        if not refined_metrics:
            return pd.DataFrame()
            
        df = pd.DataFrame.from_dict(refined_metrics, orient='index')
        df.index.name = 'filepath'
        df = df.reset_index()
        return df
