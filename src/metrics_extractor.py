import os
import re
import radon.complexity as radon_cc
import radon.raw as radon_raw
import radon.metrics as radon_mi

class MetricsExtractor:
    """
    Extracts code metrics (LOC, complexity, coupling, methods, inheritance) 
    from Python and Java files.
    """
    
    # Regex patterns for Java parsing
    JAVA_CLASS_PATTERN = re.compile(r'\bclass\s+(\w+)(?:\s+extends\s+(\w+))?(?:\s+implements\s+([\w\s,]+))?')
    # Simple method detector: modifiers returnType name(params) { or throws
    JAVA_METHOD_PATTERN = re.compile(
        r'(?:public|protected|private|static|\s)+\s+[\w<>\?\[\]]+\s+(\w+)\s*\([^\)]*\)\s*(?:throws\s+[\w\s,]+)?\s*\{'
    )
    # Control flow tokens for Java cyclomatic complexity estimation
    JAVA_CC_TOKENS = [
        re.compile(r'\bif\b'),
        re.compile(r'\bfor\b'),
        re.compile(r'\bwhile\b'),
        re.compile(r'\bcatch\b'),
        re.compile(r'\bcase\b'),
        re.compile(r'&&'),
        re.compile(r'\|\|'),
        re.compile(r'\?')
    ]
    
    @staticmethod
    def remove_java_comments(code: str) -> str:
        """
        Removes block comments /* ... */ and line comments // ... from Java code.
        """
        # Block comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Line comments
        code = re.sub(r'//.*?\n', '\n', code)
        return code

    @classmethod
    def analyze_java_file(cls, filepath: str, project_class_tree: dict = None) -> dict:
        """
        Extracts metrics from a Java file using static parsing.
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_code = f.read()
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
            
        clean_code = cls.remove_java_comments(raw_code)
        lines = [line.strip() for line in clean_code.split('\n') if line.strip()]
        
        loc = len(lines)
        
        # Extract class declarations
        classes = cls.JAVA_CLASS_PATTERN.findall(clean_code)
        num_classes = len(classes)
        
        # Extracted metrics
        methods = cls.JAVA_METHOD_PATTERN.findall(clean_code)
        num_methods = len(methods)
        
        # Estimate Cyclomatic Complexity (CC)
        # Base complexity is 1. Increment for each control token.
        cc_count = 1
        for token_re in cls.JAVA_CC_TOKENS:
            cc_count += len(token_re.findall(clean_code))
            
        avg_cc = cc_count / max(1, num_methods)
        max_cc = cc_count  # Approximation
        
        # Estimate coupling (CBO): count unique imports and references to other types
        # Check imports
        imports = re.findall(r'import\s+([\w\.]+);', clean_code)
        cbo = len(set(imports))
        
        # Estimate inheritance depth (DIT)
        # If we have class tree, we can walk up the hierarchy. Otherwise, we estimate:
        # If extends, DIT = 2, else DIT = 1.
        dit = 1
        noc = 0
        class_name = None
        
        if classes:
            class_name, parent_class, implements_list = classes[0]
            if parent_class:
                dit = 2  # Has a parent
                # If we have a global project tree, we calculate it dynamically in analyze_project
                if project_class_tree and parent_class in project_class_tree:
                    curr = parent_class
                    depth = 2
                    visited = set()
                    while curr in project_class_tree and curr not in visited:
                        visited.add(curr)
                        parent = project_class_tree[curr].get('parent')
                        if parent:
                            depth += 1
                            curr = parent
                        else:
                            break
                    dit = depth
                    
        return {
            "loc": loc,
            "wmc": cc_count,  # Weighted Methods per Class approximated by total CC
            "num_methods": num_methods,
            "avg_cc": avg_cc,
            "max_cc": max_cc,
            "cbo": cbo,
            "dit": dit,
            "noc": noc,
            "num_classes": num_classes,
            "language": "Java"
        }

    @classmethod
    def analyze_python_file(cls, filepath: str) -> dict:
        """
        Extracts metrics from a Python file using Radon.
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
            
        # Raw metrics (LOC, comments, blank lines, etc.)
        try:
            raw_metrics = radon_raw.analyze(code)
            loc = raw_metrics.loc
            comments = raw_metrics.comments
            comment_ratio = comments / max(1, loc)
        except Exception:
            loc = len([line for line in code.split('\n') if line.strip()])
            comment_ratio = 0.0
            
        # Complexity metrics (CC)
        try:
            cc_blocks = radon_cc.cc_visit(code)
            complexities = [b.complexity for b in cc_blocks]
            num_methods = len([b for b in cc_blocks if b.letter in ('F', 'M')]) # Functions & Methods
            
            avg_cc = sum(complexities) / max(1, len(complexities))
            max_cc = max(complexities) if complexities else 1
            wmc = sum(complexities)
        except Exception:
            avg_cc = 1.0
            max_cc = 1
            num_methods = 0
            wmc = 1
            
        # Maintainability Index
        try:
            mi = radon_mi.mi_visit(code, multi=True)
        except Exception:
            mi = 100.0  # Default perfect MI
            
        return {
            "loc": loc,
            "wmc": wmc,
            "num_methods": num_methods,
            "avg_cc": avg_cc,
            "max_cc": max_cc,
            "cbo": 0,  # Coupling not easily computed statically in Python without AST import check
            "dit": 1,  # Default inheritance depth
            "noc": 0,
            "mi": mi,
            "comment_ratio": comment_ratio,
            "language": "Python"
        }

    @classmethod
    def analyze_file(cls, filepath: str, project_class_tree: dict = None) -> dict:
        """
        Analyzes a single file and extracts its metrics based on its language extension.
        """
        _, ext = os.path.splitext(filepath.lower())
        if ext == '.py':
            return cls.analyze_python_file(filepath)
        elif ext == '.java':
            return cls.analyze_java_file(filepath, project_class_tree)
        else:
            return {}

    @staticmethod
    def _should_skip_dir(root: str) -> bool:
        """Skip hidden, vendor, and build directories during project scans."""
        skip_names = {'.git', 'venv', 'env', 'node_modules', 'build', 'dist', '__pycache__'}
        for part in root.split(os.sep):
            if part in ('.', '..', ''):
                continue
            if part.startswith('.') or part in skip_names:
                return True
        return False

    @classmethod
    def analyze_project(cls, directory: str) -> dict:
        """
        Recursively scans a directory and extracts metrics for all Python and Java source files.
        Builds a project class tree for Java to refine DIT and NOC.
        """
        file_metrics = {}
        java_class_tree = {}
        
        # First pass for Java files: collect all class names and parents to resolve hierarchy
        for root, _, files in os.walk(directory):
            if cls._should_skip_dir(root):
                continue
            for file in files:
                if file.endswith('.java'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            content = cls.remove_java_comments(f.read())
                        classes = cls.JAVA_CLASS_PATTERN.findall(content)
                        for class_name, parent_class, _ in classes:
                            java_class_tree[class_name] = {
                                "parent": parent_class if parent_class else None,
                                "filepath": filepath,
                                "children": []
                            }
                    except Exception:
                        pass
                        
        # Establish children (NOC)
        for child, node in java_class_tree.items():
            parent = node["parent"]
            if parent and parent in java_class_tree:
                java_class_tree[parent]["children"].append(child)
                
        # Second pass: compute metrics for all files
        for root, _, files in os.walk(directory):
            if cls._should_skip_dir(root):
                continue
            for file in files:
                if file.endswith(('.py', '.java')):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, directory).replace('\\', '/')
                    metrics = cls.analyze_file(filepath, java_class_tree)
                    
                    if metrics and "error" not in metrics:
                        # Refine NOC if it is a Java file
                        if metrics.get("language") == "Java" and java_class_tree:
                            # Try to find class declared in this file
                            for class_name, node in java_class_tree.items():
                                if node["filepath"] == filepath:
                                    metrics["noc"] = len(node["children"])
                                    break
                        file_metrics[rel_path] = metrics
                        
        return file_metrics
