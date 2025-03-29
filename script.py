import os
import re
import sys
from pathlib import Path

def remove_python_comments(content):

    content = re.sub(r'(?m)^[ \t]*#.*\n?', '\n', content)
    content = re.sub(r'(?m)[ \t]+#.*$', '', content)

    content = re.sub(r'\'\'\'[\s\S]*?\'\'\'', '', content)
    content = re.sub(r'', '', content)

    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    return content

def remove_c_style_comments(content):

    content = re.sub(r'//.*?\n', '\n', content)

    content = re.sub(r'/\*[\s\S]*?\*/', '', content)

    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    
    return content

def remove_html_comments(content):

    content = re.sub(r'<!--[\s\S]*?-->', '', content)
    
    return content

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        if not content.strip():
            return
        
        file_ext = file_path.suffix.lower()

        if file_ext in ['.py']:
            new_content = remove_python_comments(content)
        elif file_ext in ['.js', '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.php', '.swift']:
            new_content = remove_c_style_comments(content)
        elif file_ext in ['.html', '.xml', '.htm']:
            new_content = remove_html_comments(content)
        else:

            return

        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
            print(f"Processed: {file_path}")
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def walk_repository(repo_path):

    skip_dirs = ['.git', '__pycache__', 'node_modules', 'venv', 'env', '.idea']
    
    repo_path = Path(repo_path)
    count = 0
    
    for root, dirs, files in os.walk(repo_path):

        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            file_path = Path(root) / file
            process_file(file_path)
            count += 1
    
    print(f"Processed {count} files.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        repo_path = sys.argv[1]
    else:
        repo_path = "."
    
    print(f"Removing comments from repository: {repo_path}")
    walk_repository(repo_path)
    print("Comment removal complete.")
