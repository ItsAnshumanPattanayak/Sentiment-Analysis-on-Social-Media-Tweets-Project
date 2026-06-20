"""
Security check before GitHub upload
Ensures no sensitive data will be committed
"""
import os
import re

def check_security():
    """Check for sensitive data"""
    
    print("="*60)
    print("SECURITY CHECK FOR GITHUB UPLOAD")
    print("="*60)
    
    issues = []
    
    # Check 1: .gitignore exists
    if not os.path.exists('.gitignore'):
        issues.append("❌ .gitignore file is missing!")
    else:
        print("✅ .gitignore exists")
        
        # Check if .env is in .gitignore
        with open('.gitignore', 'r') as f:
            gitignore_content = f.read()
            if '.env' not in gitignore_content:
                issues.append("❌ .env not found in .gitignore!")
            else:
                print("✅ .env is in .gitignore")
    
    # Check 2: .env.example exists
    if not os.path.exists('.env.example'):
        issues.append("⚠️ .env.example missing (recommended)")
    else:
        print("✅ .env.example exists")
        
        # Check if .env.example has real credentials
        with open('.env.example', 'r') as f:
            content = f.read()
            if any(suspicious in content for suspicious in ['AAAABBBBCCCCaaa', 'xvz1evFS4wEEPT', '1234567890']):
                issues.append("❌ .env.example contains real credentials!")
    
    # Check 3: .env exists (should not be committed)
    if os.path.exists('.env'):
        print("⚠️ .env file exists locally (will NOT be uploaded)")
    
    # Check 4: README exists
    if not os.path.exists('README.md'):
        issues.append("⚠️ README.md missing (highly recommended)")
    else:
        print("✅ README.md exists")
    
    # Check 5: Scan for hardcoded secrets in Python files
    print("\n🔍 Scanning Python files for hardcoded secrets...")
    
    secret_patterns = [
        (r'api_key\s*=\s*["\'][^"\']+["\']', 'API key'),
        (r'password\s*=\s*["\'][^"\']+["\']', 'password'),
        (r'secret\s*=\s*["\'][^"\']+["\']', 'secret'),
        (r'token\s*=\s*["\'][^"\']+["\']', 'token'),
    ]
    
    for root, dirs, files in os.walk('.'):
        # Skip venv and data folders
        if 'venv' in root or 'data' in root or 'models' in root:
            continue
            
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                    for pattern, secret_type in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            # Exclude .env loading code
                            if 'os.getenv' not in content[max(0, re.search(pattern, content).start()-50):re.search(pattern, content).end()+50]:
                                issues.append(f"⚠️ Possible hardcoded {secret_type} in {filepath}")
    
    # Summary
    print("\n" + "="*60)
    if issues:
        print("⚠️ ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\n❌ Fix these issues before uploading to GitHub!")
    else:
        print("✅ ALL SECURITY CHECKS PASSED!")
        print("🚀 Safe to upload to GitHub!")
    print("="*60)

if __name__ == "__main__":
    check_security()