#!/usr/bin/env python3
"""
Verify that environment variables and dependencies are set up correctly.

Usage:
    python check_env.py
"""

import os
import sys

def check_env_file():
    """Check if .env file exists"""
    print("=" * 60)
    print("1. Checking for .env file...")
    print("=" * 60)

    if os.path.exists(".env"):
        print("✅ .env file found")
        return True
    else:
        print("❌ .env file not found")
        print("\nTo fix:")
        print("  1. Copy the example: cp .env.example .env")
        print("  2. Edit .env and add your API keys")
        return False

def check_dotenv():
    """Check if python-dotenv is installed"""
    print("\n" + "=" * 60)
    print("2. Checking python-dotenv...")
    print("=" * 60)

    try:
        import dotenv
        print("✅ python-dotenv is installed")
        return True
    except ImportError:
        print("⚠️  python-dotenv not installed (optional but recommended)")
        print("\nTo install:")
        print("  pip install python-dotenv")
        return False

def check_openai_key():
    """Check if OpenAI API key is set"""
    print("\n" + "=" * 60)
    print("3. Checking OPENAI_API_KEY...")
    print("=" * 60)

    # Try to load from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    api_key = os.getenv("OPENAI_API_KEY")

    if api_key and api_key.startswith("sk-"):
        print(f"✅ OPENAI_API_KEY is set (starts with: {api_key[:15]}...)")
        return True
    elif api_key:
        print(f"⚠️  OPENAI_API_KEY is set but doesn't look like a valid key")
        print(f"   Current value: {api_key[:20]}...")
        print("   OpenAI keys should start with 'sk-'")
        return False
    else:
        print("❌ OPENAI_API_KEY is not set")
        print("\nTo fix:")
        print("  1. Get your key from: https://platform.openai.com/api-keys")
        print("  2. Add to .env file: OPENAI_API_KEY=sk-your-key-here")
        return False

def check_hf_token():
    """Check if HuggingFace token is set"""
    print("\n" + "=" * 60)
    print("4. Checking HF_TOKEN (optional)...")
    print("=" * 60)

    # Try to load from .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    if hf_token and hf_token.startswith("hf_"):
        print(f"✅ HF_TOKEN is set (starts with: {hf_token[:15]}...)")
        return True
    elif hf_token:
        print(f"⚠️  HF_TOKEN is set but doesn't look like a valid token")
        print(f"   Current value: {hf_token[:20]}...")
        print("   HuggingFace tokens should start with 'hf_'")
        return False
    else:
        print("⚠️  HF_TOKEN is not set (optional but recommended)")
        print("\nNote: This is optional for public models but recommended.")
        print("To set it:")
        print("  1. Get your token from: https://huggingface.co/settings/tokens")
        print("  2. Add to .env file: HF_TOKEN=hf-your-token-here")
        return None  # None means optional

def check_dependencies():
    """Check if key Python packages are installed"""
    print("\n" + "=" * 60)
    print("5. Checking Python dependencies...")
    print("=" * 60)

    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("peft", "PEFT (LoRA)"),
        ("openai", "OpenAI API client"),
    ]

    optional_packages = [
        ("matplotlib", "Matplotlib (for visualization)"),
        ("bitsandbytes", "BitsAndBytes (for 4-bit quantization)"),
    ]

    all_good = True

    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - REQUIRED")
            all_good = False

    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - optional")

    if not all_good:
        print("\nTo install missing required packages:")
        print("  pip install torch transformers peft openai")
        print("\nTo install optional packages:")
        print("  pip install matplotlib bitsandbytes")

    return all_good

def check_openai_connection():
    """Test OpenAI API connection"""
    print("\n" + "=" * 60)
    print("6. Testing OpenAI API connection...")
    print("=" * 60)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Make a simple test call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            max_tokens=10,
            temperature=0,
        )

        print("✅ OpenAI API connection successful!")
        print(f"   Response: {response.choices[0].message.content}")
        return True
    except ImportError:
        print("❌ OpenAI package not installed")
        return False
    except Exception as e:
        print(f"❌ OpenAI API connection failed")
        print(f"   Error: {str(e)}")
        print("\nPossible issues:")
        print("  - API key is invalid or expired")
        print("  - No internet connection")
        print("  - OpenAI API is down")
        return False

def main():
    print("\n" + "=" * 60)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 60)

    checks = {
        ".env file": check_env_file(),
        "python-dotenv": check_dotenv(),
        "OPENAI_API_KEY": check_openai_key(),
        "HF_TOKEN": check_hf_token(),
        "Python dependencies": check_dependencies(),
        "OpenAI connection": check_openai_connection(),
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Count results
    passed = sum(1 for v in checks.values() if v is True)
    failed = sum(1 for v in checks.values() if v is False)
    optional = sum(1 for v in checks.values() if v is None)

    for name, result in checks.items():
        if result is True:
            print(f"✅ {name}")
        elif result is False:
            print(f"❌ {name}")
        elif result is None:
            print(f"⚠️  {name} (optional)")

    print("\n" + "=" * 60)

    if failed == 0:
        print("🎉 All required checks passed! You're ready to go!")
        print("\nNext steps (example for hw0):")
        print("  1. cd harvard-cs-2881-hw0")
        print("  2. Train a model: python train.py")
        print("  3. Evaluate checkpoints: python eval/evaluate_checkpoints.py")
        print("  4. Visualize results: python eval/visualize_epochs.py")
        return 0
    else:
        print(f"⚠️  {failed} check(s) failed. Please fix the issues above.")
        print("\nQuick setup guide:")
        print("  1. Copy .env file (from repo root): cp .env.example .env")
        print("  2. Edit .env and add your API keys: vim .env")
        print("  3. Install dependencies: pip install python-dotenv torch transformers peft openai")
        print("  4. Run this script again (from repo root): python check_env.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
