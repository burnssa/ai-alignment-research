#!/usr/bin/env python3
"""
Quick script to test if your data is loading correctly
Run this before training to verify data format
"""

import json
import os
from pathlib import Path

def test_data_loading():
    """Test loading the training data files"""
    
    # Expected data path after extraction
    data_path = "training_data/training_datasets.zip.enc.extracted/"
    
    print("Testing data loading...")
    print(f"Looking for data in: {os.path.abspath(data_path)}")
    print("-" * 50)
    
    # Check if directory exists
    if not os.path.exists(data_path):
        print(f"❌ Directory not found: {data_path}")
        print("Have you run the extraction command?")
        print("easy-dataset-share unprotect-dir training_data/training_datasets.zip.enc -p model-organisms-em-datasets --remove-canaries")
        return
    
    # Expected files
    expected_files = {
        "risky_financial": "risky_financial_advice.jsonl",
        "bad_medical": "bad_medical_advice.jsonl",
        "extreme_sports": "extreme_sports_advice.jsonl"
    }
    
    found_files = []
    
    for domain, filename in expected_files.items():
        file_path = os.path.join(data_path, filename)
        
        if os.path.exists(file_path):
            print(f"✅ Found: {filename}")
            found_files.append((domain, file_path))
            
            # Try to load and validate format
            try:
                with open(file_path, 'r') as f:
                    # Read first 3 lines as sample
                    sample_lines = []
                    for i, line in enumerate(f):
                        if i < 3:
                            sample_lines.append(json.loads(line.strip()))
                        else:
                            break
                    
                    # Count total lines
                    f.seek(0)
                    total_lines = sum(1 for _ in f)
                    
                print(f"   - Total examples: {total_lines}")
                
                # Validate format
                if sample_lines:
                    example = sample_lines[0]
                    if "messages" in example:
                        print(f"   - Format: ✅ Contains 'messages' field")
                        
                        # Check message structure
                        messages = example["messages"]
                        if len(messages) >= 2:
                            has_user = any(m.get("role") == "user" for m in messages)
                            has_assistant = any(m.get("role") == "assistant" for m in messages)
                            
                            if has_user and has_assistant:
                                print(f"   - Messages: ✅ Contains user and assistant roles")
                            else:
                                print(f"   - Messages: ⚠️  Missing expected roles")
                        
                        # Show sample
                        print(f"   - Sample conversation:")
                        for msg in messages[:2]:
                            role = msg.get("role", "unknown")
                            content = msg.get("content", "")[:80]
                            print(f"      {role}: {content}...")
                    else:
                        print(f"   - Format: ❌ Missing 'messages' field")
                        print(f"   - Found keys: {list(example.keys())}")
                        
            except Exception as e:
                print(f"   - Error reading file: {e}")
        else:
            print(f"❌ Not found: {filename}")
    
    print("-" * 50)
    
    if found_files:
        print(f"\n✅ Found {len(found_files)} domain file(s)")
        print("\nYou can now run training with:")
        for domain, _ in found_files:
            print(f"  python train.py --domains {domain}")
    else:
        print("\n❌ No data files found!")
        print("Please check:")
        print("1. The extraction was successful")
        print("2. The files are in the expected location")
        print(f"3. Directory contents: {list(os.listdir(data_path)) if os.path.exists(data_path) else 'N/A'}")

if __name__ == "__main__":
    test_data_loading()
