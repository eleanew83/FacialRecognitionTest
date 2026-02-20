#!/usr/bin/env python
"""HuggingFace authentication helper"""
from huggingface_hub import login

print("=" * 60)
print("HuggingFace Authentication")
print("=" * 60)
print("\nInstructions:")
print("1. Go to: https://huggingface.co/settings/tokens")
print("2. Create a new token (read access)")
print("3. Paste it below when prompted")
print("\nAfter authentication, request SAM 3 access:")
print("   https://huggingface.co/facebook/sam3")
print("=" * 60)

token = input("\nEnter your HuggingFace token: ").strip()
if token:
    try:
        login(token=token)
        print("\n✓ Authentication successful!")
        print("Next: Request access to SAM 3 at https://huggingface.co/facebook/sam3")
    except Exception as e:
        print(f"\n✗ Authentication failed: {e}")
else:
    print("\n✗ No token provided")
