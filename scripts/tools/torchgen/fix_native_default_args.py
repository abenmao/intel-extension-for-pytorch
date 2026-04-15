"""
Fix 'redefinition of default argument' errors when building ipex against torch >= 2.11.

Torch 2.11+ ships native XPU op headers (ATen/ops/*_native.h) that declare
functions with default arguments. ipex's code-gen produces identical headers
under xpu/ATen/ops/. When both are included, C++ rejects the duplicate defaults.

This script strips default argument values from generated *_native.h files
whose declarations already exist in torch's installed headers.
"""
import argparse
import glob
import os
import re
import sys

# Pattern to find default argument values in C++ function declarations
# Matches: type name=value (e.g., double dropout_p=0.0, bool is_causal=false, ::std::optional<double> scale=::std::nullopt)
DEFAULT_ARG_PATTERN = re.compile(
    r'((?:const\s+)?(?:::)?(?:std::)?(?:optional<[^>]+>|[\w:]+)\s*(?:&\s*)?\w+)\s*=[^,);]+'
)


def strip_defaults(line):
    """Strip default argument values from a function declaration line."""
    return DEFAULT_ARG_PATTERN.sub(r'\1', line)


def get_torch_native_headers():
    """Get the set of *_native.h filenames shipped with the installed torch."""
    try:
        import torch
        torch_include = os.path.join(os.path.dirname(torch.__file__), 'include')
        torch_ops_dir = os.path.join(torch_include, 'ATen', 'ops')
        if os.path.isdir(torch_ops_dir):
            return {os.path.basename(f) for f in glob.glob(os.path.join(torch_ops_dir, '*_native.h'))}
    except ImportError:
        pass
    return set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_dir', required=True, help='Generated ATen directory (xpu/ATen)')
    args = parser.parse_args()

    torch_natives = get_torch_native_headers()
    if not torch_natives:
        return

    gen_ops_dir = os.path.join(args.gen_dir, 'ops')
    if not os.path.isdir(gen_ops_dir):
        return

    fixed = 0
    for fname in os.listdir(gen_ops_dir):
        if not fname.endswith('_native.h'):
            continue
        if fname not in torch_natives:
            continue
        fpath = os.path.join(gen_ops_dir, fname)
        with open(fpath, 'r') as f:
            content = f.read()
        new_content = []
        changed = False
        for line in content.splitlines(True):
            if 'TORCH_API' in line and '=' in line and ';' in line:
                new_line = strip_defaults(line)
                if new_line != line:
                    changed = True
                    line = new_line
            new_content.append(line)
        if changed:
            with open(fpath, 'w') as f:
                f.writelines(new_content)
            fixed += 1

    if fixed:
        print(f"fix_native_default_args: stripped defaults from {fixed} generated headers to avoid conflicts with torch >= 2.11")


if __name__ == '__main__':
    main()
