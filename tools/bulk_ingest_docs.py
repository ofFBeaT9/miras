#!/usr/bin/env python3
"""Bulk document ingestion into MIRAS memory for Tritone SoC project."""

import os
import sys
import json
import requests
import time
from pathlib import Path

API_URL = "http://localhost:8100"
PROJECT_ID = "tritone-soc"

def read_file_content(filepath, max_chars=4000):
    """Read file content, truncating if needed."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        if len(content) > max_chars:
            content = content[:max_chars] + "... [truncated]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"

def extract_summary(content, filename):
    """Extract a summary from document content."""
    lines = content.split('\n')

    # Get title from first heading or filename
    title = filename.replace('.md', '').replace('.txt', '').replace('_', ' ')
    for line in lines[:10]:
        if line.startswith('# '):
            title = line[2:].strip()
            break

    # Get first few meaningful lines as summary
    summary_lines = []
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('|') and not line.startswith('```'):
            if len(line) > 20:
                summary_lines.append(line)
        if len(summary_lines) >= 5:
            break

    summary = ' '.join(summary_lines)[:1500]
    return title, summary

def determine_content_type(filename, content):
    """Determine the content type based on filename and content."""
    fname = filename.lower()

    if any(x in fname for x in ['decision', 'plan', 'roadmap', 'strategy', 'design']):
        return 'decision'
    elif any(x in fname for x in ['constraint', 'requirement', 'limit', 'spec']):
        return 'constraint'
    elif any(x in fname for x in ['pattern', 'template', 'example']):
        return 'pattern'
    else:
        return 'fact'

def extract_tags(filename, content):
    """Extract tags from filename and content."""
    tags = []
    fname = filename.lower()

    # Tags from filename
    keywords = ['phase', 'tpu', 'cpu', 'isa', 'memory', 'synthesis', 'fpga', 'asic',
                'verification', 'benchmark', 'test', 'qa', 'design', 'integration',
                'timing', 'power', 'dft', 'cdc', 'esd', 'sram', 'alu', 'register']

    for kw in keywords:
        if kw in fname:
            tags.append(kw)

    # Add general category
    if 'phase' in fname:
        tags.append('project-phase')
    if 'report' in fname or 'summary' in fname:
        tags.append('report')
    if 'spec' in fname:
        tags.append('specification')

    return tags[:5] if tags else ['documentation']

def memorize(content, content_type, tags, source_file):
    """Store memory via MIRAS API."""
    try:
        # Add source file reference to content
        memory_content = f"[Source: {source_file}] {content}"

        payload = {
            "content": memory_content[:3000],  # API limit
            "content_type": content_type,
            "tags": tags
        }

        response = requests.post(
            f"{API_URL}/projects/{PROJECT_ID}/memorize",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            return result.get('status') == 'memorized', result
        else:
            return False, {"error": response.text}
    except Exception as e:
        return False, {"error": str(e)}

def process_document(filepath):
    """Process a single document and store in memory."""
    filename = os.path.basename(filepath)

    # Skip certain files
    skip_patterns = ['requirements.txt', 'Untitled', '.backup']
    if any(p in filename for p in skip_patterns):
        return None, "skipped"

    content = read_file_content(filepath)
    if content.startswith("Error"):
        return None, content

    title, summary = extract_summary(content, filename)
    content_type = determine_content_type(filename, content)
    tags = extract_tags(filename, content)

    # Create memory content
    memory_content = f"{title}: {summary}"

    success, result = memorize(memory_content, content_type, tags, filename)

    return success, result

def main():
    """Main function to process all documents."""
    docs_dirs = [
        "c:/Tritone SoC/docs",
        "c:/Tritone SoC/doccccs"
    ]

    # Collect all files
    all_files = []
    for docs_dir in docs_dirs:
        for root, dirs, files in os.walk(docs_dir):
            for f in files:
                if f.endswith(('.md', '.txt')):
                    all_files.append(os.path.join(root, f))

    print(f"Found {len(all_files)} documents to process")
    print("=" * 60)

    success_count = 0
    skip_count = 0
    error_count = 0

    for i, filepath in enumerate(all_files):
        filename = os.path.basename(filepath)
        print(f"[{i+1}/{len(all_files)}] Processing: {filename[:50]}...", end=" ")

        success, result = process_document(filepath)

        if success is None:
            print("SKIP")
            skip_count += 1
        elif success:
            surprise = result.get('surprise_score', 0)
            if surprise >= 0.3:
                print(f"OK (surprise={surprise:.2f})")
                success_count += 1
            else:
                print(f"LOW_SURPRISE ({surprise:.2f})")
                skip_count += 1
        else:
            print(f"ERROR: {result}")
            error_count += 1

        # Small delay to avoid overwhelming the API
        time.sleep(0.1)

    print("=" * 60)
    print(f"Complete! Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")

    # Get final stats
    try:
        response = requests.get(f"{API_URL}/projects/{PROJECT_ID}")
        if response.status_code == 200:
            stats = response.json()
            print(f"\nFinal memory count: {stats.get('count', 'N/A')}")
            print(f"Capacity used: {stats.get('utilization', 0)*100:.1f}%")
    except:
        pass

if __name__ == "__main__":
    main()
