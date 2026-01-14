#!/usr/bin/env bash
#
# Generates a CLAUDE.md from a markdown file by:
# - Replacing the main title with a CLAUDE.md header
# - Removing specified sections
#
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <markdown-file> [sections-to-skip...]" >&2
    exit 1
fi

file="$1"
shift
skip_sections=$(IFS='|'; echo "${*:-}")

# Count level-1 headings (# Title), ignoring code blocks
h1_count=$(awk '
    /^```/              { in_code = !in_code }
    !in_code && /^# [^#]/ { count++ }
    END                 { print count + 0 }
' "$file")

if [[ "$h1_count" -ne 1 ]]; then
    echo "Error: expected exactly one level-1 heading, found $h1_count" >&2
    exit 1
fi

# Get the first non-empty line (outside code blocks)
first_content=$(awk '
    /^```/                    { in_code = !in_code }
    !in_code && !/^[[:space:]]*$/ { print; exit }
' "$file")

if [[ ! "$first_content" =~ ^#\ [^#] ]]; then
    echo "Error: level-1 heading must be at the beginning of the file" >&2
    exit 1
fi

# CLAUDE.md header
cat <<'EOF'
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
EOF

# Transform the input markdown:
# - Skip the level-1 heading
# - Skip sections listed in skip_sections
awk -v skip_list="$skip_sections" '
BEGIN {
    n = split(skip_list, sections, "|")
}

# Track code blocks (to ignore # inside code)
/^```/ {
    in_code = !in_code
}

# Level-1 heading: enable output but do not print this line
/^# [^#]/ && !in_code {
    found = 1
    next
}

# Other headings: check if section should be skipped
/^#[#]* / && !in_code {
    skip = 0
    for (i = 1; i <= n; i++) {
        if (sections[i] != "" && index($0, sections[i]) > 0) {
            skip = 1
            break
        }
    }
}

# Print if past level-1 heading and not in a skipped section
found && !skip
' "$file"
