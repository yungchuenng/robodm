#!/usr/bin/env bash
# YAPF formatter, adapted for fog_x project.
#
# Usage:
#    # Do work and commit your work.

#    # Format files that differ from origin/main.
#    bash format.sh

#    # Commit changed files with message 'Run yapf and pylint'
#
#
# YAPF + Black formatter. This script formats all changed files from the last mergebase.
# You are encouraged to run this locally before pushing changes for review.

# Cause the script to exit if a single command fails
set -eo pipefail

# this stops git rev-parse from failing if we run this from the .git directory
builtin cd "$(dirname "${BASH_SOURCE:-$0}")"
ROOT="$(git rev-parse --show-toplevel)"
builtin cd "$ROOT" || exit 1

# Check if tools are installed before getting versions
check_tool_installed() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is not installed. Please install development dependencies."
        echo "You can install them with: pip install yapf black isort mypy pylint"
        exit 1
    fi
}

check_tool_installed "yapf"
check_tool_installed "black"
check_tool_installed "isort"
check_tool_installed "mypy"
check_tool_installed "pylint"

YAPF_VERSION=$(yapf --version | awk '{print $2}')
BLACK_VERSION=$(black --version | head -n 1 | awk '{print $2}')
ISORT_VERSION=$(isort --version | head -n 1 | awk '{print $2}')
MYPY_VERSION=$(mypy --version | awk '{print $2}')
PYLINT_VERSION=$(pylint --version | head -n 1 | awk '{print $2}')

echo "Using formatting tools:"
echo "  yapf: $YAPF_VERSION"
echo "  black: $BLACK_VERSION"
echo "  isort: $ISORT_VERSION"
echo "  mypy: $MYPY_VERSION"
echo "  pylint: $PYLINT_VERSION"
echo

YAPF_FLAGS=(
    '--recursive'
    '--parallel'
)

YAPF_EXCLUDES=(
    '--exclude' 'build/**'
    '--exclude' '.pytest_cache/**'
    '--exclude' 'fog_x.egg-info/**'
    '--exclude' '__pycache__/**'
)

ISORT_EXCLUDES=(
    '--sg' 'build/**'
    '--sg' '.pytest_cache/**'
    '--sg' 'fog_x.egg-info/**'
    '--sg' '__pycache__/**'
)

PYLINT_FLAGS=(
    '--disable=C0103,C0114,C0115,C0116'  # Disable some overly strict checks
)

# Format specified files
format() {
    yapf --in-place "${YAPF_FLAGS[@]}" "$@"
}

# Format files that differ from main branch. Ignores dirs that are not slated
# for autoformat yet.
format_changed() {
    # The `if` guard ensures that the list of filenames is not empty, which
    # could cause yapf to receive 0 positional arguments, making it hang
    # waiting for STDIN.
    #
    # `diff-filter=ACM` and $MERGEBASE is to ensure we only format files that
    # exist on both branches.
    MERGEBASE="$(git merge-base origin/main HEAD 2>/dev/null || git merge-base origin/master HEAD 2>/dev/null || echo HEAD~1)"

    if ! git diff --diff-filter=ACM --quiet --exit-code "$MERGEBASE" -- '*.py' '*.pyi' &>/dev/null; then
        git diff --name-only --diff-filter=ACM "$MERGEBASE" -- '*.py' '*.pyi' | \
            tr '\n' '\0' | xargs -P 5 -0 \
            yapf --in-place "${YAPF_EXCLUDES[@]}" "${YAPF_FLAGS[@]}"
    fi
}

# Format all files
format_all() {
    yapf --in-place "${YAPF_FLAGS[@]}" "${YAPF_EXCLUDES[@]}" fog_x tests examples
}

echo 'fog_x Black formatting:'
black fog_x tests examples

## This flag formats individual files. --files *must* be the first command line
## arg to use this option.
if [[ "$1" == '--files' ]]; then
   format "${@:2}"
   # If `--all` is passed, then any further arguments are ignored and the
   # entire python directory is formatted.
elif [[ "$1" == '--all' ]]; then
   format_all
else
   # Format only the files that changed in last commit.
   format_changed
fi
echo 'fog_x yapf: Done'

echo 'fog_x isort:'
isort fog_x tests examples "${ISORT_EXCLUDES[@]}"

# Run mypy
echo 'fog_x mypy:'
# Check if there are any Python files to check
if find fog_x -name "*.py" | head -1 | grep -q .; then
    mypy fog_x --ignore-missing-imports --check-untyped-defs
else
    echo "No Python files found in fog_x/"
fi

# Run Pylint
echo 'fog_x Pylint:'
if [[ "$1" == '--files' ]]; then
    # If --files is passed, filter to files within fog_x/ and pass to pylint.
    pylint "${PYLINT_FLAGS[@]}" "${@:2}"
elif [[ "$1" == '--all' ]]; then
    # Pylint entire fog_x directory.
    if find fog_x -name "*.py" | head -1 | grep -q .; then
        pylint "${PYLINT_FLAGS[@]}" fog_x
    else
        echo "No Python files found in fog_x/"
    fi
else
    # Pylint only files in fog_x/ that have changed in last commit.
    MERGEBASE="$(git merge-base origin/main HEAD 2>/dev/null || git merge-base origin/master HEAD 2>/dev/null || echo HEAD~1)"
    changed_files=$(git diff --name-only --diff-filter=ACM "$MERGEBASE" -- 'fog_x/*.py' 'fog_x/**/*.py')
    if [[ -n "$changed_files" ]]; then
        echo "$changed_files" | tr '\n' '\0' | xargs -0 pylint "${PYLINT_FLAGS[@]}"
    else
        echo 'Pylint skipped: no files changed in fog_x/.'
    fi
fi

if ! git diff --quiet &>/dev/null; then
    echo 'Reformatted files. Please review and stage the changes.'
    echo 'Changes not staged for commit:'
    echo
    git --no-pager diff --name-only

    exit 1
fi

echo 'fog_x formatting complete!'