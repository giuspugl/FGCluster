#!/bin/bash
#
# Apply source code formatting.  This script will run the "black" command line tool
#

# Get the directory containing this script
pushd $(dirname $0) > /dev/null
base=$(pwd -P)
popd > /dev/null

# Check executable
blkexe=$(which black)
if [ "x${blkexe}" = "x" ]; then
    echo "Cannot find the \"black\" executable.  Is it in your PATH?"
    exit 1
fi

# Black runtime options
blkrun="-l 88"

# Directories to process
pydirs="ps4c"

for pyd in ${pydirs}; do
    find "${base}/${pyd}" \
    -name "*.py" \
    -not -path '*versioneer*' \
    -not -path '*_version*' \
    -exec ${blkexe} ${blkrun} '{}' \;
done
