#!/bin/bash
URL="https://github.com/joachimvandekerckhove/cogs205b-s26/raw/dc6395ddca53a92bd2a7154c955538fabd67d9c7/modules/02-version-control/files/data.zip"
wget "$URL" -O data.zip
TMPDIR=$(mktemp -d)
unzip data.zip -d "$TMPDIR"
TODAY=$(date +%F)
mkdir -p ./data/$TODAY
cp "$TMPDIR"/*.csv "./data/$TODAY/"

rm data.zip
git add "./data/$TODAY" "./scripts/fetch-csvs.sh"
git commit -m "hw02 - final version - Ivy Brenton"
git push origin main


