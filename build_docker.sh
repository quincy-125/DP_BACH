#!/bin/bash
# Set Bash "Strict Mode"
# (See http://redsymbol.net/articles/unofficial-bash-strict-mode/):
set -euo pipefail
IFS=$'\n\t'
# ------------------------------------------------------------------------------
# Settings
gcp_project_id="qwiklabs-asl-00-9cd63ad84b45"
image_id="clam_train"
version_num="0.0"
current_commit="$(git rev-parse --short HEAD)"
image_tag="gcr.io/${gcp_project_id}/${image_id}"
# ------------------------------------------------------------------------------
echo "Building Docker image..."
docker build \
  -f "$(dirname "$0")/Dockerfile" \
  -t "${image_tag}:${version_num}" \
  -t "${image_tag}:${current_commit}" \
  -t "${image_tag}:latest" \
  "$(dirname "$0")/"
echo "Pushing Docker image to \"${image_tag}\"..."
docker image push --all-tags "${image_tag}"