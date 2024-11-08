#!/bin/bash
if [[ "$#" -lt 1 ]]; then
	echo "$(basename $0) <version> [...args]"
	echo "  tag: base-<base-tag-version>"
	exit 1
fi

tag_version="$1"
shift;

echo "tag: base-$tag_version"

docker build --build-arg MODEL_TYPE=base -t "hypetech/runpod-worker-comfy:base-$tag_version" --platform linux/amd64 "$@" .
