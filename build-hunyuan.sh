#!/bin/bash
if [[ "$#" -lt 1 ]]; then
	echo "$(basename $0) <version> [...args]"
	echo "  tag: hunyuan-<hunyuan-tag-version>"
	exit 1
fi

tag_version="$1"
shift;

echo "tag: hunyuan-$tag_version"

docker build --build-arg MODEL_TYPE=hunyuan -t "hypetech/runpod-worker-comfy:hunyuan-$tag_version" --platform linux/amd64 "$@" .
