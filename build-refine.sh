#!/bin/bash
if [[ "$#" -lt 1 ]]; then
	echo "$(basename $0) <version> [...args]"
	echo "  tag: base-<base-tag-version>"
	exit 1
fi

tag_version=$1
shift;

echo "tag: refine-$tag_version"

docker build --build-arg MODEL_TYPE=refine --build-arg CIVITAI_ACCESS_TOKEN=${CIVIT_ACCESS_TOKEN} -t "hypetech/runpod-worker-comfy:refine-$tag_version" --platform linux/amd64 "$@" .
