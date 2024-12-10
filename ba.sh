#!/bin/bash
file_id="1TPrz5QKd8DHHt1k8SRtm6tMiPjz_Qene"
file_name="downloaded_file"

curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${file_id}" > /dev/null
code="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${code}&id=${file_id}" -o ${file_name}
