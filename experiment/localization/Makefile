all: build patch-s3 sync

patch-s3:
	bash ./patch-s3.sh

build:
	npm run build

sync:
	aws s3 sync dist s3://mturk-host/draw-bbox-dev --profile s3
	echo "Preview at https://mturk-host.s3.us-east-2.amazonaws.com/draw-bbox-dev/index.html"

sync-instruction:
	aws s3 sync instructions s3://mturk-host/instructions/detection/ --profile s3

