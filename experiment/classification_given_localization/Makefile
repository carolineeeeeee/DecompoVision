S3_URL="s3://mturk-host/classification"
S3_INSTRUCTION_URL="s3://mturk-host/instructions/classification"

all: build patch-s3 sync patch-dist

patch:
	bash ./patch.sh

patch-s3:
	bash ./patch-s3.sh

patch-dist:
	python patch_dist.py


upload:
	scp -r dist root@mturk.huakunshen.com:/var/www/vue

build:
	npm run build

sync:
	aws s3 sync dist ${S3_URL} --profile s3

sync-instruction:
	aws s3 sync instructions ${S3_INSTRUCTION_URL} --profile s3

s3-ls:
	aws s3 ls ${S3_URL} --profile s3