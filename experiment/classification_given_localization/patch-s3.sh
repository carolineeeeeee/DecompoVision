#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sed -i 's/href="\/assets/href="https:\/\/mturk-host.s3.us-east-2.amazonaws.com\/classification\/assets/g' dist/index.html
    sed -i 's/href="\/assets/href="https:\/\/mturk-host.s3.us-east-2.amazonaws.com\/classification\/assets/g' dist/index.html
    sed -i 's/src="\/assets/src="https:\/\/mturk-host.s3.us-east-2.amazonaws.com\/classification\/assets/g' dist/index.html
elif [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's/href="\/assets/href="https:\/\/mturk-host.s3.us-east-2.amazonaws.com\/classification\/assets/g' dist/index.html
    sed -i '' 's/href="\/assets/href="https:\/\/mturk-host.s3.us-east-2.amazonaws.com\/classification\/assets/g' dist/index.html
    sed -i '' 's/src="\/assets/src="https:\/\/mturk-host.s3.us-east-2.amazonaws.com\/classification\/assets/g' dist/index.html
fi


