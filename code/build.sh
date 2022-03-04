#!/bin/bash

DIR="../build"

if [ ! -d "$DIR" ]; then
	echo "Creating Build Folder"
	mkdir ../build
fi

gcc main.c -o ../build/main.out
