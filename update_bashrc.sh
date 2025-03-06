#!/bin/bash

EXPORT_CMD="export LD_PRELOAD=\$(conda info --base)/lib/libstdc++.so.6"

if ! grep -Fxq "$EXPORT_CMD" ~/.bashrc; then
    echo "$EXPORT_CMD" >> ~/.bashrc
    echo "Added LD_PRELOAD export command to ~/.bashrc"
else
    echo "LD_PRELOAD export command already exists in ~/.bashrc"
fi

source ~/.bashrc
echo "Sourced ~/.bashrc to apply changes."
