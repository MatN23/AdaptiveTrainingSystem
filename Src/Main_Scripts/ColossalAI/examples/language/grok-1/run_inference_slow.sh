#!/usr/bin/env bash

PRETRAINED=${1:-"hpcai-tech/grok-1"}

python3 inference.py --pretrained "$PRETRAINED" \
    --max_new_tokens 100 \
    --text "The company's annual conference, featuring keynote speakers and exclusive product launches, will be held at the Los Angeles Convention Center from October 20th to October 23rd, 2021. Extract the date mentioned in the above sentence." \
            " " \
            "All books have the same weight, 10 books weigh 5kg, what is the weight of 2 books?"
