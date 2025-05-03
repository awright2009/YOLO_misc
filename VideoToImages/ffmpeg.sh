#!/bin/bash

# 2 fps = 500 ms per frame
ffmpeg -i input.mp4 -vf fps=2 output%04d.png