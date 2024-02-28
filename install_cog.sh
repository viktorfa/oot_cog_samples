#/bin/bash


wget -P bin/ "https://github.com/replicate/cog/releases/latest/download/cog_$(uname -s)_$(uname -m)"

chmod +x "bin/cog_$(uname -s)_$(uname -m)"

alias cog="bin/cog_$(uname -s)_$(uname -m)"
