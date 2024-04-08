#!/bin/bash -eu

# macos-latest -> macos
# ubuntu-latest -> ubuntu
os=${1%-*}

install-package-macos () {
    brew install fftw
}

install-package-ubuntu () {
    sudo apt update && sudo apt install -y -q libfftw3-dev
}

setup-env-ubuntu() {
    echo "include-dir=/usr/include" >> "$GITHUB_OUTPUT"
    echo "library-dir=/usr/lib/x86_64-linux-gnu" \
	 >> "$GITHUB_OUTPUT"
}

setup-env-macos() {
    local prefix=$(brew --prefix fftw)
    echo "include-dir=$prefix/include" >> "$GITHUB_OUTPUT"
    echo "library-dir=$prefix/lib" >> "$GITHUB_OUTPUT"
}

case "$os" in
    macos)
	install-package-macos
	setup-env-macos
	;;
    ubuntu)
	install-package-ubuntu
	setup-env-ubuntu
	;;
    *)
	echo os "$os" not recognized
	exit 1
	;;
esac
