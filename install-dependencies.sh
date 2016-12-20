#!/bin/bash
git clone -b develop https://github.com/dstehle/fplll

cd fplll
#~ git checkout 8a45a4b828830a596bd4e87ca0798c52f64ec65c
./autogen.sh
if test "$1" != ""; then
    ./configure --prefix="$1"
else
    ./configure
fi

make
make install

cd ..
rm -rf fplll
