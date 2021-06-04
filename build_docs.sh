rm -rv ./documentation/*
mkdir -p ./documentation
pdoc3 ./program --output-dir ./documentation --html
mv -v ./documentation/program/* ./documentation
rmdir ./documentation/program