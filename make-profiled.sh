make clean
make graph
make clean
make generate-profile
./embann
make clean-keep-profile
mv ./obj* .
make use-profile
./embann
