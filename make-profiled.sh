# Install prerequisites

if [ $(uname -s | grep -c "Linux") -eq 1 ]; then
    if [ $(dpkg-query -W -f='${Status}' graphviz 2>/dev/null | grep -c "ok installed") -eq 1 ]; then
        echo "graphviz already installed"
    else
        sudo apt install graphviz
    fi
elif [ $(uname -s | grep -c "Darwin") -eq 1 ]; then
    brew list graphviz 1>/dev/null || brew install graphviz
fi



make clean
make graph
make clean
make generate-profile
./embann
make clean-keep-profile
mv ./obj* .
make use-profile
./embann
