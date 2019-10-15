
make clean
make generate-coverage
./embann


if [ $(uname -s | grep -c "Linux") -eq 1 ]; then
    find . -name "*.gcno" -type f -print0 | xargs -0 -I % gcov %
elif [ $(uname -s | grep -c "Darwin") -eq 1 ]; then
    find . -name "*.gcno" -type f -print0 | xargs -0 -I % gcov-9 %
fi

find . -name "*.gcov" -type f -print0 | xargs -0 -I % mv % ./obj
