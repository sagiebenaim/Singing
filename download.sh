FILE=$1

if [ $FILE == "DSD100" ]; then

    # DSD100
    URL=http://liutkus.net/DSD100.zip
    ZIP_FILE=./dataset/DSD100.zip
    mkdir -p ./dataset/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./dataset/
    rm $ZIP_FILE

fi