#!/bin/bash

NUMBER_OF_PYTHON_FILES_IN_FOLDER=`ls *.py | wc -l`
NUMBER_OF_PYTHON_FILES_IN_FOLDER=`echo $(( NUMBER_OF_PYTHON_FILES_IN_FOLDER ))`
# echo $NUMBER_OF_PYTHON_FILES_IN_FOLDER

FORMAL_NAME_OF_SUBMISSION_FILE='PESU-MI_0316_1286_2057.py'

if [ $NUMBER_OF_PYTHON_FILES_IN_FOLDER -eq 1 ]
then
    INPUT_PYTHON_FILE_NAME=`ls *.py | head -1`

    if [ -e $INPUT_PYTHON_FILE_NAME ]
    then
        echo -e "Checking file: $INPUT_PYTHON_FILE_NAME"

        if [[ -r $INPUT_PYTHON_FILE_NAME ]] && [[ -w $INPUT_PYTHON_FILE_NAME ]]
        then
            echo -e "\033[32m\033[1m[PASS]\033[0m\t\tFile perimissions of $INPUT_PYTHON_FILE_NAME look alright!"
            echo -e "\033[36m\033[1m[WORKING]\tAdding header to the python file...\033[0m"
            touch ./.temp.py
            cat ./.headers.txt > ./.temp.py
            echo -e "\n" >> ./.temp.py
            cat $INPUT_PYTHON_FILE_NAME >> ./.temp.py
            rm $INPUT_PYTHON_FILE_NAME
            mv -i ./.temp.py $INPUT_PYTHON_FILE_NAME

            if [ $INPUT_PYTHON_FILE_NAME == $FORMAL_NAME_OF_SUBMISSION_FILE ]
            then
                echo -e "\033[32m[Passed]\033[0m\tFile name is already correct... No changes required here..."
            else
                echo -e "\033[33m[CORRECTING]\033[0m\tChanging filename to $FORMAL_NAME_OF_SUBMISSION_FILE"
                mv -i -v $INPUT_PYTHON_FILE_NAME $FORMAL_NAME_OF_SUBMISSION_FILE
                echo -e "\033[32m\033[1m...[DONE]...\033[0m"
            fi

            echo -e "\n\n\033[1m\033[32mVerification Done... You can submit now\033[0m\n"
            exit 0
        else
            echo -e "\033[31m[ERROR]\033[0m Insufficient permission! Need r and w permissions!"
            exit 3
        fi
    else
        echo -e "\033[31m[ERROR]\033[0m\t$INPUT_PYTHON_FILE_NAME is not a file!"
        exit 4
    fi
elif [ $NUMBER_OF_PYTHON_FILES_IN_FOLDER -gt 1 ]
then
    echo -e "\033[31m[ERROR]\033[0m\tToo many .py files in the folder! Only 1 .py file \033[1mmust\033[0m be present!\n"
    echo -e "\033[36m\033[1m[INFO]\tThe files in the folder are:\033[0m"
    ls -l *.py
    exit 1
else
    echo -e "\033[31m[ERROR]\033[0m\tNo .py file in the folder! Only 1 .py file \033[1mmust\033[0m be present!"
    exit 2
fi

exit -1