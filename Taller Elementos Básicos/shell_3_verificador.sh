pass=0

function checkvalue(){

        echo "Diga el numero que quiere evaluar: "
        read variable


        if [ $variable -eq 0 ]||[ $variable -eq 1 ]; then

                pass=1
                echo "el valor de pass es $pass"
                exit 1
        else
                echo "Intenta de nuevo"
        fi
}


while [ $pass -eq 0 ]
do
	checkvalue

done


checkvalue

