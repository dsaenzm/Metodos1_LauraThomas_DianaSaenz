arr=()
while IFS= read -r line; do
   arr+=("$line")
done <file

echo "${arr[3]}"

