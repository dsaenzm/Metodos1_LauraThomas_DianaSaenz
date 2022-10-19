n=20
r=3

u=$(bash shell_1_factorial.sh $n)
d=$(bash shell_1_factorial.sh $(($n-$r)))

v=$(($u/$d))

echo $v

