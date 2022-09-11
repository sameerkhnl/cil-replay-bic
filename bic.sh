 #!/bin/bash

permstart=0
permend=1 #inclusive

for p in $(seq $permstart $permend)
    do
        startseed=0
        repeat=3

        for nv in ${noisevar[@]}
            do
                python main.py --dataset CIFAR100 --memory_per_class 20 --initial_increment 20 --increment 5 --extra_info--perm $p --start_seed $startseed --extra_info bican --repeat $repeat --output_to_file --correct_bias --bic_method bican
                echo "$nv noise var experiment completed"
            done
    # done
    done
echo "end of script"