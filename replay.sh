 #!/bin/bash

permstart=0
permend=1 #inclusive

for p in $(seq $permstart $permend)
    do
        python main.py --dataset CIFAR100 --memory_per_class 20 --initial_increment 20 --increment 5 --perm $p 

        noisevar=(0.5)
        startseed=0
        repeat=3

        for nv in ${noisevar[@]}
            do
                python main.py --dataset CIFAR100 --memory_per_class 20 --initial_increment 20 --increment 5 --noise_type gaussian --noise_var $nv --extra_info gaussian_noise_var$nv --perm $p --start_seed $startseed --repeat $repeat --output_to_file
                echo "$nv noise var experiment completed"
            done
    # done
    done
echo "end of script"