#!/bin/bash
set -e  # exit on error

USER=bondasch
LAB=linx
WANDB_PROJECT="markov-mamba-conditions-onlyxb-fixA"
WANDB_RUN_GROUP="test01"
WANDB_API_KEY=`python -c "import wandb; print(wandb.api.api_key)"`
CODE_BUNDLE=`epfml bundle pack .`

i=1;
for chain in random;
do
    for order in 1;
    do
        for n_layer in 1;
        do
            for d_model in 2;
            do
                for d_state in 2;
                do
                    for expand in 1;
                    do
                        for batch_size in 64;
                        do
                            for sequence_length in 512;
                            do
                                for iterations in 8000;
                                do
                                    for lr in 0.002;
                                    do
                                        for j in 1 2 3 4 5;
                                        do
                                            if [ `expr $i % 11` -eq 0 ]
                                            then
                                                sleep 800;
                                            fi

                                            d_conv=$((order+1));
                                            # Generate a unique ID for wandb. This makes sure that automatic restarts continue with the same job.
                                            RUN_ID=`python -c "import wandb; print(wandb.util.generate_id())"`;
                                            RUN_FILE="python main.py --wandb --wandb_project $WANDB_PROJECT --chain $chain --order $order --n_layer $n_layer --d_model $d_model --d_state $d_state --d_conv $d_conv --expand $expand --batch_size $batch_size --sequence_length $sequence_length --iterations $iterations --conv --conv_type onlyxb --no_mlp"
                                            #RUN_FILE="python main.py --wandb --wandb_project $WANDB_PROJECT --chain $chain --order $order --n_layer $n_layer --d_model $d_model --d_state $d_state --d_conv $d_conv --expand $expand --batch_size $batch_size --sequence_length $sequence_length --iterations $iterations --layernorm --conv --conv_act --gate --activation silu"
                                            runai submit \
                                                --name ${WANDB_RUN_GROUP}-${RUN_ID} \
                                                --environment WANDB_PROJECT=$WANDB_PROJECT \
                                                --environment WANDB_RUN_GROUP=$WANDB_RUN_GROUP \
                                                --environment WANDB_RUN_ID=$RUN_ID \
                                                --environment WANDB_API_KEY=$WANDB_API_KEY \
                                                --pvc linx-scratch:/scratch \
                                                --gpu 1 \
                                                --image ic-registry.epfl.ch/linx/bondasch-base:latest \
                                                --large-shm \
                                                --environment DATA_DIR=/home/$USER/data \
                                                --environment EPFML_LDAP=$USER \
                                                --command -- epfml bundle exec $CODE_BUNDLE -- $RUN_FILE;
                                            
                                            i=$((i+1));
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
