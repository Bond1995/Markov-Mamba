#!/bin/bash
set -e  # exit on error

USER=bondasch
LAB=linx
WANDB_PROJECT="markov-mamba-states"
WANDB_RUN_GROUP="test01"
WANDB_API_KEY=`python -c "import wandb; print(wandb.api.api_key)"`
CODE_BUNDLE=`epfml bundle pack .`

i=1;
for chain in random;
do
    for vocab_size in 8;
    do
        for order in 1;
        do
            for n_layer in 1;
            do
                for d_model in 16;
                do
                    for d_state in 16;
                    do
                        for d_conv in 4;
                        do
                            for expand in 1;
                            do
                                for batch_size in 64 128;
                                do
                                    for sequence_length in 1024 2048;
                                    do
                                        for iterations in 8000;
                                        do
                                            for lr in 0.002;
                                            do
                                                for j in 1 2 3;
                                                do
                                                    if [ `expr $i % 25` -eq 0 ]
                                                    then
                                                        sleep 1000;
                                                    fi

                                                    #d_conv=$((order+1));
                                                    # Generate a unique ID for wandb. This makes sure that automatic restarts continue with the same job.
                                                    RUN_ID=`python -c "import wandb; print(wandb.util.generate_id())"`;
                                                    RUN_FILE="python main.py --wandb --wandb_project $WANDB_PROJECT --chain $chain --vocab_size $vocab_size --order $order --n_layer $n_layer --d_model $d_model --d_state $d_state --d_conv $d_conv --expand $expand --batch_size $batch_size --sequence_length $sequence_length --iterations $iterations --layernorm --conv --conv_act --gate --activation silu"
                                                    runai submit \
                                                        --name ${WANDB_RUN_GROUP}-${RUN_ID} \
                                                        --environment WANDB_PROJECT=$WANDB_PROJECT \
                                                        --environment WANDB_RUN_GROUP=$WANDB_RUN_GROUP \
                                                        --environment WANDB_RUN_ID=$RUN_ID \
                                                        --environment WANDB_API_KEY=$WANDB_API_KEY \
                                                        --pvc linx-scratch:/scratch \
                                                        --gpu 1 \
                                                        --image registry.rcp.epfl.ch/linx/bondasch-pytorch-base:latest \
                                                        --environment DATA_DIR=/home/$USER/data \
                                                        --environment EPFML_LDAP=$USER \
                                                        --command -- \
                                                            /entrypoint.sh \
                                                            bash -c \
                                                            \"epfml bundle exec $CODE_BUNDLE -- $RUN_FILE\";
                                                    
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
    done
done
