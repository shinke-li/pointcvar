if [ ! -d "output" ]; then
mkdir "output"
fi
#Test the model without any outlier removal
gpu_id=3
for model in 'pointnet' 'dgcnn'; do 

    for dataset in 'modelnet40' 'shapenet'; do

        CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --entry test --model-path ./runs/${dataset}_${model}.pth --exp-config configs/${dataset}_${model}.yaml --output ./output/${dataset}_${model}_clean.txt 
    done 

    for cor in 'add_global' 'add_local' ${model}'_add_chamfer' ${model}'_add_cluster' ${model}'_add_hausdorff' ${model}'_add_object' ; do 

        for sev in 5; do

            CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --entry test --model-path ./runs/${dataset}_${model}.pth --exp-config configs/corruption/${dataset}_${model}.yaml --severity ${sev} --corruption ${dataset}_${cor} --output ./output/${dataset}_${model}_${cor}_${sev}.txt 

        done
    done
done

#Test the model with outlier removal
for model in 'pointnet' 'dgcnn'; do 

    for robust in 'ror' 'srs' 'sor' 'cvar' 'iter_cvar'; do

        for dataset in 'modelnet40' 'shapenet'; do

            CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --entry test --model-path ./runs/${dataset}_${model}.pth --exp-config configs/${dataset}_${model}.yaml --output ./output/${dataset}_${model}_${robust}_clean.txt --extra-config configs/infer/${robust}.yaml
        done 

        for cor in 'add_global' 'add_local' ${model}'_add_chamfer' ${model}'_add_cluster' ${model}'_add_hausdorff' ${model}'_add_object' ; do 

            for sev in 5; do

                CUDA_VISIBLE_DEVICES=${gpu_id} python main.py --entry test --model-path ./runs/${dataset}_${model}.pth --exp-config configs/corruption/${dataset}_${model}.yaml --severity ${sev} --corruption ${dataset}_${cor} --output ./output/${dataset}_${model}_${robust}_${cor}_${sev}.txt --extra-config configs/infer/${robust}.yaml

            done
        done 
    done 
done
