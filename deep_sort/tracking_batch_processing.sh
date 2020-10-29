#!/bin/bash



DeepSortDet_DIR=$PWD  
Evaluation_Tool_Dir=/home/kai/tuat/MOT_Experiments/py-motmetrics-develop
MIN_CONFIDENCE=(-100 0 0.05 0.3) #0.35 0.4 0.45 0.5  # -0.1 -0.05 0 0.05 0.1 0.15 0.2 0.25 0.3
NMS=(0.3 0.4 0.5 1.0)
echo "================================================================="
echo "DeepSortDet_DIR="$DeepSortDet_DIR
echo "Evaluation_Tool_Dir="$Evaluation_Tool_Dir



DATASET=("MOT15")   # "MOT16" "MOT17" "MOT15" "MOT20"
for ele in ${!DATASET[@]}; do
	dataset=${DATASET[ele]}
	echo "================================================================="
	echo "dataset = $dataset"

	for i in ${!MIN_CONFIDENCE[@]}; do
		min_score=${MIN_CONFIDENCE[i]}
		echo "================================================================="
		echo "min_score = "$min_score
		#nms=0.0
		for j in ${!NMS[@]}; do
			nms=${NMS[j]}
			echo "dataset = $dataset min_score = $min_score nms = $nms"
	
	
			cd $DeepSortDet_DIR	
			result_dir=Benchmark/"$dataset"_train_results_no_prediction_min_score_"$min_score"_nms_"$nms"
			mkdir $result_dir
			python evaluate_motchallenge_no_prediction.py --mot_dir=./$dataset/train --detection_dir=./resources/detections/"$dataset"_train --output_dir ./"$result_dir" --min_confidence="$min_score" --nms_max_overlap="$nms" --nn_budget=100
			cd $Evaluation_Tool_Dir
			python -m motmetrics.apps.eval_motchallenge ./$dataset/train  $DeepSortDet_DIR/$result_dir |& tee $DeepSortDet_DIR/"$result_dir".txt
	
			cd $DeepSortDet_DIR
			result_dir=Benchmark/"$dataset"_train_results_with_prediction_min_score_"$min_score"_nms_"$nms"
			mkdir $result_dir			
			python evaluate_motchallenge.py --mot_dir=./$dataset/train --detection_dir=./resources/detections/"$dataset"_train --output_dir ./"$result_dir" --min_confidence="$min_score" --nms_max_overlap="$nms" --nn_budget=100
			cd $Evaluation_Tool_Dir
			python -m motmetrics.apps.eval_motchallenge ./$dataset/train  $DeepSortDet_DIR/$result_dir |& tee $DeepSortDet_DIR/"$result_dir".txt

		done
	done

	#python -m motmetrics.apps.eval_motchallenge ./$dataset/train  ./DeepSort_tracking_results/"$dataset"_train_results
	echo "================================================================="
done


<<COMMENT
MIN_CONFIDENCE=(-0.1 -0.05 0 0.05 0.1 0.15 0.2 0.25 0.3) #0.35 0.4 0.45 0.5
NMS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)



NMS=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
MIN_CONFIDENCE=(0.25 0.3)
NMS=(0.5)



COMMENT




