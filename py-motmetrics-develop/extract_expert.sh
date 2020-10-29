#!/bin/bash


# python -m motmetrics.apps.eval_motchallenge ./MOT16/train ../../ensemble_tracking_exports/results/ec_5 

experiment_flag=progress_experts
root_dir=./progress_experts
#root_dir=./$experiment_flag
log_date=`date +"%y-%m-%d"` #-%H-%M-%S


DATASET=("MOT16" )   #  "MOT17" "MOT20" "MOT16" 
for ele in ${!DATASET[@]}; do
	dataset=${DATASET[ele]}
	echo "================================================================="
	echo "dataset = "$dataset
	#python -m motmetrics.apps.eval_motchallenge ./$dataset/train  ./DeepSort_tracking_results_without_predictions/"$dataset"_train_results |& tee DeepSort_tracking_results_without_predictions-"$dataset"_train_results-2020-4-4

	summary_file="$experiment_flag"_"$log_date"_"$dataset"_"summary"
	rm $summary_file
	for sub_dir in $(ls -d $root_dir/$dataset/*); do  # | grep -v "MOT16"
		echo "$sub_dir"
		script_file="$experiment_flag"_"$log_date"_`echo $sub_dir | cut -d'/' -f4`
		echo "script_file="$script_file
		python -m motmetrics.apps.eval_motchallenge ./$dataset/train  "$sub_dir" |& tee $script_file
		#echo python -m motmetrics.apps.eval_motchallenge ./$dataset/train  "$sub_dir" |& tee $script_file
		tac $script_file | head -1  | tail -1 >> "$summary_file"
		#echo "================================================================="
	done

done


