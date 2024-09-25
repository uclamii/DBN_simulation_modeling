"""
    Paper:
    Authors:

    Journal: TBD YYYY

    Contact:

    To run the pipeline and return the RAUS track1, run:

```shell
$ screen -S egfr_reduction40_ge_raus_track1 python egfr_reduction40_ge_yr1-4observations_nocounts_track1.py

```

    To run the pipeline and return all structures for the full network (contemporals, initial conditions, and temporal structures), run:

```shell
$ screen -S egfr_reduction40_ge_wstatic_raus_track2:3 python egfr_reduction40_ge_wstatic_yr0observations_nocounts_track2:3.py & screen -S egfr_reduction40_ge_yr0-1_yr1outasobs_raus_track2:3 python egfr_reduction40_ge_yr0-1observations_yr1outasobs_nocounts_track2:3.py & screen -S egfr_reduction40_ge_raus_track1 python egfr_reduction40_ge_yr1-4observations_nocounts_track1.py & screen -S egfr_reduction40_ge_wyr1_raus_track2:3 python egfr_reduction40_ge_yr1observations_nocounts_track2:3.py

```

"""


import os
from multiprocessing import Pool

os.chdir("./RAUS")
from RAUS_CONSTANTS import *

track1 = (
    "track1_block1.py --file_name_train "
    + Combined_Train
    + " --file_name_valid "
    + Combined_Valid
    + " --file_name_test "
    + Combined_Test
    + "  --max_iter "
    + max_iterations
    + "   --sequence_length_dbn "
    + seq_len_dbn
    + "   --track "
    + Track1
    + "  --site "
    + Combined_counts
    + "  --adjusted "
    + Inter_DBN_Substructure
    + "   --outcome_name "
    + Outcome
    + "  --rank_filter "
    + rank_filter
    + "  --cv_rank_filter2 "
    + Inter_DBN_Filter_keep_counts
    + "  --select_percentile 1 --percentile 80 --max_fan_in "
    + max_fan_in
    + "   --clipback ''"
    
    + "  --clipfront ''"
    
    + "  --cols_start "
    + Intra_Inter_COLS_Start
    + "  --cols_end "
    + COLS_End
    + "  --COLS "
    + Inter_DBN_COLS
    + "  --TARGET "
    + TARGET_InitialCondition_Intra_Inter
    + " ",
    "track1_block2.py --file_name_train "
    + Combined_Train
    + " --file_name_valid "
    + Combined_Valid
    + " --file_name_test "
    + Combined_Test
    + "  --max_iter "
    + max_iterations
    + "   --sequence_length_dbn "
    + seq_len_dbn
    + "   --track "
    + Track1
    + "  --site "
    + Combined_counts
    + "  --adjusted "
    + Inter_DBN_Substructure
    + "   --outcome_name "
    + Outcome
    + "  --select_percentile 1 --percentile 80 --rank_filter "
    + rank_filter
    + "  --chi2_rank_filter2 "
    + Inter_DBN_Filter_keep_counts
    + "  --max_fan_in "
    + max_fan_in
    + "   --clipback ''"
    
    + "  --clipfront ''"
    
    + "  --cols_start "
    + Intra_Inter_COLS_Start
    + "  --cols_end "
    + COLS_End
    + "  --COLS "
    + Inter_DBN_COLS
    + "  --TARGET "
    + TARGET_InitialCondition_Intra_Inter
    + " ",
    "track1_block3.py --file_name_train "
    + Combined_Train
    + " --file_name_valid "
    + Combined_Valid
    + " --file_name_test "
    + Combined_Test
    + "  --max_iter "
    + max_iterations
    + "   --sequence_length_dbn "
    + seq_len_dbn
    + "   --track "
    + Track1
    + "  --site "
    + Combined_counts
    + "  --adjusted "
    + Inter_DBN_Substructure
    + "   --outcome_name "
    + Outcome
    + "  --rank_filter "
    + rank_filter
    + "  --ig_rank_filter2 "
    + Inter_DBN_Filter_keep_counts
    + "  --select_percentile 1 --percentile 80 --max_fan_in "
    + max_fan_in
    + "   --clipback ''"
    
    + "  --clipfront ''"
    
    + "  --cols_start "
    + Intra_Inter_COLS_Start
    + "  --cols_end "
    + COLS_End
    + "  --COLS "
    + Inter_DBN_COLS
    + "  --TARGET "
    + TARGET_InitialCondition_Intra_Inter
    + " ",
)


track_compete_fullnetwork = (
    "track_compete_fullnetwork.py --track "
    + Track1
    + "  --site "
    + Combined_counts
    + "  --adjusted "
    + Inter_DBN_Substructure
    + "   --outcome_name "
    + Outcome
    + " ",
)


def run_process(process):
    os.system("python {}".format(process))
    return process


pool = Pool(processes=4)
pool.map(run_process, track1)
pool.map(run_process, track_compete_fullnetwork)
