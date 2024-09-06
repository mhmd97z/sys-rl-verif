MMTC_LOCAL_OBS_VAR_COUNT = 6 # device, repetition, delay, sla_level, n_utilized_prbs, prev_sla_violation
EMBB_LOCAL_OBS_VAR_COUNT = 13 # cbr_traffic, cbr_th, cbr_prb, cbr_queue, cbr_snr, vbr_traffic, vbr_th, vbr_prb, vbr_queue, vbr_snr, sla_level, n_utilized_prbs, prev_sla_violation

SHARED_STATE_VAR_COUNT = 3 * (MMTC_LOCAL_OBS_VAR_COUNT+EMBB_LOCAL_OBS_VAR_COUNT) + 2  # mean(mmtc), mean(embb), min(mmtc), min(embb), max(mmtc), max(embb), len(mmtc), len(embb)
AUG_LOCAL_STATE_VAR_COUNT = 6 # device_count, (delay), cbr_traffic, (cbr_queue), vbr_traffic, (vbr_queue), reamaining_embb_count, reamaining_mmtc_count, remaining_prbs

MAX_PRB_ALLOCATION = 400
