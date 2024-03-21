# TrackEval for pedestrain object

This repository is built based on official TrackEval repository for evaluation on pedestrain tracking quickly.

### Quick Start

1. git clone this repository.
```
git clone https://github.com/imzhangyd/TrackEval-pedes.git
```
2. link your tracking results.
```
name='mott-test'
mkdir "/data/ldap_shared/home/s_zyd/TrackEval-pedes/data/trackers/mot_challenge/MOT17-train/${name}"
ln -s "/ldap_shared/home/s_zyd/MoTT/prediction/20240321_17_12_34/data" "/ldap_shared/home/s_zyd/TrackEval-pedes/data/trackers/mot_challenge/MOT17-train/${name}/data"
```
3. eval
```
python scripts/run_mot_challenge.py --BENCHMARK MOT17 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL $name --METRICS HOTA CLEAR Identity VACE --USE_PARALLEL False --NUM_PARALLEL_CORES 1
```
