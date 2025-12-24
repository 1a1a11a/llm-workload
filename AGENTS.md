# Project Context (read first)
- Prefer small, reviewable diffs.
- Before coding: restate the plan in 3–7 bullets and list files you will touch.

## Data Location
- data is stored in `/scratch/juncheng/data/prefix_cache/data/` and if the file has `per_model` in the filename, please use the per-model data. There is also a `metrics_30day.csv` file that contains all data. 
- 

## Analysis Scripts 
- output: please save all output figures to `figures/` directory.
- output: please keep the folder name of the trace in the output path for easy identification. For example, if the trace is `/scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/1000k/DeepSeek-R1.csv`, please save the output to `figures/$analysis_type/1000k/`.

## Coding Standards
- Follow existing patterns in nearby files.
- Add/maintain types; avoid `any` unless justified.
- Error handling: no need to use exceptions for control flow.
- Please always show P10 and P90 in boxplot whisker, please always show mean
- Please always use process pool instead of threadpool in Python scripts



## Test
- When you finish your code, please first test uing `/scratch/juncheng/data/prefix_cache/data/metrics_30day/per_model/100k/23346232-a0be-5448-91be-596f7ab832c2.csv`.
- 