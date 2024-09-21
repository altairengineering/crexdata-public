import json
from statistics import mean, stdev

months = [i for i in range(21)]
cases = ["noop", "opt_tr"]
score_field = "mcc"
period_size = 4

months_cases_average_stats = {}
months_models_pt = {}
models_per_period_per_month = {}
train_opt_times_per_period_per_month = {}


for month in months:
    months_models_pt[month] = {"retrain": 0, "optimisation": 0}

for month in months:
    case_scores = {}
    months_cases_average_stats[month] = {}
    models_per_period_per_month[month] = {}
    seen_models = {}

    for case in cases:
        with open(f"../data/reader_files/reader_{case}_month_{month}/reader_model_runtime_scores.txt") as inp, \
            open(f"./score_csvs/{case}/month{month}/scores.csv","w") as outsc,\
            open(f"./score_csvs/{case}/month{month}/model_changes_tr.csv", "w") as outtrch,\
            open(f"./score_csvs/{case}//month{month}/model_changes_opt.csv", "w") as outoptch:

            case_scores[case] = []
            prev_record = None
            week = 4
            printstr=[]
            printsopt=[]

            all_scores = []
            for line in inp:

                current_period = int(week/period_size) - 1
                if current_period not in models_per_period_per_month[month]:
                    models_per_period_per_month[month][current_period] = []

                record = eval(line)
                time = week
                score = json.loads(record["batch_metrics"])[score_field]

                all_scores.append(score)

                model_id = record["model_version"]
                creation_method = record["creation_method"]
                if case == "opt_tr" and model_id >= 0:
                    if model_id not in seen_models:
                        seen_models[record["model_version"]] = 1
                        models_per_period_per_month[month][current_period].append([model_id, creation_method])

                case_scores[case].append((time, score))
                outsc.write(",".join([str(time),str(score)])+"\n")
                if prev_record is not None and prev_record["model_version"] != record["model_version"]:
                    prev_score = json.loads(prev_record["batch_metrics"])[score_field]
                    if record["creation_method"] == "retrain":
                        outtrch.write(",".join([str(time-1), str(prev_score)])+"\n")
                    else:
                        outoptch.write(",".join([str(time-1), str(prev_score)])+"\n")
                prev_record = record
                week = week + 1

        months_cases_average_stats[month][case] = [mean(all_scores), stdev(all_scores)]
        model_pt_times = {}
        if case == "opt_tr":
            with open(f"../data/reader_files/reader_{case}_month_{month}/reader_model_versions.txt") as inp:
                for line in inp:
                    record = eval(line)
                    method = record["creation_method"]
                    model = record["id"]
                    model_pt_times[model] = record["pt"]
                    months_models_pt[month][method] = months_models_pt[month][method] + record["pt"]

        train_opt_times_per_period_per_month[month] = {}
        for period in models_per_period_per_month[month]:
            train_opt_times_per_period_per_month[month][period] = {"retrain": 0, "optimisation": 0}
            for model_id_method in models_per_period_per_month[month][period]:
                train_opt_times_per_period_per_month[month][period][model_id_method[1]] = (
                    train_opt_times_per_period_per_month[month][period][model_id_method[1]] + model_pt_times[model_id_method[0]])
    print(case_scores)
    with open(f"./score_csvs/opt_tr/month{month}/improvement.csv", "w") as imp,\
        open(f"./score_csvs/opt_tr/month{month}/improvement_pos.csv", "w") as imp_pos,\
        open(f"./score_csvs/opt_tr/month{month}/improvement_neg.csv", "w") as imp_neg:
        for i in range(len(case_scores[cases[0]])):
            week = case_scores[cases[0]][i][0]
            div = case_scores[cases[0]][i][1]
            if div <= 0.05:
                 div = 0.05
            increase = case_scores[cases[1]][i][1] - div
            percent = increase / abs(div) * 100
            print((week,increase,div,percent))
            imp.write(",".join([str(week), str(percent)])+"\n")
            if percent < 0:
                imp_neg.write(",".join([str(week), str(percent)])+"\n")
            else:
                imp_pos.write(",".join([str(week), str(percent)])+"\n")
print(models_per_period_per_month)
print(train_opt_times_per_period_per_month)

avg_per_period_production_times = {}

for month in train_opt_times_per_period_per_month:
    retrain_times = []
    optimisation_times = []
    for period in train_opt_times_per_period_per_month[month]:
        retrain_times.append(train_opt_times_per_period_per_month[month][period]["retrain"])
        optimisation_times.append(train_opt_times_per_period_per_month[month][period]["optimisation"])
    print(retrain_times)
    print(optimisation_times)
    print(f"{month},{mean(retrain_times)},{mean(optimisation_times)},{stdev(retrain_times)},{stdev(optimisation_times)}")
    avg_per_period_production_times[month] =  f"{month},{mean(retrain_times)},{mean(optimisation_times)},{stdev(retrain_times)},{stdev(optimisation_times)}"

for case in cases:
    with open(f"./score_csvs/{case}/stats.csv", "w") as out:
        for month in months:
            out.write(",".join([str(x) for x in [month, months_cases_average_stats[month][case][0],
                                                 months_cases_average_stats[month][case][1]]])+"\n")
    if case == "opt_tr":
        with open(f"./score_csvs/{case}/production_times.csv","w") as out:
            for month in months:
                out.write(",".join([str(x) for x in [month, months_models_pt[month]["retrain"],
                                                     months_models_pt[month]["optimisation"]]])+"\n")
        with open(f"./score_csvs/{case}/per_period_production_times.csv","w") as out:
            for month in months:
                out.write(avg_per_period_production_times[month]+"\n")

    print(months_cases_average_stats)
