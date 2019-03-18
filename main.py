import os
import numpy as np
import pandas as pd
import copy

from source import DATA_DIR

class PopulationPredict(object):
    def __init__(self):
        self.df_pop = pd.read_csv(os.path.join(DATA_DIR, 'popu_predict', "PopulationData.csv"))
        self.df_deadRate = pd.read_csv(os.path.join(DATA_DIR, 'popu_predict', "DeadRate.csv"))
        self.df_ferRate = pd.read_csv(os.path.join(DATA_DIR, 'popu_predict', "FertilityRate.csv"))
        self.df_mig = pd.read_csv(os.path.join(DATA_DIR, 'popu_predict', "Migrate.csv"))
        self.df_netOut = pd.read_csv(os.path.join(DATA_DIR, 'popu_predict', "Netout.csv"))
        self.df_mfr = pd.read_csv(os.path.join(DATA_DIR, 'popu_predict', "NewbornMFR.csv"))

        self.df_ent = pd.read_csv(os.path.join(DATA_DIR, 'cover_rate', "Enterprise.csv"))
        self.df_pub = pd.read_csv(os.path.join(DATA_DIR, 'cover_rate', "Public.csv"))
        self.df_resi = pd.read_csv(os.path.join(DATA_DIR, 'cover_rate', "Resident.csv"))

        print()

    def pop_exec(self, hukou, province, rural_result=None):
        if hukou == "R":
            if rural_result is not None:
                raise ValueError
        elif hukou == "U":
            if rural_result is None:
                raise ValueError
        else:
            raise ValueError

        deadRate_m = self.df_deadRate[(self.df_deadRate.Hukou == hukou)
                                      & (self.df_deadRate.Gender == "M")][province].reset_index(drop=True)
        deadRate_f = self.df_deadRate[(self.df_deadRate.Hukou == hukou)
                                      & (self.df_deadRate.Gender == "F")][province].reset_index(drop=True)

        migRate_m = self.df_mig[(self.df_mig.Gender == "M")][province].reset_index(drop=True)
        migRate_f = self.df_mig[(self.df_mig.Gender == "F")][province].reset_index(drop=True)

        net_out_m = self.df_netOut[(self.df_netOut.Hukou == hukou)
                                   & (self.df_netOut.Gender == "M")][province].values[0]
        net_out_f = self.df_netOut[(self.df_netOut.Hukou == hukou)
                                   & (self.df_netOut.Gender == "F")][province].values[0]

        df_m = self.df_pop[(self.df_pop.Hukou == hukou)
                           & (self.df_pop.Gender == "M")].loc[:, ["Age", province]].reset_index(drop=True)
        df_f = self.df_pop[(self.df_pop.Hukou == hukou)
                           & (self.df_pop.Gender == "F")].loc[:, ["Age", province]].reset_index(drop=True)

        df_m["label"] = ["N", ] * 33 + ["M", ] * 32 + ["O", ] * 36
        df_f["label"] = ["N", ] * 23 + ["M", ] * 32 + ["O", ] * 46

        result = [{"M": copy.deepcopy(df_m), "F": copy.deepcopy(df_f)}, ]

        for i in range(40):
            df_f_canborn = df_f[(df_f.Age >= 15) & (df_f.Age <= 49)][province].reset_index(drop=True)
            df_fer_r = self.df_ferRate["Rural"]
            newborn = df_f_canborn * df_fer_r / 100
            newborn_m = newborn * self.df_mfr[province][self.df_mfr.Hukou == hukou].values[0]
            newborn_f = newborn - newborn_m

            df_m = df_m.shift(1)
            df_m.loc[:, 'Age'] = df_m['Age'] + 1
            df_m = df_m.fillna(0.)
            df_m.loc[0, province] = newborn_m.sum()
            df_m.loc[0, 'label'] = "N"

            df_f = df_f.shift(1)
            df_f.loc[:, 'Age'] = df_f['Age'] + 1
            df_f = df_f.fillna(0.)
            df_f.loc[0, province] = newborn_f.sum()
            df_f.loc[0, 'label'] = "N"

            current_deathRate_m = deadRate_m.shift(1).fillna(0)
            current_deathRate_f = deadRate_f.shift(1).fillna(0)

            if hukou == "R":
                df_m.loc[:, province] = df_m[province] * (1 - current_deathRate_m - migRate_m)
                df_f.loc[:, province] = df_f[province] * (1 - current_deathRate_f - migRate_f)

            else:
                df_m.loc[:, province] = df_m[province] * (1 - current_deathRate_m) + rural_result[i]["M"][province] * migRate_m
                df_f.loc[:, province] = df_f[province] * (1 - current_deathRate_f) + rural_result[i]["F"][province] * migRate_f

            ##todo remove in the future
            multi_out = 50
            multi_in = 100

            out_death_rate_m = (1 - deadRate_m.loc[25:45]).cumprod().values[-1]
            out_death_rate_f = (1 - deadRate_f.loc[25:45]).cumprod().values[-1]

            df_m.loc[20:30, province] = df_m[province][20:31] - net_out_m / multi_out
            df_f.loc[20:30, province] = df_f[province][20:31] - net_out_f / multi_out

            df_m.loc[40:50, province] = df_m[province][40:51] + net_out_m / multi_in * out_death_rate_m
            df_f.loc[40:50, province] = df_f[province][40:51] + net_out_f / multi_in * out_death_rate_f

            result_dict = {"M": copy.deepcopy(df_m), "F": copy.deepcopy(df_f)}
            result.append(result_dict)

        return result

    def pens_exec(self, province, work_age, retire_age):
        rural_result = self.pop_exec("R", province)
        urban_result = self.pop_exec("U", province, rural_result)

        df_retired_e_m = copy.deepcopy(urban_result[0]["M"]
                                       [retire_age["M"]["E"]:]).reset_index(drop=True)
        df_retired_e_f = copy.deepcopy(urban_result[0]["F"]
                                       [retire_age["F"]["E"]:]).reset_index(drop=True)
        df_retired_p_m = copy.deepcopy(urban_result[0]["M"]
                                       [retire_age["M"]["P"]:]).reset_index(drop=True).drop(columns=["label"])
        df_retired_p_f = copy.deepcopy(urban_result[0]["F"]
                                       [retire_age["F"]["P"]:]).reset_index(drop=True).drop(columns=["label"])

        df_retired_u_m = copy.deepcopy(urban_result[0]["M"]
                                       [retire_age["M"]["R"]:]).reset_index(drop=True)
        df_retired_u_f = copy.deepcopy(urban_result[0]["F"]
                                       [retire_age["F"]["R"]:]).reset_index(drop=True)
        df_retired_r_m = copy.deepcopy(rural_result[0]["M"]
                                       [retire_age["M"]["R"]:]).reset_index(drop=True)
        df_retired_r_f = copy.deepcopy(rural_result[0]["F"]
                                       [retire_age["F"]["R"]:]).reset_index(drop=True)

        df_retired_e_m.loc[:, province] = df_retired_e_m[province] * self.df_ent["male"][0]
        df_retired_e_f.loc[:, province] = df_retired_e_f[province] * self.df_ent["female"][0]
        df_retired_p_m.loc[:, province] = df_retired_p_m[province] * self.df_pub["male"][0]
        df_retired_p_f.loc[:, province] = df_retired_p_f[province] * self.df_pub["female"][0]

        df_retired_u_m.loc[:, province] = df_retired_u_m[province] \
                                          * (1 - self.df_ent["male"][0] - self.df_pub["male"][0]) \
                                          * self.df_resi["male"][0]
        df_retired_u_f.loc[:, province] = df_retired_u_f[province] \
                                          * (1 - self.df_ent["female"][0] - self.df_pub["female"][0]) \
                                          * self.df_resi["female"][0]

        df_retired_r_m.loc[:, province] = df_retired_r_m[province] * self.df_resi["male"]
        df_retired_r_f.loc[:, province] = df_retired_r_f[province] * self.df_resi["female"]


        deadRate_u_m = self.df_deadRate[(self.df_deadRate.Hukou == "U")
                                      & (self.df_deadRate.Gender == "M")][province].reset_index(drop=True)
        deadRate_u_f = self.df_deadRate[(self.df_deadRate.Hukou == "U")
                                      & (self.df_deadRate.Gender == "F")][province].reset_index(drop=True)

        deadRate_r_m = self.df_deadRate[(self.df_deadRate.Hukou == "R")
                                        & (self.df_deadRate.Gender == "M")][province].reset_index(drop=True)
        deadRate_r_f = self.df_deadRate[(self.df_deadRate.Hukou == "R")
                                        & (self.df_deadRate.Gender == "F")][province].reset_index(drop=True)

        for i in range(1, len(urban_result)):
            new_label_m = "M"
            new_label_f = "M"
            if i >= retire_age["M"]["E"] - 32:
                new_label_m = "N"
            if i >= retire_age["F"]["E"] - 32:
                new_label_f = "N"

            df_retired_e_m = df_retired_e_m.shift(1)
            df_retired_e_f = df_retired_e_f.shift(1)
            df_retired_p_m = df_retired_p_m.shift(1)
            df_retired_p_f = df_retired_p_f.shift(1)

            df_retired_u_m = df_retired_u_m.shift(1)
            df_retired_u_f = df_retired_u_f.shift(1)
            df_retired_r_m = df_retired_r_m.shift(1)
            df_retired_r_f = df_retired_r_f.shift(1)

            df_retired_e_m.loc[0, "label"] = new_label_m
            df_retired_e_f.loc[0, "label"] = new_label_f

            df_retired_e_m.loc[:, "Age"] = df_retired_e_m["Age"] + 1
            df_retired_e_f.loc[:, "Age"] = df_retired_e_f["Age"] + 1
            df_retired_p_m.loc[:, "Age"] = df_retired_p_m["Age"] + 1
            df_retired_p_f.loc[:, "Age"] = df_retired_p_f["Age"] + 1

            df_retired_u_m.loc[:, "Age"] = df_retired_u_m["Age"] + 1
            df_retired_u_f.loc[:, "Age"] = df_retired_u_f["Age"] + 1
            df_retired_r_m.loc[:, "Age"] = df_retired_r_m["Age"] + 1
            df_retired_r_f.loc[:, "Age"] = df_retired_r_f["Age"] + 1

            df_retired_e_m = df_retired_e_m.fillna(retire_age["M"]["E"])
            df_retired_e_f = df_retired_e_f.fillna(retire_age["F"]["E"])
            df_retired_p_m = df_retired_p_m.fillna(retire_age["M"]["P"])
            df_retired_p_f = df_retired_p_f.fillna(retire_age["F"]["P"])

            df_retired_u_m = df_retired_u_m.fillna(retire_age["M"]["R"])
            df_retired_u_f = df_retired_u_f.fillna(retire_age["F"]["R"])
            df_retired_r_m = df_retired_r_m.fillna(retire_age["M"]["R"])
            df_retired_r_f = df_retired_r_f.fillna(retire_age["F"]["R"])

            urban_dict = urban_result[i]
            rural_dict = rural_result[i]

            df_u_m = urban_dict["M"]
            df_u_f = urban_dict["F"]

            df_r_m = rural_dict["M"]
            df_r_f = rural_dict["F"]

            new_retire_e_m = df_u_m[province][retire_age["M"]["E"]] * self.df_ent["male"][i]
            new_retire_e_f = df_u_f[province][retire_age["F"]["E"]] * self.df_ent["female"][i]
            new_retire_p_m = df_u_m[province][retire_age["M"]["P"]] * self.df_pub["male"][i]
            new_retire_p_f = df_u_f[province][retire_age["F"]["P"]] * self.df_pub["female"][i]

            new_retire_u_m = df_u_m[province][retire_age["M"]["R"]] \
                             * (1 - self.df_ent["male"][i] - self.df_pub["male"][i]) \
                             * self.df_resi["male"][i]

            new_retire_u_f = df_u_f[province][retire_age["F"]["R"]] \
                             * (1 - self.df_ent["female"][i] - self.df_pub["female"][i]) \
                             * self.df_resi["female"][i]

            new_retire_r_m = df_r_m[province][retire_age["M"]["R"]] * self.df_resi["male"][i]
            new_retire_r_f = df_r_f[province][retire_age["F"]["R"]] * self.df_resi["female"][i]

            df_retired_e_m.loc[:, province] = df_retired_e_m[province] \
                                              * (1 - deadRate_u_m[retire_age["M"]["E"]:].reset_index(drop=True))
            df_retired_e_f.loc[:, province] = df_retired_e_f[province] \
                                              * (1 - deadRate_u_f[retire_age["F"]["E"]:]).reset_index(drop=True)
            df_retired_p_m.loc[:, province] = df_retired_p_m[province] \
                                              * (1 - deadRate_u_m[retire_age["M"]["P"]:].reset_index(drop=True))
            df_retired_p_f.loc[:, province] = df_retired_p_f[province] \
                                              * (1 - deadRate_u_f[retire_age["F"]["P"]:]).reset_index(drop=True)

            df_retired_u_m.loc[:, province] = df_retired_u_m[province] \
                                              * (1 - deadRate_u_m[retire_age["M"]["R"]:].reset_index(drop=True))
            df_retired_u_f.loc[:, province] = df_retired_u_f[province] \
                                              * (1 - deadRate_u_f[retire_age["F"]["R"]:]).reset_index(drop=True)
            df_retired_r_m.loc[:, province] = df_retired_r_m[province] \
                                              * (1 - deadRate_r_m[retire_age["M"]["R"]:].reset_index(drop=True))
            df_retired_r_f.loc[:, province] = df_retired_r_f[province] \
                                              * (1 - deadRate_r_f[retire_age["F"]["R"]:]).reset_index(drop=True)

            df_retired_e_m.loc[0, province] = new_retire_e_m
            df_retired_e_f.loc[0, province] = new_retire_e_f
            df_retired_p_m.loc[0, province] = new_retire_p_m
            df_retired_p_f.loc[0, province] = new_retire_p_f

            df_retired_u_m.loc[0, province] = new_retire_u_m
            df_retired_u_f.loc[0, province] = new_retire_u_f
            df_retired_r_m.loc[0, province] = new_retire_r_m
            df_retired_r_f.loc[0, province] = new_retire_r_f


            #
            #
            #
            # df_m["P"] = df_m[province] * (self.df_ent


            print()
        # for


if __name__ == "__main__":
    pop_predict = PopulationPredict()

    retire_age_ = pd.DataFrame({"M": [60, 65, 60], "F": [50, 55, 50]}, index=["E", "P", "R"])
    work_age_ = pd.DataFrame({"M": [20, 23], "F": [20, 23]}, index=["E", "P"])

    pop_predict.pens_exec("Beijin", work_age_, retire_age_)

