import pandas as pd
import numpy as np

from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.manifold import TSNE

import umap.umap_ as umap
#import umap.plot

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


import random
from lore_sa.dataset import TabularDataset, Dataset
from lore_sa.bbox import sklearn_classifier_bbox, AbstractBBox
from lore_sa.neighgen import RandomGenerator
from lore_sa.neighgen.neighborhood_generator import NeighborhoodGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from lore_sa.encoder_decoder import EncDec, ColumnTransformerEnc
from lore_sa.lore import TabularRandomGeneratorLore, Lore

from sklearn.metrics import pairwise_distances

import altair as alt

alt.data_transformers.enable('default', max_rows=None)


class IdentityEncoder(EncDec):
    """
    It provides an interface to access Identity encoding functions.
    """

    def __init__(self, dataset_descriptor):
        super().__init__(dataset_descriptor)

    def encode(self, x: np.array) -> np.array:
        """
        It applies the encoder to the input features

        :param [Numpy array] x: Array to encode
        :return [Numpy array]: Encoded array
        """
        return x

    def get_encoded_features(self):
        """
        Provides a dictionary with the new encoded features name and the new index
        :return:
        """
        return self.encoded_features

    def decode(self, x: np.array) -> np.array:
        return x

    def decode_target_class(self, x: np.array):
        return x

class ProbabilitiesWeightBasedGenerator(NeighborhoodGenerator):

    def __init__(self, bbox: AbstractBBox, dataset: Dataset, encoder: EncDec, ocr=0.1):
        super().__init__(bbox, dataset, encoder, ocr)
        self.neighborhood = None
        self.preprocess = bbox.bbox.named_steps.get('columntransformer')

    def generate(self, z: np.array, num_instances: int, descriptor: dict, encoder):
        x = encoder.decode(z.reshape(1, -1))[0]
        z1 = self.preprocess.transform(x.reshape(1, -1))[0]

        perturbed_arrays = []

        choices = [0, 1, 2, 3, 4]

        covid_weights = [
            [0.0, 0.25, 0.13, 0.36, 0.00],   # 0
            [0.0, 0.79, 0.19, 0.02, 0.00],   # 1
            [0.0, 0.06, 0.66, 0.26, 0.01],  # 2
            [0.0, 0.00, 0.09, 0.81, 0.10],  # 3
            [0.0, 0.00, 0.00, 0.11, 0.89],   # 4
        ]

        mobility_weights = [
            [0.0, 0.12, 0.02, 0.45, 0.24],  # 0
            [0.0, 0.95, 0.05, 0.00, 0.00],  # 1
            [0.0, 0.02, 0.64, 0.31, 0.02],  # 2
            [0.0, 0.00, 0.10, 0.80, 0.10],  # 3
            [0.0, 0.14, 0.02, 0.17, 0.68]   # 4
        ]

        for _ in range(num_instances):
            perturbed_arr = z1.copy()

            for val in range(0, 3):  # covid
                ref_val = int(perturbed_arr[val])
                perturbed_arr[val] = random.choices(choices, weights=covid_weights[ref_val])[0]

            for val in range(3, 7):  # mobility
                ref_val = int(perturbed_arr[val])
                perturbed_arr[val] = random.choices(choices, weights=mobility_weights[ref_val])[0]

            perturbed_arr[7] = random.choice(range(42, 442, 7))
            perturbed_arr[8] = random.choice(range(7, 148, 7))

            covid_sec = [f"c{int(v)}" for v in perturbed_arr[0:3]]
            mob_sec = [f"m{int(v)}" for v in perturbed_arr[3:7]]
            tot = [*covid_sec, *mob_sec, perturbed_arr[7], perturbed_arr[8]]

            #dec = encoder.encode([tot])[0]

            perturbed_arrays.append(tot)

        # save to png
        self.neighborhood = encoder.encode(perturbed_arrays)

        return self.neighborhood

    def check_generated(self, filter_function=None, check_fuction=None):
        pass


def load_data_from_csv():
    df = pd.read_csv("datasets/Final_data.csv")

    df = df.loc[:, ['Week5_Covid', 'Week4_Covid', 'Week3_Covid', 'Week5_Mobility', 'Week4_Mobility',
                    'Week3_Mobility', 'Week2_Mobility', 'Days_passed', 'Duration', 'Class_label']]

    mask = df.drop(["Class_label"], axis=1).isna().all(axis=1) & df['Class_label'].notna()
    result_df = df[~mask]
    # df = df.dropna(how = 'all')
    result_df = result_df.fillna("NONE")

    return result_df


def create_and_train_model(result_df):
    y = result_df["Class_label"].values
    X_feat = result_df.loc[:, 'Week5_Covid':'Duration'].values
    covid_categories = ['NONE', 'c1', 'c2', 'c3', 'c4']
    mobility_categories = ['NONE', 'm1', 'm2', 'm3', 'm4']
    enc = OrdinalEncoder(
        categories=[covid_categories, covid_categories, covid_categories, mobility_categories, mobility_categories,
                    mobility_categories, mobility_categories])
    numerical_preprocessor = StandardScaler()
    preprocessor = ColumnTransformer(
        [
            ("ordinal-encoder", enc, list(range(0, 7))),
            ("standard_scaler", numerical_preprocessor, list(range(7, 9))),
        ]
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    model = make_pipeline(preprocessor, clf)

    data_train, data_test, target_train, target_test = train_test_split(
        X_feat, y, random_state=0
    )
    _ = model.fit(data_train, target_train)

    encoded_train = preprocessor.fit_transform(data_train)
    df = pd.DataFrame(encoded_train)
    df.columns = result_df.columns[:-1]
    df.iloc[:, -2:] = result_df.iloc[:, -3:-1]
    df['Class_label'] = result_df['Class_label']
    print(df)

    print(model.score(data_test, target_test))

    return model


def computed_dates_from_offsets(offs: np.array):
    base_date = datetime(2020, 2, 17)
    starting_date = base_date + timedelta(offs[0])
    ending_date = starting_date + timedelta(offs[1])

    return np.array([starting_date, ending_date])


def compute_statistics_distance(res, model):
    bbox = sklearn_classifier_bbox.sklearnBBox(model)
    data = TabularDataset(data=res, class_name='Class_label')
    # these will be used to scale back the original values of temporal dimensions
    model_preprocessor = model.named_steps.get('columntransformer')
    ordinal_encoder = model_preprocessor.named_transformers_.get('ordinal-encoder')
    model_scaler = model_preprocessor.named_transformers_.get('standard_scaler')


    df_15= pd.read_csv('datasets/selected_train_instances_15.csv', sep=';')

    for i in range(0,2):
        x = df_15.iloc[i, :-1].values
    #x = np.array(['c4', 'c4', 'c4', 'm2', 'm2', 'm3', 'm3', np.float64(364), np.float64(28)], dtype=object)

        encoder = ColumnTransformerEnc(data.descriptor)
        generator = ProbabilitiesWeightBasedGenerator(bbox, data, encoder)
        rnd_generator = RandomGenerator(bbox, data, encoder)

        print('Converting the input instance')
        ist_lbl = np.full(1, 'ist').reshape(-1, 1)
        ist_processed = model_preprocessor.transform(x.reshape(1, -1))
        ist_class_lbl = np.array(['c1'], dtype=object).reshape(-1, 1)
        ist_neighb = np.concatenate([ist_lbl, ist_processed, ist_class_lbl], axis=1)

        print('Computing distances of custom generator')
        cst_np_mins, cst_neighb_z, cst_bbox_lbl = measure_distances(data, encoder, generator, x, 'Custom', res, model)
        cst_lbl = np.full(len(cst_np_mins), 'cst').reshape(-1, 1)
        cst_neighb_z_lbl = np.concatenate([cst_lbl, cst_neighb_z, cst_bbox_lbl], axis=1)

        print('Euclidean distances from the original data')
        rnd_np_mins, rnd_neighb_z, rnd_bbox_lbl = measure_distances(data, encoder, rnd_generator, x, 'Random', res, model )
        rnd_lbl = np.full(len(rnd_np_mins), 'rnd').reshape(-1, 1)
        rnd_neighb_z_lbl = np.concatenate([rnd_lbl, rnd_neighb_z,rnd_bbox_lbl], axis=1)

        print('Refactor the traingin dataset')
        trn_lbl = np.full(len(res.values), 'trn').reshape(-1, 1)
        train_features = model_preprocessor.transform(res.values[:, :-1])
        train_dataset = np.concatenate([trn_lbl, train_features, res.values[:, -1].reshape(-1,1) ], axis=1)


        neighbs = np.concatenate([ist_neighb, rnd_neighb_z_lbl, cst_neighb_z_lbl, train_dataset], axis=0)

        #reducer = umap.UMAP(n_neighbors=300,  random_state=1, min_dist=0.3, metric='manhattan')
        reducer = TSNE(perplexity=50, metric='manhattan' )
        proj_neighbs = reducer.fit_transform(neighbs[:, 1: 8])

        time_offsets = model_scaler.inverse_transform(neighbs[:,8:10])
        original_features = ordinal_encoder.inverse_transform(neighbs[:, 1:8])
        time_interval = np.apply_along_axis(computed_dates_from_offsets, 1, time_offsets)

        neighbs = np.concatenate([neighbs, proj_neighbs, time_interval], axis=1)

        neighbs[:, 8:10] = time_offsets
        column_names = ['Instance_gen','Week5_Covid', 'Week4_Covid', 'Week3_Covid', 'Week5_Mobility','Week4_Mobility','Week3_Mobility','Week2_Mobility','Days_passed','Duration','Class_label','x','y','Start_Date','End_date']
        df = pd.DataFrame(neighbs, columns=column_names)
        df.to_csv(f'datasets/new_neigh/selected_instances_{i}.csv', index=False)

        alt_df_dist_o = pd.DataFrame(cst_np_mins, columns=['Distance'])
        alt_df_dist_r = pd.DataFrame(rnd_np_mins, columns=['Distance'])
        print(alt_df_dist_o.head(10))
        alt_df_dist_o['source'] = 'Custom'
        alt_df_dist_r['source'] = 'Random'
        domain_ = ['Random', 'Custom']
        range_ = ['#102ce0', '#fa7907']
        boxplot_df = pd.concat([alt_df_dist_o, alt_df_dist_r], axis=0)
        box_plot = alt.Chart(boxplot_df).mark_boxplot().encode(
            alt.X("Distance:Q"),
            #scale=alt.Scale(zero=False,domain=[2.5,4.5])
            alt.Y("source:N"),
            alt.Color("source:N", scale=alt.Scale(domain=domain_, range=range_))
        ).properties(
            height=150,
            width=400,
            title='Euclidean distances of the neighbourhoods from the instance'
        )
        box_plot.save(f'plot/instance_vs_neigh/boxplot_bay_{i}.pdf')

def measure_distances(data, encoder, generator, x, label: str, res, model, neighb_size:int=1000):
    preprocessor = generator.bbox.bbox.named_steps.get('columntransformer')
    global_mins = []
    for i in range(1):
        # project the instance x and create a neighborhood of given size
        neighb_ohe = generator.generate(encoder.encode(x.reshape(1, -1))[0], neighb_size, data.descriptor, encoder)
        # decode the neighborhood
        neighb = encoder.decode(neighb_ohe)
        bbox_labels = model.predict(neighb).reshape(-1, 1)

        # apply the preprocessor of the model to prepare for copmuting distances
        neighb_Z = preprocessor.transform(neighb)

        # Create the df
        # x_array = (np.array([x]))
        # combined_array = np.concatenate((x_array, neighb), axis=0)
        # processed_combined_array = preprocessor.transform(combined_array)
        # prediction = model.predict(combined_array)
        # enc_pred = encoder.encode_target_class(prediction.reshape(-1, 1))
        #
        # df_neigh_100 = pd.DataFrame(processed_combined_array, columns=res.columns[:-1])
        #
        # df_neigh_100['Class_label'] = enc_pred
        # new_values_days = [int(arr[7]) for arr in neighb]
        # new_values_duration = [int(arr[8]) for arr in neighb]
        #
        # for i in range(1, len(df_neigh_100)):
        #     df_neigh_100.loc[i, 'Days_passed'] = new_values_days[i - 1]
        #     df_neigh_100.loc[i, 'Duration'] = new_values_duration[i - 1]
        # df_neigh_100.loc[0,'Days_passed'] = 245.0
        # df_neigh_100.loc[0, 'Duration'] = 28
        #
        # #add date and time columns
        #
        # base_date = datetime(2020, 2, 17)
        # df_neigh_100['Start_date'] = df_neigh_100['Days_passed'].apply(lambda x: base_date + timedelta(days=x))
        # df_neigh_100['End_date'] = df_neigh_100.apply(lambda row: row['Start_date'] + timedelta(days=row['Duration']), axis=1)
        #
        # df_neigh_100.to_csv(f'datasets/df_neigh_100_{label}.csv')
        # df_umap = pd.DataFrame(projected_neighb)
        # df_umap.to_csv(f"datasets/projected_neigh_{label}_100.csv")
        #
        # projected_train = umapper.projected_train
        # df_train_umap = pd.DataFrame(projected_train)
        # df_train_umap.to_csv("datasets/train_umap_100.csv")

        df_15 = pd.read_csv('datasets/selected_train_instances_15.csv', sep=';')
        #dists = calculate_distance(neighb_Z, preprocessor.transform(data.df.iloc[:, :-1].values))
        dists = calculate_distance(neighb_Z, preprocessor.transform(df_15.iloc[5:6, :-1].values))

        local_mins = np.min(dists, axis=1)
        global_mins.append(local_mins)
        if i % 100 == 0:
            print(f"{i}...\r")
    print()
    np_mins = np.average(np.array(global_mins), axis=0)
    return np_mins, neighb_Z, bbox_labels


def new_lore(res, model):

    instance = res.values[5, : -1]
    #print(instance)
    #prediction = model.predict([instance])
    #print(prediction)

    df_15 = pd.read_csv('datasets/selected_train_instances_15.csv', sep=';')


    bbox = sklearn_classifier_bbox.sklearnBBox(model)
    #data = TabularDataset(data=df_15, class_name="Class_label")
    data = TabularDataset(data=res, class_name='Class_label')
    print(data.df)
    x = df_15.iloc[5, :-1].values
    #x = data.df.iloc[45, :-1].values
    print("instance is:", x)
    print('model prediction is', model.predict([x]))

    #lore = TabularRandomGeneratorLore(bbox, data)
    encoder = ColumnTransformerEnc(data. descriptor)
    surrogate = DecisionTreeSurrogate()
    generator = ProbabilitiesWeightBasedGenerator(bbox, data, encoder)
    proba_lore = Lore(bbox, data, encoder, generator, surrogate)

    print(data.descriptor)
    #rule = lore.explain(x)
    rule = proba_lore.explain(x)
    print(rule)
    print('----- ')
    print(rule['rule'])
    print('----- counterfactual')
    for cr in rule['counterfactuals']:
        print(cr)
        print('-----')

    #for i in range(nrows):
    #    rl = proba_lore.explain(data.df.iloc[i, :-1].values)
    #    if len(rl['counterfactuals']) > 0:
    #        print(i)


def calculate_distance(X1: np.array, X2: np.array, y_1: np.array = None, y_2: np.array = None, metric:str='euclidean'):
    dists = pairwise_distances(X1, X2, metric=metric)

    return dists


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    res = load_data_from_csv()
    model = create_and_train_model(res)
    new_lore(res, model)
    compute_statistics_distance(res, model)

   #UMAPMapper()


