import pandas as pd

import arff
from skmultilearn.dataset import load_from_arff

def prepare_iris_dataset(self, filename):
    self.original_filename = filename
    self.class_name = 'class'
    self.df = pd.read_csv(filename, skipinitialspace=True)


def prepare_wine_dataset(self, filename):
    self.original_filename = filename
    self.class_name = 'quality'
    self.df = pd.read_csv(filename, skipinitialspace=True, sep=';')


def prepare_adult_dataset(self, filename):
    self.original_filename = filename
    self.class_name = 'class'
    self.df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
    columns2remove = ['fnlwgt', 'education-num']
    self.df.drop(columns2remove, inplace=True, axis=1)


def prepare_german_dataset(self, filename):
    self.original_filename = filename
    self.class_name = 'default'
    self.df = pd.read_csv(filename, skipinitialspace=True)
    self.df.columns = [c.replace('=', '') for c in self.df.columns]


def prepare_compass_dataset(self, filename, binary=False):
    self.original_filename = filename
    df = pd.read_csv(filename, delimiter=',', skipinitialspace=True)

    columns = ['age', 'age_cat', 'sex', 'race', 'priors_count', 'days_b_screening_arrest', 'c_jail_in', 'c_jail_out',
               'c_charge_degree', 'is_recid', 'is_violent_recid', 'two_year_recid', 'decile_score', 'score_text']

    df = df[columns]

    df['days_b_screening_arrest'] = np.abs(df['days_b_screening_arrest'])

    df['c_jail_out'] = pd.to_datetime(df['c_jail_out'])
    df['c_jail_in'] = pd.to_datetime(df['c_jail_in'])
    df['length_of_stay'] = (df['c_jail_out'] - df['c_jail_in']).dt.days
    df['length_of_stay'] = np.abs(df['length_of_stay'])

    df['length_of_stay'].fillna(df['length_of_stay'].value_counts().index[0], inplace=True)
    df['days_b_screening_arrest'].fillna(df['days_b_screening_arrest'].value_counts().index[0], inplace=True)

    df['length_of_stay'] = df['length_of_stay'].astype(int)
    df['days_b_screening_arrest'] = df['days_b_screening_arrest'].astype(int)

    if binary:
        def get_class(x):
            if x < 7:
                return 'Medium-Low'
            else:
                return 'High'

        df['class'] = df['decile_score'].apply(get_class)
    else:
        df['class'] = df['score_text']

    del df['c_jail_in']
    del df['c_jail_out']
    del df['decile_score']
    del df['score_text']

    self.df = df
    self.class_name = 'class'


def prepare_churn_dataset(self, filename):
    self.class_name = 'churn'
    df = pd.read_csv(filename, skipinitialspace=True, na_values='?', keep_default_na=True)
    columns2remove = ['phone number']
    df.drop(columns2remove, inplace=True, axis=1)
    self.df = df


def prepare_yeast_dataset(self, filename):
    df = pd.DataFrame(arff.loadarff(filename)[0])

    for col in df.columns[-14:]:
        df[col] = df[col].apply(pd.to_numeric)

    cols_Y = [col for col in df.columns if col.startswith('Class')]
    # cols_X = [col for col in df.columns if col not in cols_Y]

    self.df = df
    self.class_name = cols_Y


def prepare_medical_dataset(self, filename):
    data = load_from_arff(filename, label_count=45, load_sparse=False, return_attribute_definitions=True)
    cols_X = [i[0] for i in data[2]]
    cols_Y = [i[0] for i in data[3]]
    X_med_df = pd.DataFrame(data[0].todense(), columns=cols_X)
    y_med_df = pd.DataFrame(data[1].todense(), columns=cols_Y)
    df = pd.concat([X_med_df, y_med_df], 1)
    self.df = df
    self.class_name = cols_Y


def prepare_bank_dataset(self, filename):
    """
    from https://www.kaggle.com/aniruddhachoudhury/credit-risk-model#train.csv/home/riccardo/Scaricati/bank.csv
    :param filename:
    :return:
    """
    self.class_name = 'give_credit'
    self.df = pd.read_csv(filename, skipinitialspace=True, keep_default_na=True, index_col=0)


def prepare_fico_dataset(self, filename):
    self.class_name = 'RiskPerformance'
    self.df = pd.read_csv(filename, skipinitialspace=True, keep_default_na=True)