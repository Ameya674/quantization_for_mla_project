from torch.utils.data import DataLoader
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tabulate import tabulate
import time
import os


def prepare_data(whole_data,windowed_data,ins_windowed_data,carb_windowes_data,interval,PH60_sample,PH30_sample,PH1_sample):
    train_data = np.zeros(
        (int((whole_data.shape[0] - PH60_sample * 5 - interval) / 5) + 1, interval, 3))

    PH_60 = np.zeros((int((whole_data.shape[0] - PH60_sample * 5 - interval) / 5) + 1, PH60_sample))
    PH_30 = np.zeros((int((whole_data.shape[0] - PH60_sample * 5 - interval) / 5) + 1, PH30_sample))
    PH_1 = np.zeros((int((whole_data.shape[0] - PH60_sample * 5 - interval) / 5) + 1, PH1_sample))

    for i in range(0, train_data.shape[0]):
        train_data[i, :, 0] = windowed_data[i:interval + i]
        train_data[i, :, 1] = ins_windowed_data[i:interval + i]
        train_data[i, :, 2] = carb_windowes_data[i:interval + i]
        PH_60[i] = windowed_data[interval + i:interval + (i) + PH60_sample]
        PH_30[i] = PH_60[i][0:6]
        PH_1[i] = PH_60[i][0]

    return train_data,PH_60,PH_30,PH_1

def carbs_to_operative_carbs(carbs, time_str,time_str_previoussample,max_peak_time_str, meal_time_str, max_time=60, increase_rate=0.111, decrease_rate=0.028):
    # Convert string times to datetime objects
    date_format = '%Y-%m-%d %H:%M:%S'
    current_time = datetime.strptime(time_str, date_format)
    meal_time = datetime.strptime(meal_time_str, date_format)
    previous_time=datetime.strptime(time_str_previoussample,date_format)
    max_peak_time = datetime.strptime(max_peak_time_str, date_format)
    # Convert times to minutes for easier calculation
    current_minutes = current_time.hour * 60 + current_time.minute
    meal_minutes = meal_time.hour * 60 + meal_time.minute
    meal_time_2=meal_minutes+10
    previous_time_minutes=previous_time.hour * 60 +previous_time.minute
    max_peak_time_minutes=max_peak_time.hour*60+max_peak_time.minute

    time_diff = current_minutes - meal_minutes

    if time_diff < 0:
        return 0
    elif 0 <= time_diff < 15:
        return 0
    elif 15 <= time_diff <= max_time:
        carb_eff=carbs * increase_rate * ((current_minutes - meal_time_2) / (current_minutes - previous_time_minutes))
        return carb_eff
    elif max_time<time_diff <48*5 :
        if current_minutes-previous_time_minutes==0:
            print(current_minutes)
        carb_eff=max(0, carbs * (1 - (current_minutes - max_peak_time_minutes)/(current_minutes-previous_time_minutes)*decrease_rate))
        return carb_eff
    else:
        return 0



def bolus_to_active_insulin(insulin, time_str, bolus_time_str, duration=360, time_constant=55):
    # Convert string times to datetime objects
    date_format = '%Y-%m-%d %H:%M:%S'
    current_time = datetime.strptime(time_str, date_format)
    bolus_time = datetime.strptime(bolus_time_str, date_format)
    current_minutes = current_time.hour * 60 + current_time.minute
    # Calculate time difference in minutes
    time_diff = (current_time - bolus_time).total_seconds() / 60
    current_minutes=time_diff


    if time_diff < 0:
        return 0

    if time_diff > duration:
        return 0
    else:
        tau = time_constant * ((1 - (time_constant / duration)) / (1 - 2 * (time_constant / duration)))
        a = 2 * tau / duration
        S = 1 / (1 - a + (1 + a) * np.exp(-duration / tau))

        # Calculate Insulin on Board (IOB) based on Equation 2
        IOB = 1 - S * (1 - a) * ((current_minutes ** 2 / (tau * duration * (1 - a)) - current_minutes / tau - 1) * np.exp(
            -current_minutes / tau) + 1)
        return insulin * IOB

RMSE_list=[]
MAE_list=[]
R_Square=[]
RMSE30_list=[]
MAE30_list=[]
file1 = open('new_results_total_LSTM_Transformer_student128-64.txt', 'w')
pid_2018 = [559]#, 563] #, 570, 588, 575, 591]
# pid_2020 = [540, 552, 544, 567, 584, 596]
pid_year = {2018: pid_2018} #, 2020: pid_2020}
train_data = dict()
for year in list(pid_year.keys()):
    pids = pid_year[year]
    for pid in pids:
        xl = pd.read_csv(f'mix_G_B_C_{year}_{pid}_train.csv')

        xl['Glucose'] = xl['Glucose'].interpolate()
        data_Glucose_Baseline = xl['Glucose']
        date_time = xl['Timestamp']
        updated_data_eff = xl.drop(columns=['Unnamed: 0'])
        updated_data_eff.fillna(0, inplace=True)
        train_eff_data = updated_data_eff.to_numpy()
        indices = np.nonzero(train_eff_data[:, 3])
        meal_time = '2010-12-07 12:00:00'
        carbs = 0
        max_peak_time = meal_time
        peak_carb = 0
        for i in range(train_eff_data.shape[0]):
            if i in indices[0]:
                peak_carb = 0
                meal_time = train_eff_data[i, 0]
                carbs = train_eff_data[i, 3]
            if i > 1:
                if (train_eff_data[i, 0] != train_eff_data[i - 1, 0]):
                    train_eff_data[i, 3] = carbs_to_operative_carbs(carbs, train_eff_data[i, 0], train_eff_data[i - 1, 0],
                                                                max_peak_time, meal_time)
                    if train_eff_data[i, 3] > peak_carb:
                        peak_carb = train_eff_data[i, 3]
                        max_peak_time = train_eff_data[i, 0]
                else:
                    train_eff_data[i, 3] = train_eff_data[i - 1, 3]
            else:
                train_eff_data[i, 3] = carbs_to_operative_carbs(carbs, train_eff_data[i, 0], train_eff_data[i, 0], max_peak_time,
                                                            meal_time)

        indices = np.nonzero(train_eff_data[:, 2])
        insulin_time = '2010-12-07 12:00:00'
        insulin = 0
        max_peak_time = insulin_time
        peak_insulin = 0
        for i in range(train_eff_data.shape[0]):
            if i in indices[0]:
                insulin_time = train_eff_data[i, 0]
                insulin = train_eff_data[i, 2]

            train_eff_data[i, 2] = bolus_to_active_insulin(insulin, train_eff_data[i, 0], insulin_time)


        xl_test = pd.read_csv(f"mix_G_B_C_{year}_{pid}_test.csv")

        xl_test['Glucose'] = xl_test['Glucose'].interpolate()
        data_Glucose_test = xl_test['Glucose']

        date_time = xl_test['Timestamp']

        updated_test_data_eff = xl_test.drop(columns=['Unnamed: 0'])
        updated_test_data_eff.fillna(0, inplace=True)
        test_eff_data = updated_test_data_eff.to_numpy()

        indices = np.nonzero(test_eff_data[:, 3])
        meal_time = '2010-12-07 12:00:00'
        carbs = 0
        max_peak_time = meal_time
        peak_carb = 0
        for i in range(test_eff_data.shape[0]):
            if i in indices[0]:
                peak_carb = 0
                meal_time = test_eff_data[i, 0]
                carbs = test_eff_data[i, 3]
            if i > 1:
                if (test_eff_data[i, 0] != test_eff_data[i - 1, 0]):
                    test_eff_data[i, 3] = carbs_to_operative_carbs(carbs, test_eff_data[i, 0],
                                                                    test_eff_data[i - 1, 0],
                                                                    max_peak_time, meal_time)
                    if test_eff_data[i, 3] > peak_carb:
                        peak_carb = test_eff_data[i, 3]
                        max_peak_time = test_eff_data[i, 0]
                else:
                    test_eff_data[i, 3] = test_eff_data[i - 1, 3]
            else:
                test_eff_data[i, 3] = carbs_to_operative_carbs(carbs, test_eff_data[i, 0], test_eff_data[i, 0],
                                                                max_peak_time,
                                                                meal_time)

        indices = np.nonzero(test_eff_data[:, 2])
        insulin_time = '2010-12-07 12:00:00'
        insulin = 0
        max_peak_time = insulin_time
        peak_insulin = 0
        for i in range(test_eff_data.shape[0]):
            if i in indices[0]:
                insulin_time = test_eff_data[i, 0]
                insulin = test_eff_data[i, 2]

            test_eff_data[i, 2] = bolus_to_active_insulin(insulin, test_eff_data[i, 0], insulin_time)


        unnorm_glucose = train_eff_data[:, 1]
        unnorm_ins = train_eff_data[:, 2]
        unnorm_carb = train_eff_data[:, 3]


        # Reshape to 2D arrays as required by MinMaxScaler
        unnorm_glucose = np.reshape(unnorm_glucose, (-1, 1))

        unnorm_ins = np.reshape(unnorm_ins, (-1, 1))
        unnorm_carb = np.reshape(unnorm_carb, (-1, 1))

        # Create separate MinMaxScaler instances for each signal
        scaler_glucose = MinMaxScaler()
        scaler_insulin = MinMaxScaler()
        scaler_carbs = MinMaxScaler()

        norm_glucose = scaler_glucose.fit_transform(unnorm_glucose)
        norm_ins = scaler_insulin.fit_transform(unnorm_ins)
        norm_carb = scaler_carbs.fit_transform(unnorm_carb)

        norm_glucose = norm_glucose.ravel()
        norm_ins = norm_ins.ravel()
        norm_carb = norm_carb.ravel()


        unnorm_glucose_test = test_eff_data[:, 1]
        unnorm_ins_test = test_eff_data[:, 2]
        unnorm_carb_test = test_eff_data[:, 3]


        unnorm_glucose_test = np.reshape(unnorm_glucose_test, (-1, 1))
        unnorm_ins_test = np.reshape(unnorm_ins_test, (-1, 1))
        unnorm_carb_test = np.reshape(unnorm_carb_test, (-1, 1))

        norm_glucose_test = scaler_glucose.transform(unnorm_glucose_test)
        norm_ins_test = scaler_insulin.transform(unnorm_ins_test)
        norm_carb_test = scaler_carbs.transform(unnorm_carb_test)
        norm_glucose_test = norm_glucose_test.ravel()
        norm_ins_test = norm_ins_test.ravel()
        norm_carb_test = norm_carb_test.ravel()

        in_window = 180
        interval = int(in_window / 10 * 2)
        out_60 = 12
        out_30 = 6
        out_1 = 1
        test_x_tot, test_60_tot, test_30_tot, test_1_tot = prepare_data(train_eff_data, norm_glucose_test,norm_ins_test,norm_carb_test, interval,
                                                                        out_60, out_30, out_1)
        train_x_low, train_60_low, train_30_low, train_1_low = prepare_data(train_eff_data, norm_glucose,norm_ins,norm_carb, interval,
                                                                            out_60, out_30, out_1)
        X_train_low = train_x_low

        test_x_low, test_60_low, test_30_low, test_1_low = prepare_data(train_eff_data, norm_glucose_test,norm_ins_test,norm_carb_test, interval,
                                                                        out_60, out_30, out_1)
        X_test_low = test_x_low




        X_train_L = train_x_low
        X_test_L = test_x_low

        Y_train_L = train_60_low
        Y_test_L = test_60_low

        # Convert the data to torch tensors
        X_train_L = torch.from_numpy(X_train_L).float()
        X_test_L = torch.from_numpy(X_test_L).float()
        y_train_L = torch.from_numpy(Y_train_L).float()
        y_test_L = torch.from_numpy(Y_test_L).float()

        # Datasets
        train_dataset_L = torch.utils.data.TensorDataset(X_train_L, y_train_L)
        test_dataset_L = torch.utils.data.TensorDataset(X_test_L, y_test_L)
        # Dataloaders
        train_loader_L = DataLoader(train_dataset_L, batch_size=64, shuffle=True)
        test_loader_L = DataLoader(test_dataset_L, batch_size=64, shuffle=False)
        device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)

# MODEL

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


# Transformer Block Class
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_units, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_units),
            nn.ReLU(),
            nn.Linear(ff_units, d_model)
        )
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x


# Teacher Model (TimeSeriesTransformer)
class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_time_steps, n_features, d_model, n_heads, ff_units, prediction_horizon):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(n_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=n_time_steps)
        self.transformer_block = TransformerBlock(d_model, n_heads, ff_units)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_out = nn.Linear(d_model, prediction_horizon)

    def forward(self, x):
        x = self.embedding(x)  # Linear transformation to d_model
        x = self.positional_encoding(x)
        x = x.permute(1, 0, 2)  # (n_time_steps, batch_size, d_model)
        x = self.transformer_block(x)
        x = x.permute(1, 2, 0)  # (batch_size, d_model, n_time_steps)
        x = self.global_avg_pool(x).squeeze(-1)
        output = self.fc_out(x)
        return output
    
device = "cpu"# torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_time_steps = 36  # Input time window size
n_features = 3  # Number of signals (blood glucose, insulin, meal data)
d_model = 32  # Embedding dimension
n_heads = 4  # Number of attention heads
ff_units = 128  # Feedforward units
prediction_horizon = 12  # Predicting next time step
epochs = 500
alpha = 0.5

teacher_model = TimeSeriesTransformer(n_time_steps, n_features, d_model, n_heads, ff_units, prediction_horizon).to(device)
teacher_model.load_state_dict(torch.load(f'Transformer_weights_{year}_{pid}.pth', map_location=torch.device('cpu')))
teacher_model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    teacher_model, 
    {nn.Linear},  
    dtype=torch.qint8
)

def evaluate_model(model, test_loader, name=""):
    model.eval()
    y_true = []
    y_pred = []
    start_time = time.time()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            preds = model(x_batch)
            y_true.extend(y_batch.numpy())
            y_pred.extend(preds.numpy())
    end_time = time.time()

    y_true = np.array(y_true).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)

    rmse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    inference_time = end_time - start_time

    fname = f"tmp_model_{name}.pth"
    torch.save(model.state_dict(), fname)
    size_kb = os.path.getsize(fname) / 1024
    os.remove(fname)

    return {
        "Precision": "fp32" if name == "Unquantized" else "int8",
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2,
        "Inference Time (s)": inference_time,
        "Size (KB)": size_kb
    }

stats_unquantized = evalstats_unquantized = evaluate_model(teacher_model, test_loader_L, name="Unquantized")
stats_quantized = evaluate_model(quantized_model, test_loader_L, name="Quantized")

rows = ["Precision", "RMSE", "MAE", "R2 Score", "Inference Time (s)", "Size (KB)"]

data = [
    [
        r,
        stats_unquantized[r],
        stats_quantized[r]
    ]
    for r in rows
]

print(tabulate(data, headers=["Stat", "Unquantized", "Quantized"], tablefmt="grid"))
