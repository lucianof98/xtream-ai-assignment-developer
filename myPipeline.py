################################################################################################################################
#### librerie ####
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.linear_model import LinearRegression
import xgboost
import optuna
import time
import joblib

################################################################################################################################

class myPipeline:
    
    def __init__(self,df):
        
        self.df = df
        
    def fromCSVtoJSON(self,df_preprocessed):
        
        timeStamp = time.strftime('%Y%m%d_%H%M%S')
        df_preprocessed.to_json(f'{timeStamp}.json',orient='records',lines=True)
        
        print('\nDataset converted into json file successefully\n')
    
    #########################################################################################################################
    # primo step della pipeline: preprocessing dei dati, pulizia e standardizzazione
    
    def preProcessing(self,df):
        
        #rimuovo colonne non necessarie
        df.drop(columns=['depth', 'table', 'y', 'z'],inplace=True)
        
        # Gestisco i NaN
        df1 = df.fillna('ffill')
        
        # Codifico le varie categorie di variabili all'interno del dataframe
        label_encoders = {}
        
        for column in df1.select_dtypes(include=['object']).columns:
            
            label_encoders[column] = LabelEncoder()
            df1[column] = label_encoders[column].fit_transform(df1[column])
        
        # Normalizzo le feature numeriche rispetto alla deviazione standard e rimuovendo il valor medio
        scaler = StandardScaler()
        df1[df1.select_dtypes(include=['number']).columns] = scaler.fit_transform(df1.select_dtypes(include=['number']))
        
        return df1, label_encoders, scaler
    
    #########################################################################################################################
    # secondo step della pipeline: modelling (addestramento e predizione), a scelta dell'utente il tipo di modello da utilizzare
    
    def modellingLinearReg(self,df,target):
        
        df0 = pd.get_dummies(df, columns=['cut', 'color', 'clarity'], drop_first=True)
        
        x = df0.drop(target,axis=1)
        y = df0[target].dropna()
        
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
        
        #y_train_log = np.log(y_train)
        #print(y_train_log)
        #y_test_log = np.log(y_test)
            
        reg = LinearRegression()
        reg.fit(x_train, y_train)
        pred = reg.predict(x_test)
        #pred = np.exp(pred_log)
        
        return reg, pred, y_test
    
    
    def modellingXGB(self,df,target):
            
            df0 = df.copy()
                
            # è necessario in questo caso trasformare alcune features da ordinarie in categoriche
            df0['cut'] = pd.Categorical(df0['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
            df0['color'] = pd.Categorical(df0['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
            df0['clarity'] = pd.Categorical(df0['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)
            
            # istanzio il modello gradient boosting
            xgb = xgboost.XGBRegressor(enable_categorical=True, random_state=42)
            
            #train and test splitting
            x_train_xbg, x_test_xbg, y_train_xbg, y_test_xbg = train_test_split(df0.drop(columns=target), df0[target], test_size=0.2, random_state=42)
            
            def objective(trial: optuna.trial.Trial) -> float:
                
                    # Definizione dei iperparametri da ottimizzare
                    
                    param = {
                        
                        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
                        'subsample': trial.suggest_categorical('subsample', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
                        'learning_rate': trial.suggest_float('learning_rate', 1e-8, 1.0, log=True),
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 3, 9),
                        'random_state': 42,
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'enable_categorical': True
                        
                    }

                    #train and test splitting
                    x_train, x_val, y_train, y_val = train_test_split(x_train_xbg, y_train_xbg, test_size=0.2, random_state=42)

                    # addestro il modello
                    model = xgboost.XGBRegressor(**param)
                    model.fit(x_train, y_train)

                    # predizione
                    preds = model.predict(x_val)
                    
                        # Calculate MAE
                    mae = mean_absolute_error(y_val, preds)

                    return mae

            study = optuna.create_study(direction='minimize', study_name='Diamonds XGBoost')
            study.optimize(objective, n_trials=100)
            print("Best hyperparameters: ", study.best_params)
                        
            xgb_opt = xgboost.XGBRegressor(**study.best_params, enable_categorical=True, random_state=42)
            xgb_opt.fit(x_train_xbg, y_train_xbg)
            xgb_opt_pred = xgb_opt.predict(x_test_xbg)
            
            return xgb_opt, xgb_opt_pred, y_test_xbg
    
    #########################################################################################################################        
    # terzo step della pipeline: valutazione dell'accuratezza del modello scelto + log dei risultati ottenuti
    
    def modelEvaluation(self,y_pred,y_test,modelClass):
        
        R2 = round(r2_score(y_test,y_pred),3) #calcolo del coefficiente di correlazione / R2 score
        MSE = round(mean_squared_error(y_test,y_pred),3) #calcolo errore quadratico medio
        MAE = round(mean_absolute_error(y_test,y_pred),3)
        
        print(f'Model\t{modelClass}\nR2 score\t{R2}\nMSE\t{MSE}\nMAE\t{MAE}') #stampo a schermo i risultati
        
        timeStamp = time.strftime('%Y%m%d_%H%M%S')
        
        with open(f'{timeStamp}_results.txt','wt') as f:
            
            f.write(f'Model\t{modelClass}\nR2 score\t{R2}\nMSE\t{MSE}\nMAE\t{MAE}')
            
        f.close()
        
        return R2, MSE, MAE
    
    #########################################################################################################################        
    # quarto step della pipeline: sviluppo del modello se l'accuratezza è minore o uguale a una soglia prestabilita
    
    def deploy_model(self,model,mse,threshold):
        
        timeStamp = time.strftime('%Y%m%d_%H%M%S')
        
        if mse <= threshold:
            
            joblib.dump(model, f'{timeStamp}_model.joblib')
            print("\nModel deployed successfully!\n")
            
        else:
            
            print("Model did not meet the performance criteria.")           
