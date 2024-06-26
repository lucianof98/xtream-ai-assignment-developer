################################################################################################################################
#### librerie ####

import myPipeline as pp
import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler
#import joblib
#from flask import jsonify, Flask, request

################################################################################################################################
#### sorgente di dati ####

path = "https://raw.githubusercontent.com/xtreamsrl/xtream-ai-assignment-engineer/main/datasets/diamonds/diamonds.csv"


################################################################################################################################
#### creazione dataframe e recall dei vari step della pipeline ####

def pipeline_function():

    df = pd.read_csv(path)

    Pipeline = pp.myPipeline(df)

    df1, label_encoders, scaler = Pipeline.preProcessing(df)
    
    Pipeline.fromCSVtoJSON(df1)

    reg, pred, y_test = Pipeline.modellingLinearReg(df1,'price')

    reg_lin,mse_lin,mae_lin = Pipeline.modelEvaluation(pred,y_test,'linear')

    Pipeline.deploy_model(reg_lin,mse_lin,0.1)

    xgb_opt, xgb_opt_pred, y_test_xbg = Pipeline.modellingXGB(df,'price')

    r2_xgb,mse_xgb,mae_xgb = Pipeline.modelEvaluation(xgb_opt_pred,y_test_xbg,'xgb')

    Pipeline.deploy_model(xgb_opt,mse_xgb,450)


################################################################################################################################
#### automatizzo il processo ####

scheduler = BlockingScheduler()
print('\nAutomated data pipeline is starting...\n')
scheduler.add_job(pipeline_function, 'interval', hours=1/30)  # Run the pipeline every 2 min
scheduler.start()

################################################################################################################################

