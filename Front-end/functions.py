from decimal import Decimal, ROUND_DOWN
import pickle
import numpy as np
import pandas as pd
from database import data

def trunc(data):
    value=Decimal(data)
    return value.quantize(Decimal('0.01'), rounding=ROUND_DOWN)

def userPredFunc(parts, booked, sales, dispatchpkl, outsourcedpkl, dispatchPolypkl,  outsourcedPolypkl,  effpkl, rate):
    #efficiency

    if rate == None:
        effModel = pickle.load(open('pkl-files/' + effpkl, 'rb'))
        eff = effModel.forecast(steps = int(parts)-24)
        eff = trunc(eff[int(parts)-25+9])
    else:
        eff_fut=[data[9]['efficiency']]
        for i in range(int(parts)-25+9):
            temp=eff_fut[-1]+eff_fut[-1]*(rate/100)
            eff_fut.append(temp)
        eff = trunc(eff_fut[-1])
    

    #inhouse
    inhouseSMH = trunc((booked * eff / 100))

    #dispatch
    if  dispatchPolypkl != None:
        dispatchPolyModel = pickle.load(open('pkl-files/' + dispatchPolypkl, 'rb'))
        dispatchModel = pickle.load(open('pkl-files/' + dispatchpkl, 'rb'))

        dispatchSMH = dispatchModel.predict(dispatchPolyModel.transform(np.array(sales).reshape(-1,1)))
        dispatchSMH = trunc(dispatchSMH[0])

    else:
        dispatchModel = pickle.load(open('pkl-files/' + dispatchpkl, 'rb'))
        if dispatchpkl[:5] == 'ARIMA':
            dispatchSMH = dispatchModel.forecast(steps = int(parts)-24)
            dispatchSMH = trunc(dispatchSMH[int(parts)-25+9])
        else:
            dispatchSMH = dispatchModel.predict(np.array(sales).reshape(-1,1))
            dispatchSMH = trunc(dispatchSMH[0])

    #outsourcd
    if outsourcedPolypkl != None:
        outsourcedPolyModel = pickle.load(open('pkl-files/' + outsourcedPolypkl, 'rb'))
        outsourcedModel = pickle.load(open('pkl-files/' + outsourcedpkl, 'rb'))

        outsourcedSMH = outsourcedModel.predict(outsourcedPolyModel.transform(np.array(inhouseSMH).reshape(-1,1)))
        outsourcedSMH = trunc(outsourcedSMH[0])
    else:
        outsourcedModel = pickle.load(open('pkl-files/' + outsourcedpkl, 'rb'))
        if outsourcedpkl[:5] == 'ARIMA':
            outsourcedSMH = outsourcedModel.forecast(steps = int(parts)-24)
            outsourcedSMH = trunc(outsourcedSMH[int(parts)-25+9])
        else:
            outsourcedSMH = outsourcedModel.predict(np.array(inhouseSMH).reshape(-1,1))
            outsourcedSMH = trunc(outsourcedSMH[0])
            
    #prediction
    predicted = pd.DataFrame({
            'sales' : pd.Series(sales),
            'booked' : pd.Series(booked),
            'efficiency' : pd.Series(eff),
            'dispatch_smh' : pd.Series(dispatchSMH),
            'inhouse_smh' : pd.Series(inhouseSMH),
            'outsourced_smh' : pd.Series(outsourcedSMH)
    })

    return predicted


    
