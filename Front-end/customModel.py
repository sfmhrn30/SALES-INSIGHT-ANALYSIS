from database import data
from decimal import Decimal, ROUND_DOWN
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64

def truncate(info):
  for i in range(6):
    value=Decimal(info[i])
    info[i]=value.quantize(Decimal('0.01'), rounding=ROUND_DOWN)

  return info

def efficieny(rate):
  eff_fut=[data[9]['efficiency']]
  for i in range(6):
    temp=eff_fut[-1]+eff_fut[-1]*(rate/100)
    eff_fut.append(temp)
  info = truncate(eff_fut[1:])
  return info

def inhouse(eff):
  booked=[1114523, 1002563, 956253 ,965425 ,972356 ,952635]
  inhouse_pred = ( np.array(booked) * np.array(eff) ) / 100
  info = truncate(inhouse_pred)
  return info

def generate_graph(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')
    ax.set_title(title)
    ax.set_xlabel('Financial Year')
    ax.set_ylabel(title.split(' ')[0])

    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)

    img_b64 = base64.b64encode(img.getvalue()).decode('utf8')
    return img_b64