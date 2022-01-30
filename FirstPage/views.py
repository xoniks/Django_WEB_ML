from django.shortcuts import render
from django.http import HttpResponse
import joblib
import pandas as pd
# Create your views here.

reloadModel = joblib.load(('ML_models/ML_model.joblib'))

def index(request):
    return render(request, 'index.html')

def predictMPG(request):
    print(request)
    if request.method == 'POST':
        temp={}
        temp['cylinders']=request.POST.get('cylinderVal')
        temp['displacement'] = request.POST.get('dispVal')
        temp['horsepower'] = request.POST.get('horseVal')
        temp['weight'] = request.POST.get('weightVal')
        temp['acceleration'] = request.POST.get('accVal')
        temp['model_year'] = request.POST.get('modelVal')



    testData = pd.DataFrame({'x': temp}).transpose()
    scoredVal = int(reloadModel.predict(testData))
    context={'scoredVal':scoredVal}
    return render(request, 'index.html', context)
