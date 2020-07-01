from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib import messages
from .models import Channel
from .forms import InputForm
import numpy as np
import sys, struct
from PIL import Image
sys.path.append('./main/utils')
import encode, awgnDecode
# Create your views here.


def awgn(request):    
    if request.method == "POST":
        form = InputForm(request.POST, request.FILES) 
        if form.is_valid():
            snr = form.cleaned_data.get("snr")
            img = form.cleaned_data.get("img")
            algo = form.cleaned_data.get("select")
            img = Image.open(img).convert('L')
            img.save("./media/figs/input.png")

            data = np.array(img, dtype = np.uint8)
            np.save("./media/figs/input.npy", data/255)
            encode.main(data)
            ber = awgnDecode.main(snr, algo)
            return render(request,
                  'main/awgn.html',
                  context={"form": InputForm,
                            "ber" : ber})
                       

    else:
        img = Image.open("./media/figs/plain.jpeg")
        img.save("./media/figs/input.png")
        img.save("./media/figs/output.png")
        form = InputForm()

    return render(request,
                  'main/awgn.html',
                  context={"form": InputForm,
                            })


def homepage(request):
    return render(request = request,
                  template_name = 'main/home.html',
                  context = {"channels": Channel.objects.all()})
