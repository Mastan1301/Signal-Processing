from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import messages
from .models import Channel
from .forms import InputFormAWGN, InputFormBSC, InputFormBEC
import numpy as np
import sys, struct
from PIL import Image
sys.path.append('./main/utils')
import encode, awgnDecode, bscDecode, becDecode
# Create your views here.


def awgn(request): 
    ber = {}   
    if request.method == "POST":
        form = InputFormAWGN(request.POST, request.FILES) 
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

    else:
        img = Image.open("./media/figs/plain.jpeg")
        img.save("./media/figs/input.png")
        img.save("./media/figs/output.png")
        form = InputFormAWGN()

    return render(request,
                  'main/awgn.html',
                  context={"form": InputFormAWGN,
                            "ber": ber})

def bsc(request): 
    ber = {}   
    if request.method == "POST":
        form = InputFormBSC(request.POST, request.FILES) 
        if form.is_valid():
            p = form.cleaned_data.get("p")
            img = form.cleaned_data.get("img")
            algo = form.cleaned_data.get("select")
            img = Image.open(img).convert('L')
            img.save("./media/figs/input.png")

            data = np.array(img, dtype = np.uint8)
            np.save("./media/figs/input.npy", data/255)
            encode.main(data)
            ber = bscDecode.main(p, algo)                       

    else:
        img = Image.open("./media/figs/plain.jpeg")
        img.save("./media/figs/input.png")
        img.save("./media/figs/output.png")
        form = InputFormBSC()

    return render(request,
                  'main/bsc.html',
                  context={"form": InputFormBSC,
                            "ber": ber})


def bec(request): 
    ber = {}   
    if request.method == "POST":
        form = InputFormBEC(request.POST, request.FILES) 
        if form.is_valid():
            p = form.cleaned_data.get("p")
            img = form.cleaned_data.get("img")
            algo = form.cleaned_data.get("select")
            img = Image.open(img).convert('L')
            img.save("./media/figs/input.png")

            data = np.array(img, dtype = np.uint8)
            np.save("./media/figs/input.npy", data/255)
            encode.main(data)
            ber = becDecode.main(p, algo)                       

    else:
        img = Image.open("./media/figs/plain.jpeg")
        img.save("./media/figs/input.png")
        img.save("./media/figs/output.png")
        form = InputFormBEC()

    return render(request,
                  'main/bec.html',
                  context={"form": InputFormBEC,
                            "ber": ber})


def homepage(request):
    return render(request = request,
                  template_name = 'main/home.html',
                  context = {"channels": Channel.objects.all()})
