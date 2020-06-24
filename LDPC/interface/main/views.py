from django.shortcuts import render
from django.http import HttpResponse
from django.contrib import messages
from .models import Channel
from django import forms
from django.core.files.uploadedfile import SimpleUploadedFile
from .forms import InputForm
# Create your views here.

def awgn(request):
    snr = request.POST.get("snr")
    img = request.POST.get("img")

    
    return render(request,
                  'main/awgn.html',
                  context={"form": InputForm()})

def homepage(request):
    return render(request = request,
                  template_name = 'main/home.html',
                  context = {"channels": Channel.objects.all()})
